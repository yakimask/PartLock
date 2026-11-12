"""
Tokenizer comparison on the 272-dim HumanML3D representation (with rotations).

Layout (verified from representation_272.py):
  [0:2]    root xz velocity (heading removed)
  [2:8]    heading angular velocity, 6D
  [8:74]   local joint positions (22x3, heading removed, xz at origin)
  [74:140] local joint velocities (22x3)
  [140:272] SMPL parent-relative rotations (22x6D, first two ROWS of the matrix),
            root (joint 0) with heading removed.
30 fps. Splits from split/{train,val}.txt.

Per-joint feature = pos3 + vel3 + rot6 = 12 dims. Root extras (8) go to the torso part.
  torso: joints 0,3,6,9 (+8 root dims) = 56   head: 12,15 = 24
  arm_R / arm_L / leg_R / leg_L: 4 joints = 48 each        total 272

  M  6 independent part VQ-VAEs, k=512 each              -> 54 bits / 4 frames
  H  global (all 272 ch) k=4096 + 6 part codes k=128,
     part p decoded from [global, part p] only            -> 54 bits / 4 frames
Params matched by widening M's channels (--m-ch).
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vq_mix as V   # noqa: E402  (VQ)
import tvq           # noqa: E402  (enc1d, dec1d, vq_seq)
BASE = os.environ.get("PARTLOCK_DIR", ".")   # folder that holds data/, ckpt/, runs/

ROOT = os.path.join(BASE, "data", "motion_raw", "h272")
OUTDIR = os.path.join(BASE, "runs")
CKPT_DIR = os.path.join(BASE, "ckpt")
T = 64
PARENTS = tvq.PARENTS
PART_J = {"torso": [0, 3, 6, 9], "head": [12, 15], "arm_R": [14, 17, 19, 21],
          "arm_L": [13, 16, 18, 20], "leg_R": [2, 5, 8, 11], "leg_L": [1, 4, 7, 10]}
PNAMES = list(PART_J)
CROSS = tvq.CROSS


def joint_dims(j):
    return (list(range(8 + 3 * j, 11 + 3 * j)) + list(range(74 + 3 * j, 77 + 3 * j))
            + list(range(140 + 6 * j, 146 + 6 * j)))


PART_DIMS = {}
for p, js in PART_J.items():
    d = [] if p != "torso" else list(range(8))
    for j in js:
        d += joint_dims(j)
    PART_DIMS[p] = d
assert sorted(sum(PART_DIMS.values(), [])) == list(range(272))
PD = [PART_DIMS[p] for p in PNAMES]


# ------------------------------------------------------------------ data
def find_motion_dir():
    for dp, dn, fn in os.walk(os.path.join(ROOT, "motion_data")):
        if sum(f.endswith(".npy") for f in fn) > 1000:
            return dp
    raise SystemExit("motion npy dir not found under " + ROOT)


def load_split(name, budget, stride, seed=0):
    mdir = find_motion_dir()
    ids = open(os.path.join(ROOT, "split", name + ".txt")).read().split()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    wins, total, used = [], 0, 0
    for i in ids:
        p = os.path.join(mdir, i + ".npy")
        if not os.path.exists(p):
            continue
        a = np.load(p).astype(np.float32)
        if a.ndim != 2 or a.shape[1] != 272 or a.shape[0] < T:
            continue
        for s in range(0, a.shape[0] - T + 1, stride):
            wins.append(a[s:s + T])
        total += a.shape[0]
        used += 1
        if total >= budget:
            break
    return np.stack(wins), used


# ------------------------------------------------------------------ rotations
def rot6d_to_mat(x):
    """x (...,6) = first two ROWS -> (...,3,3) rows b1,b2,b3 (pytorch3d convention)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def local_rots(x):           # x (B,T,272) raw -> (B,T,22,3,3)
    return rot6d_to_mat(x[..., 140:272].reshape(*x.shape[:-1], 22, 6))


def global_rots(R):          # chain parent-relative rotations
    G = [None] * 22
    for j in range(22):
        p = PARENTS[j]
        G[j] = R[..., j, :, :] if p < 0 else G[p] @ R[..., j, :, :]
    return torch.stack(G, dim=-3)


def geodesic_deg(A, B):
    tr = (A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)))


def positions(x):            # (B,T,22,3)
    return x[..., 8:74].reshape(*x.shape[:-1], 22, 3)


# ------------------------------------------------------------------ models
class PartTok(nn.Module):
    def __init__(self, k=512, d=128, ch=128):
        super().__init__()
        self.enc = nn.ModuleList([tvq.enc1d(len(c), ch, d) for c in PD])
        self.vq = nn.ModuleList([V.VQ(k, d) for _ in PD])
        self.dec = nn.ModuleList([tvq.dec1d(d, ch, len(c)) for c in PD])
        self.kp = k

    def codes(self, x):
        out = []
        for p, c in enumerate(PD):
            out.append(tvq.vq_seq(self.vq[p], self.enc[p](x[..., c].permute(0, 2, 1))))
        return out

    def forward(self, x, swap=None):
        y = torch.empty_like(x)
        commit = 0.0
        for p, (q, cm, _) in enumerate(self.codes(x)):
            commit = commit + cm
            y[..., PD[p]] = self.dec[p](q).permute(0, 2, 1)
        return y, commit / 6


class HierTok(nn.Module):
    def __init__(self, kg=4096, kp=128, d=128, ch=128, gch=256):
        super().__init__()
        self.genc = tvq.enc1d(272, gch, d)
        self.gvq = V.VQ(kg, d)
        self.enc = nn.ModuleList([tvq.enc1d(len(c), ch, d) for c in PD])
        self.vq = nn.ModuleList([V.VQ(kp, d) for _ in PD])
        self.dec = nn.ModuleList([tvq.dec1d(2 * d, ch, len(c)) for c in PD])
        self.kp = kp

    def forward(self, x):
        qg, cg, _ = tvq.vq_seq(self.gvq, self.genc(x.permute(0, 2, 1)))
        y = torch.empty_like(x)
        commit = cg
        for p, c in enumerate(PD):
            q, cm, _ = tvq.vq_seq(self.vq[p], self.enc[p](x[..., c].permute(0, 2, 1)))
            commit = commit + cm
            y[..., c] = self.dec[p](torch.cat([qg, q], 1)).permute(0, 2, 1)
        return y, commit / 7


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ metrics
class Metrics:
    def __init__(self, mean, std, Xtr_raw):
        self.mean, self.std = mean, std
        # rest bone direction in parent frame, estimated from GT train data
        with torch.no_grad():
            x = Xtr_raw[: min(2000, Xtr_raw.shape[0])]
            G = global_rots(local_rots(x))
            P = positions(x)
            rest = torch.zeros(22, 3, device=x.device)
            for j, p in enumerate(PARENTS):
                if p < 0:
                    continue
                v = P[..., j, :] - P[..., p, :]
                loc = (G[..., p, :, :].transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
                rest[j] = F.normalize(F.normalize(loc, dim=-1).reshape(-1, 3).mean(0), dim=0)
            self.rest = rest

    def denorm(self, z):
        return z * self.std + self.mean

    @torch.no_grad()
    def all(self, y_raw, x_raw):
        Py, Px = positions(y_raw), positions(x_raw)
        out = {}
        out["mpjpe_mm"] = float(((Py - Px) ** 2).sum(-1).mean().sqrt()) * 1000
        Ry, Rx = local_rots(y_raw), local_rots(x_raw)
        Gy, Gx = global_rots(Ry), global_rots(Rx)
        out["rot_local_deg"] = float(geodesic_deg(Ry, Rx).mean())
        out["rot_global_deg"] = float(geodesic_deg(Gy, Gx).mean())
        per = {}
        for pn, js in PART_J.items():
            per[pn] = dict(
                pos_mm=round(float(((Py[..., js, :] - Px[..., js, :]) ** 2).sum(-1).mean().sqrt()) * 1000, 2),
                rot_global_deg=round(float(geodesic_deg(Gy[..., js, :, :], Gx[..., js, :, :]).mean()), 2))
        out["per_part"] = per
        se = []
        for j, p in CROSS:
            se.append(((Py[..., j, :] - Py[..., p, :]).norm(dim=-1)
                       - (Px[..., j, :] - Px[..., p, :]).norm(dim=-1)).abs().mean())
        out["seam_bone_mm"] = float(torch.stack(se).mean()) * 1000
        out["consistency_deg"] = self.consistency(y_raw)
        out["consistency_gt_deg"] = self.consistency(x_raw)
        return out

    def consistency(self, x_raw):
        """angle between bone dir from positions and bone dir predicted by rotations"""
        G = global_rots(local_rots(x_raw))
        P = positions(x_raw)
        angs = []
        for j, p in enumerate(PARENTS):
            if p < 0:
                continue
            dpos = F.normalize(P[..., j, :] - P[..., p, :], dim=-1)
            drot = (G[..., p, :, :] @ self.rest[j].view(3, 1)).squeeze(-1)
            angs.append(torch.rad2deg(torch.acos((dpos * drot).sum(-1).clamp(-1, 1))).mean())
        return float(torch.stack(angs).mean())


# ------------------------------------------------------------------ train
def train(m, Ztr, Zva, steps, bs, lr, beta, log_every, mt):
    dev = Ztr.device
    m.to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for s in range(1, steps + 1):
        xb = Ztr[torch.randint(0, Ztr.shape[0], (bs,), device=dev)]
        y, commit = m(xb)
        loss = F.mse_loss(y, xb) + beta * commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if s == 20:
            torch.cuda.synchronize()
            sps = 20 / (time.time() - t0)
            print("    speed %.1f step/s -> est %.0f s" % (sps, steps / sps), flush=True)
        if log_every and s % log_every == 0:
            r = evaluate(m, Zva, mt)
            print("    step %5d  pos %6.2f mm  rot(global) %5.2f deg  (%.0fs)"
                  % (s, r["mpjpe_mm"], r["rot_global_deg"], time.time() - t0), flush=True)
    return time.time() - t0


@torch.no_grad()
def evaluate(m, Zva, mt, return_y=False):
    m.eval()
    ys = [m(Zva[i:i + 256])[0] for i in range(0, Zva.shape[0], 256)]
    m.train()
    y = torch.cat(ys)
    r = mt.all(mt.denorm(y), mt.denorm(Zva))
    return (r, y) if return_y else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=["H", "M"])
    ap.add_argument("--m-ch", type=int, default=0, help="0 = auto-match H params")
    ap.add_argument("--tag", default="t272_v1")
    ap.add_argument("--probe", action="store_true", help="only print data/param info")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    mean = torch.from_numpy(np.load(os.path.join(ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
    std = torch.from_numpy(np.load(os.path.join(ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
    Wtr, ntr = load_split("train", args.frames, 16)
    Wva, nva = load_split("val", max(40000, args.frames // 10), T)
    Xtr = torch.from_numpy(Wtr).to(dev)
    Xva = torch.from_numpy(Wva).to(dev)
    print("train windows %d (%d files)  val windows %d (%d files)  load %.0fs"
          % (Xtr.shape[0], ntr, Xva.shape[0], nva, time.time() - t0), flush=True)
    Ztr, Zva = (Xtr - mean) / std, (Xva - mean) / std
    mt = Metrics(mean, std, Xtr)
    print("GT self-consistency (deg): %.3f" % mt.consistency(Xva[:256]), flush=True)
    R = local_rots(Xva[:64])
    orth = (R @ R.transpose(-1, -2) - torch.eye(3, device=dev)).abs().max()
    print("6D->matrix orthonormality max err: %.2e" % float(orth), flush=True)
    del Xtr

    hp = nparams(HierTok())
    mch = args.m_ch
    if mch == 0:
        best = None
        for ch in range(128, 260, 2):
            d = abs(nparams(PartTok(ch=ch)) - hp)
            if best is None or d < best[0]:
                best = (d, ch)
        mch = best[1]
    print("params H %d  M(ch=%d) %d  diff %+.2f%%" % (hp, mch, nparams(PartTok(ch=mch)),
          (nparams(PartTok(ch=mch)) - hp) / hp * 100), flush=True)
    if args.probe:
        return

    results = {}
    for name in args.models:
        torch.manual_seed(0)
        m = HierTok() if name == "H" else PartTok(ch=mch)
        print("\n=== %s  params %d ===" % (name, nparams(m)), flush=True)
        secs = train(m, Ztr, Zva, args.steps, args.bs, args.lr, args.beta, args.log_every, mt)
        r = evaluate(m, Zva, mt)
        r.update(params=nparams(m), train_seconds=round(secs, 1))
        results[name] = r
        torch.save(m.state_dict(), os.path.join(CKPT_DIR, "%s_%s.pt" % (args.tag, name)))
        print("%s: pos %.2f mm | rot local %.2f / global %.2f deg | seam %.2f mm | "
              "consistency %.2f deg (GT %.2f) | %.0fs" % (
                  name, r["mpjpe_mm"], r["rot_local_deg"], r["rot_global_deg"],
                  r["seam_bone_mm"], r["consistency_deg"], r["consistency_gt_deg"], secs), flush=True)
        for pn, v in r["per_part"].items():
            print("    %-6s pos %7.2f mm  rot %6.2f deg" % (pn, v["pos_mm"], v["rot_global_deg"]), flush=True)
        del m
        torch.cuda.empty_cache()
        with open(os.path.join(OUTDIR, args.tag + ".json"), "w", encoding="utf-8") as f:
            json.dump(dict(args=vars(args), m_ch=mch, results=results,
                           total_seconds=round(time.time() - t0, 1)), f, ensure_ascii=False, indent=2)
    print("done total %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
