"""
Temporal tokenizer comparison at MoSCo's bit budget (54 bits / 4 frames).

  M   MoSCo/ParCo tokenizer: 6 independent part VQ-VAEs (conv1d, down x4),
      codebook 512 x dim 128 each. No cross-part information anywhere.
  H   ours: global code (whole-body encoder, k=4096) + 6 part codes (k=128).
      Part p decoded from [global code, part p code] only -> strictly local.
  W   reference: T2M-GPT style whole-body tokenizer, k=512 (9 bits / 4 frames,
      i.e. LESS information; shown for scale only).

Bits per 4-frame token step: M = 6*9 = 54, H = 12 + 6*7 = 54, W = 9.

Data: HumanML3D new_joints, root-relative 21x3 positions, windows of 64 frames.
NOTE: MoSCo uses 263-dim features; we only have positions, so numbers are not
comparable to MoSCo's paper -- only M vs H under identical conditions.
"""
import os
import sys
import glob
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vq_mix as V   # noqa: E402
BASE = os.environ.get("PARTLOCK_DIR", ".")   # folder that holds data/, ckpt/, runs/

DEST = V.DEST
OUTDIR = os.path.join(BASE, "runs")
CKPT_DIR = os.path.join(BASE, "ckpt")
T = 64

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
           18, 19]
JPART = {0: "root"}
for pn, js in V.PART6.items():
    for j in js:
        JPART[j] = pn
BONES = [(j, p) for j, p in enumerate(PARENTS) if p >= 0]
CROSS = [(j, p) for j, p in BONES if JPART[j] != JPART[p]]


# ----------------------------------------------------------------- data
def load_windows(max_frames, val_frac=0.1, seed=0, stride_tr=16):
    files = [p for p in glob.glob(os.path.join(DEST, "**", "*.npy"),
                                  recursive=True)
             if not os.path.basename(p).startswith(".")]
    files.sort()
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    nv = max(1, int(len(files) * val_frac))
    vf, tf = files[:nv], files[nv:]

    def grab(fs, budget, stride):
        wins, total, used = [], 0, 0
        for p in fs:
            try:
                a = np.load(p).astype(np.float32)
            except Exception:
                continue
            if a.ndim != 3 or a.shape[1] != 22 or a.shape[0] < T:
                continue
            a = (a - a[:, 0:1, :])[:, 1:, :]           # (n,21,3)
            for s in range(0, a.shape[0] - T + 1, stride):
                wins.append(a[s:s + T])
            total += a.shape[0]
            used += 1
            if total >= budget:
                break
        return np.stack(wins), used

    Wtr, ntr = grab(tf, max_frames, stride_tr)
    Wva, nva = grab(vf, max(20000, max_frames // 10), T)
    return Wtr, Wva, ntr, nva


# ----------------------------------------------------------------- conv blocks
class Res(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, 1, dil, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


def enc1d(cin, ch, d):
    L = [nn.Conv1d(cin, ch, 3, 1, 1), nn.ReLU()]
    for _ in range(2):
        L += [nn.Conv1d(ch, ch, 4, 2, 1), Res(ch, 1), Res(ch, 3)]
    L += [nn.Conv1d(ch, d, 3, 1, 1)]
    return nn.Sequential(*L)


def dec1d(d, ch, cout):
    L = [nn.Conv1d(d, ch, 3, 1, 1), nn.ReLU()]
    for _ in range(2):
        L += [Res(ch, 3), Res(ch, 1), nn.Upsample(scale_factor=2, mode="nearest"),
              nn.Conv1d(ch, ch, 3, 1, 1)]
    L += [nn.ReLU(), nn.Conv1d(ch, cout, 3, 1, 1)]
    return nn.Sequential(*L)


def vq_seq(vq, z):
    """z: (B,d,T') -> quantized (B,d,T'), commit, idx (B,T')"""
    B, d, t = z.shape
    zf = z.permute(0, 2, 1).reshape(B * t, d)
    q, c, idx = vq(zf)
    return q.reshape(B, t, d).permute(0, 2, 1), c, idx.reshape(B, t)


def part_in(x, p):        # x (B,T,21,3) -> (B, 3n, T)
    c = V.COLS[p]
    B = x.shape[0]
    return x[:, :, c, :].reshape(B, T, -1).permute(0, 2, 1)


def part_out(y, p, out):  # y (B,3n,T) -> write into out (B,T,21,3)
    c = V.COLS[p]
    B = y.shape[0]
    out[:, :, c, :] = y.permute(0, 2, 1).reshape(B, T, len(c), 3)


class PartTok(nn.Module):           # M
    def __init__(self, k=512, d=128, ch=128):
        super().__init__()
        self.enc = nn.ModuleList([enc1d(3 * n, ch, d) for n in V.NJ])
        self.vq = nn.ModuleList([V.VQ(k, d) for _ in V.NJ])
        self.dec = nn.ModuleList([dec1d(d, ch, 3 * n) for n in V.NJ])
        self.kp = k

    def encode(self, x):
        idx = []
        for p in range(6):
            z = self.enc[p](part_in(x, p))
            idx.append(self._nn(self.vq[p], z))
        return None, torch.stack(idx, -1)            # (B,T',6)

    @staticmethod
    def _nn(vq, z):
        B, d, t = z.shape
        zf = z.permute(0, 2, 1).reshape(B * t, d)
        e = vq.emb
        d2 = (zf * zf).sum(1, keepdim=True) - 2 * zf @ e.T + (e * e).sum(1)
        return d2.argmin(1).reshape(B, t)

    def decode(self, ig, ip):
        B = ip.shape[0]
        out = torch.empty(B, T, 21, 3, device=ip.device)
        for p in range(6):
            q = self.vq[p].emb[ip[..., p]].permute(0, 2, 1)
            part_out(self.dec[p](q), p, out)
        return out

    def forward(self, x):
        B = x.shape[0]
        out = torch.empty_like(x)
        commit = 0.0
        for p in range(6):
            q, c, _ = vq_seq(self.vq[p], self.enc[p](part_in(x, p)))
            commit = commit + c
            part_out(self.dec[p](q), p, out)
        return out, commit / 6


class HierTok(nn.Module):           # H
    def __init__(self, kg=4096, kp=128, d=128, ch=128):
        super().__init__()
        self.genc = enc1d(63, 256, d)
        self.gvq = V.VQ(kg, d)
        self.enc = nn.ModuleList([enc1d(3 * n, ch, d) for n in V.NJ])
        self.vq = nn.ModuleList([V.VQ(kp, d) for _ in V.NJ])
        self.dec = nn.ModuleList([dec1d(2 * d, ch, 3 * n) for n in V.NJ])
        self.kp = kp

    def encode(self, x):
        B = x.shape[0]
        zg = self.genc(x.reshape(B, T, 63).permute(0, 2, 1))
        ig = PartTok._nn(self.gvq, zg)
        ip = torch.stack([PartTok._nn(self.vq[p], self.enc[p](part_in(x, p)))
                          for p in range(6)], -1)
        return ig, ip

    def decode(self, ig, ip):
        B = ip.shape[0]
        qg = self.gvq.emb[ig].permute(0, 2, 1)
        out = torch.empty(B, T, 21, 3, device=ip.device)
        for p in range(6):
            q = self.vq[p].emb[ip[..., p]].permute(0, 2, 1)
            part_out(self.dec[p](torch.cat([qg, q], 1)), p, out)
        return out

    def forward(self, x):
        B = x.shape[0]
        qg, cg, _ = vq_seq(self.gvq,
                           self.genc(x.reshape(B, T, 63).permute(0, 2, 1)))
        out = torch.empty_like(x)
        commit = cg
        for p in range(6):
            q, c, _ = vq_seq(self.vq[p], self.enc[p](part_in(x, p)))
            commit = commit + c
            part_out(self.dec[p](torch.cat([qg, q], 1)), p, out)
        return out, commit / 7


class WholeTok(nn.Module):          # W (reference)
    def __init__(self, k=512, d=128, ch=256):
        super().__init__()
        self.enc = enc1d(63, ch, d)
        self.vq = V.VQ(k, d)
        self.dec = dec1d(d, ch, 63)

    def forward(self, x):
        B = x.shape[0]
        q, c, _ = vq_seq(self.vq, self.enc(x.reshape(B, T, 63).permute(0, 2, 1)))
        y = self.dec(q).permute(0, 2, 1).reshape(B, T, 21, 3)
        return y, c


# ----------------------------------------------------------------- metrics
def full22(y):
    B, t = y.shape[:2]
    z = torch.zeros(B, t, 22, 3, device=y.device)
    z[:, :, 1:] = y
    return z


def bone_err(y, x, bones):
    a, b = full22(y), full22(x)
    e = []
    for j, p in bones:
        e.append(((a[:, :, j] - a[:, :, p]).norm(dim=-1)
                  - (b[:, :, j] - b[:, :, p]).norm(dim=-1)).abs())
    return float(torch.stack(e, -1).mean()) * 1000


def mpjpe(y, x):
    return float(((y - x) ** 2).sum(-1).mean().sqrt()) * 1000


@torch.no_grad()
def evaluate(m, Wva):
    m.eval()
    ys = []
    for i in range(0, Wva.shape[0], 256):
        ys.append(m(Wva[i:i + 256])[0])
    y = torch.cat(ys)
    m.train()
    return y


def train(m, Wtr, Wva, steps, bs, lr, beta, log_every):
    dev = Wtr.device
    m.to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                              pct_start=0.05)
    n = Wtr.shape[0]
    t0 = time.time()
    for s in range(1, steps + 1):
        xb = Wtr[torch.randint(0, n, (bs,), device=dev)]
        y, commit = m(xb)
        loss = F.mse_loss(y, xb) + beta * commit
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sch.step()
        if s == 20:
            torch.cuda.synchronize()
            sps = 20 / (time.time() - t0)
            print("    speed %.1f step/s -> est %.0f s for %d steps"
                  % (sps, steps / sps, steps), flush=True)
        if log_every and s % log_every == 0:
            y = evaluate(m, Wva)
            print("    step %5d  val MPJPE %7.2f mm  (%.0fs)"
                  % (s, mpjpe(y, Wva), time.time() - t0), flush=True)
    return time.time() - t0


@torch.no_grad()
def swap_test(m, Wva, g, rows, name):
    m.eval()
    ig, ip = m.encode(Wva)
    y0 = m.decode(ig, ip)
    base = dict(all=bone_err(y0, Wva, BONES), cross=bone_err(y0, Wva, CROSS))
    print("  [%s] no swap: MPJPE %.2f  bone %.2f  cross-part bone %.2f mm"
          % (name, mpjpe(y0, Wva), base["all"], base["cross"]), flush=True)
    tpos = ip.shape[1] // 2
    fr = slice(tpos * 4 - 4, tpos * 4 + 8)      # frames influenced by the token
    for p, pn in enumerate(V.PART_NAMES):
        sw = ip.clone()
        off = torch.randint(1, m.kp, (sw.shape[0],), generator=g, device=sw.device)
        sw[:, tpos, p] = (sw[:, tpos, p] + off) % m.kp
        y1 = m.decode(ig, sw)
        d2 = ((y1 - y0) ** 2).sum(-1)[:, fr]
        inside = torch.zeros(21, dtype=torch.bool, device=sw.device)
        inside[V.COLS[p]] = True
        r = dict(model=name, part=pn,
                 move_in_mm=round(float(d2[..., inside].mean().sqrt()) * 1000, 2),
                 move_out_mm=round(float(d2[..., ~inside].mean().sqrt()) * 1000, 2),
                 bone_mm=round(bone_err(y1[:, fr], Wva[:, fr], BONES), 2),
                 cross_bone_mm=round(bone_err(y1[:, fr], Wva[:, fr], CROSS), 2),
                 bone_noswap_mm=round(bone_err(y0[:, fr], Wva[:, fr], BONES), 2),
                 cross_noswap_mm=round(bone_err(y0[:, fr], Wva[:, fr], CROSS), 2))
        rows.append(r)
        print("  [%s] %-6s in %7.2f  out %6.2f  bone %6.2f (no swap %6.2f)  "
              "cross %6.2f (no swap %6.2f)" % (
                  name, pn, r["move_in_mm"], r["move_out_mm"], r["bone_mm"],
                  r["bone_noswap_mm"], r["cross_bone_mm"], r["cross_noswap_mm"]),
              flush=True)
    m.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--models", nargs="+", default=["M", "H", "W"])
    ap.add_argument("--m-ch", type=int, default=128)
    ap.add_argument("--tag", default="tvq_v1")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    Wtr_np, Wva_np, ntr, nva = load_windows(args.frames)
    Wtr = torch.from_numpy(Wtr_np).to(dev)
    Wva = torch.from_numpy(Wva_np).to(dev)
    print("train windows %d (%d files)  val windows %d (%d files)  T=%d  "
          "load_s %.1f" % (Wtr.shape[0], ntr, Wva.shape[0], nva, T,
                           time.time() - t0), flush=True)
    print("cross-part bones:", [(V.JOINT_NAMES[j] if hasattr(V, "JOINT_NAMES")
                                 else j, p) for j, p in CROSS], flush=True)

    makers = {"M": lambda: PartTok(ch=args.m_ch), "H": lambda: HierTok(),
              "W": lambda: WholeTok()}
    bits = {"M": 54, "H": 54, "W": 9}
    results, swaps = [], []
    g = torch.Generator(device=dev).manual_seed(1)
    for name in args.models:
        torch.manual_seed(0)
        m = makers[name]()
        npar = sum(p.numel() for p in m.parameters())
        print("\n=== %s  bits/4frames=%d  params=%.2fM ===" %
              (name, bits[name], npar / 1e6), flush=True)
        secs = train(m, Wtr, Wva, args.steps, args.bs, args.lr, args.beta,
                     args.log_every)
        y = evaluate(m, Wva)
        r = dict(model=name, bits_per_4frames=bits[name], params=npar,
                 val_mpjpe_mm=round(mpjpe(y, Wva), 3),
                 bone_mm=round(bone_err(y, Wva, BONES), 3),
                 cross_bone_mm=round(bone_err(y, Wva, CROSS), 3),
                 train_seconds=round(secs, 1))
        results.append(r)
        print("%s  val MPJPE %.2f mm  bone %.2f  cross-part bone %.2f  (%.0fs)"
              % (name, r["val_mpjpe_mm"], r["bone_mm"], r["cross_bone_mm"], secs),
              flush=True)
        if name in ("M", "H"):
            torch.save(m.state_dict(), os.path.join(CKPT_DIR, "%s_%s.pt"
                                                    % (args.tag, name)))
            swap_test(m, Wva, g, swaps, name)
        del m
        torch.cuda.empty_cache()

    out = os.path.join(OUTDIR, args.tag + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(args=vars(args), results=results, swaps=swaps,
                       cross_bones=CROSS,
                       total_seconds=round(time.time() - t0, 1)),
                  f, ensure_ascii=False, indent=2)
    print("\nwrote", out, " total_s %.1f" % (time.time() - t0))


if __name__ == "__main__":
    main()
