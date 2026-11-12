"""
Experiment A: does an encoder that mixes across parts close the gap between
per-part and whole-body quantization?

rd_joints_full.json (raw-coordinate k-means, in-sample) showed per-part VQ is
far worse than whole-body VQ at matched bitrate (12 bits: 61.17 vs 151.70 mm).
That test had no encoder. MoGenTS claims its joint-axis attention recovers the
cross-part information. This measures how much.

Conditions
  C      whole-body VQ-VAE                         (baseline)
  A      per-part VQ-VAE, no cross-part path at all
  B      per-part VQ-VAE, attention over part tokens in encoder AND decoder
  B_enc  per-part VQ-VAE, attention in encoder only (parts stay independently
         decodable -> the version compositional control would need)

Plus KM_whole / KM_parts: raw-coordinate k-means, re-measured on the SAME
held-out split so every number in this file is comparable.

Data: HumanML3D new_joints, root-relative, 21 joints x 3, metres.
Metric: MPJPE in mm on held-out frames. Loss is MSE on positions, so the
training objective and the metric are the same quantity.
"""
import os
import glob
import json
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
BASE = os.environ.get("PARTLOCK_DIR", ".")   # folder that holds data/, ckpt/, runs/

DEST = os.path.join(BASE, "data", "motion_raw", "humanml3d", "HumanML3D", "new_joints")
OUTDIR = os.path.join(BASE, "runs")

PART6 = {
    "torso": [3, 6, 9],
    "head":  [12, 15],
    "arm_R": [14, 17, 19, 21],
    "arm_L": [13, 16, 18, 20],
    "leg_R": [2, 5, 8, 11],
    "leg_L": [1, 4, 7, 10],
}
PART_NAMES = list(PART6.keys())
COLS = [[j - 1 for j in PART6[p]] for p in PART_NAMES]   # into 0..20
NJ = [len(c) for c in COLS]                              # [3,2,4,4,4,4]
assert sum(NJ) == 21


# ----------------------------------------------------------------- data
def load_split(max_frames, val_frac=0.1, seed=0):
    files = [p for p in glob.glob(os.path.join(DEST, "**", "*.npy"),
                                  recursive=True)
             if not os.path.basename(p).startswith(".")]
    files.sort()
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    # split by FILE so val frames come from unseen motions
    n_val_files = max(1, int(len(files) * val_frac))
    val_files = files[:n_val_files]
    train_files = files[n_val_files:]

    def grab(fs, budget):
        chunks, total, used = [], 0, 0
        for p in fs:
            try:
                a = np.load(p)
            except Exception:
                continue
            if a.ndim != 3 or a.shape[1] != 22 or a.shape[2] != 3:
                continue
            chunks.append(a.astype(np.float32))
            total += a.shape[0]
            used += 1
            if total >= budget:
                break
        X = np.concatenate(chunks, axis=0)
        if X.shape[0] > budget:
            X = X[np.random.default_rng(seed).choice(X.shape[0], budget,
                                                     replace=False)]
        return X, used

    Xtr, ntr = grab(train_files, max_frames)
    Xva, nva = grab(val_files, max(20000, int(max_frames * 0.1)))
    rel = lambda X: (X - X[:, 0:1, :])[:, 1:, :]
    return rel(Xtr), rel(Xva), ntr, nva, len(files)


# ----------------------------------------------------------------- k-means
def assign_chunked(X, C):
    k = C.shape[0]
    chunk = max(256, (1 << 25) // max(k, 1))
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for i in range(0, X.shape[0], chunk):
        d = cn.unsqueeze(0) - 2.0 * (X[i:i + chunk] @ C.T)
        out[i:i + chunk] = d.argmin(1)
    return out


def kmeans_fit(X, k, iters=30, seed=0):
    n = X.shape[0]
    g = torch.Generator(device=X.device).manual_seed(seed)
    C = X[torch.randperm(n, generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = assign_chunked(X, C)
        Cn = torch.zeros_like(C)
        cnt = torch.zeros(k, device=X.device, dtype=X.dtype)
        Cn.index_add_(0, a, X)
        cnt.index_add_(0, a, torch.ones(n, device=X.device, dtype=X.dtype))
        nz = cnt > 0
        Cn[nz] /= cnt[nz].unsqueeze(1)
        ne = int((~nz).sum())
        if ne:
            Cn[~nz] = X[torch.randint(0, n, (ne,), generator=g,
                                      device=X.device)]
        C = Cn
    return C


# ----------------------------------------------------------------- VQ (EMA)
class VQ(nn.Module):
    def __init__(self, k, d, decay=0.99, eps=1e-5):
        super().__init__()
        self.k, self.d, self.decay, self.eps = k, d, decay, eps
        self.register_buffer("emb", torch.randn(k, d) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(k))
        self.register_buffer("emb_avg", self.emb.clone())
        self.register_buffer("inited", torch.zeros(1))

    @torch.no_grad()
    def _init_from(self, z):
        n = z.shape[0]
        idx = torch.randperm(n, device=z.device)[:self.k]
        if idx.numel() < self.k:
            idx = torch.randint(0, n, (self.k,), device=z.device)
        self.emb.copy_(z[idx])
        self.emb_avg.copy_(self.emb)
        self.cluster_size.fill_(1.0)
        self.inited.fill_(1)

    def forward(self, z):
        # z: (N, d)
        if self.training and self.inited.item() == 0:
            self._init_from(z.detach())
        d2 = (z * z).sum(1, keepdim=True) - 2 * z @ self.emb.T \
            + (self.emb * self.emb).sum(1).unsqueeze(0)
        idx = d2.argmin(1)
        zq = self.emb[idx]
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(idx, self.k).to(z.dtype)
                cs = oh.sum(0)
                ea = oh.T @ z.detach()
                self.cluster_size.mul_(self.decay).add_(cs, alpha=1 - self.decay)
                self.emb_avg.mul_(self.decay).add_(ea, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs_n = (self.cluster_size + self.eps) / (n + self.k * self.eps) * n
                self.emb.copy_(self.emb_avg / cs_n.unsqueeze(1))
                # revive dead codes
                dead = self.cluster_size < 0.05
                nd = int(dead.sum())
                if nd:
                    r = torch.randint(0, z.shape[0], (nd,), device=z.device)
                    self.emb[dead] = z.detach()[r]
                    self.emb_avg[dead] = self.emb[dead]
                    self.cluster_size[dead] = 1.0
        commit = F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()          # straight-through
        return zq, commit, idx


# ----------------------------------------------------------------- blocks
def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h),
                         nn.GELU(), nn.Linear(h, o))


class MixBlock(nn.Module):
    """Self-attention over the 6 part tokens + FFN. Pre-norm."""
    def __init__(self, d, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.at = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(),
                                nn.Linear(d * 2, d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.at(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x


class WholeVQVAE(nn.Module):
    """Condition C."""
    def __init__(self, k, d=64, h=256):
        super().__init__()
        self.enc = mlp(63, h, d)
        self.vq = VQ(k, d)
        self.dec = mlp(d, h, 63)
        self.n_codes_per_frame = 1

    def forward(self, x):                      # x: (B,21,3)
        B = x.shape[0]
        z = self.enc(x.reshape(B, -1))
        zq, commit, _ = self.vq(z)
        y = self.dec(zq).reshape(B, 21, 3)
        return y, commit


class PartVQVAE(nn.Module):
    """Conditions A / B / B_enc.  mix_enc / mix_dec toggle the cross-part path."""
    def __init__(self, k, d=32, h=128, dt=128, mix_enc=False, mix_dec=False,
                 n_mix=2, heads=4):
        super().__init__()
        self.d, self.mix_enc, self.mix_dec = d, mix_enc, mix_dec
        self.n_codes_per_frame = 6
        self.e_in = nn.ModuleList([mlp(3 * n, h, dt) for n in NJ])
        self.e_mix = nn.ModuleList([MixBlock(dt, heads) for _ in range(n_mix)]) \
            if mix_enc else None
        self.e_out = nn.ModuleList([nn.Linear(dt, d) for _ in NJ])
        self.part_emb = nn.Parameter(torch.zeros(1, 6, dt))
        self.vq = nn.ModuleList([VQ(k, d) for _ in NJ])
        self.d_in = nn.ModuleList([nn.Linear(d, dt) for _ in NJ])
        self.d_mix = nn.ModuleList([MixBlock(dt, heads) for _ in range(n_mix)]) \
            if mix_dec else None
        self.d_out = nn.ModuleList([mlp(dt, h, 3 * n) for n in NJ])
        self.part_emb_d = nn.Parameter(torch.zeros(1, 6, dt))

    def forward(self, x):                      # x: (B,21,3)
        B = x.shape[0]
        toks = []
        for p, cols in enumerate(COLS):
            toks.append(self.e_in[p](x[:, cols, :].reshape(B, -1)))
        t = torch.stack(toks, 1) + self.part_emb        # (B,6,dt)
        if self.e_mix is not None:
            for blk in self.e_mix:
                t = blk(t)
        commit = 0.0
        zqs = []
        for p in range(6):
            z = self.e_out[p](t[:, p])
            zq, c, _ = self.vq[p](z)
            commit = commit + c
            zqs.append(self.d_in[p](zq))
        u = torch.stack(zqs, 1) + self.part_emb_d
        if self.d_mix is not None:
            for blk in self.d_mix:
                u = blk(u)
        y = x.new_empty(B, 21, 3)
        for p, cols in enumerate(COLS):
            y[:, cols, :] = self.d_out[p](u[:, p]).reshape(B, len(cols), 3)
        return y, commit / 6.0


# ----------------------------------------------------------------- train/eval
def mpjpe_mm(y, x):
    return float(((y - x) ** 2).sum(-1).mean().sqrt()) * 1000


def eval_mm(model, Xva):
    model.eval()
    se, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, Xva.shape[0], 8192):
            xb = Xva[i:i + 8192]
            y, _ = model(xb)
            se += float(((y - xb) ** 2).sum(-1).sum())
            cnt += xb.shape[0] * 21
    model.train()
    return (se / cnt) ** 0.5 * 1000


def train_one(model, Xtr, Xva, epochs, bs, lr, beta, dev, log_every=0):
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    n = Xtr.shape[0]
    steps = max(1, (n // bs)) * epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n - bs + 1, bs):
            xb = Xtr[perm[i:i + bs]]
            y, commit = model(xb)
            rec = F.mse_loss(y, xb)
            loss = rec + beta * commit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
        if log_every and (ep + 1) % log_every == 0:
            print("    ep %3d / %d   step %5d   train_rec %.6f   "
                  "val %7.2f mm" % (ep + 1, epochs, step, float(rec),
                                    eval_mm(model, Xva)), flush=True)
    return eval_mm(model, Xva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--bits", type=int, nargs="+", default=[6, 12, 24, 36])
    ap.add_argument("--km-iters", type=int, default=30)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--tag", default="vq_mix_v1")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    t0 = time.time()

    Xtr_np, Xva_np, ntr, nva, nfiles = load_split(args.frames)
    Xtr = torch.from_numpy(np.ascontiguousarray(Xtr_np)).to(dev)
    Xva = torch.from_numpy(np.ascontiguousarray(Xva_np)).to(dev)
    print("device %s  train %d frames / %d files   val %d frames / %d files   "
          "(files total %d)  load_s %.1f"
          % (dev, Xtr.shape[0], ntr, Xva.shape[0], nva, nfiles,
             time.time() - t0), flush=True)
    print("NOTE: val motions are different FILES from train -> held out.\n",
          flush=True)

    results = []

    def rec(cond, bits, k, mm, secs, extra=None):
        row = dict(cond=cond, bits_per_frame=bits, k=k,
                   codes_per_frame=(1 if cond.endswith("whole") or cond == "C"
                                    else 6),
                   mpjpe_mm=round(mm, 3), seconds=round(secs, 1))
        if extra:
            row.update(extra)
        results.append(row)
        print("%-10s bits=%-3d k=%-6d  val MPJPE = %8.2f mm   (%.1fs)"
              % (cond, bits, k, mm, secs), flush=True)

    # ---- k-means baselines on the SAME split (held-out eval) ----
    print("=== k-means baselines (raw coordinates, held-out eval) ===",
          flush=True)
    tr_whole = Xtr.reshape(Xtr.shape[0], -1).contiguous()
    va_whole = Xva.reshape(Xva.shape[0], -1).contiguous()
    for bits in args.bits:
        kw = 1 << bits
        if kw > 65536:
            print("KM_whole   bits=%-3d  k=%d -> skipped (codebook too large)"
                  % (bits, kw), flush=True)
            continue
        ts = time.time()
        C = kmeans_fit(tr_whole, kw, iters=args.km_iters)
        a = assign_chunked(va_whole, C)
        y = C[a].reshape(Xva.shape)
        rec("KM_whole", bits, kw, mpjpe_mm(y, Xva), time.time() - ts)

    for bits in args.bits:
        kp = 1 << (bits // 6)
        if bits % 6:
            continue
        ts = time.time()
        y = torch.empty_like(Xva)
        for p, cols in enumerate(COLS):
            Ctr = Xtr[:, cols, :].reshape(Xtr.shape[0], -1).contiguous()
            Cva = Xva[:, cols, :].reshape(Xva.shape[0], -1).contiguous()
            C = kmeans_fit(Ctr, kp, iters=args.km_iters)
            a = assign_chunked(Cva, C)
            y[:, cols, :] = C[a].reshape(Xva.shape[0], len(cols), 3)
        rec("KM_parts", bits, kp, mpjpe_mm(y, Xva), time.time() - ts)

    # ---- VQ-VAE conditions ----
    print("\n=== VQ-VAE (%d epochs, bs %d, lr %g, beta %g) ==="
          % (args.epochs, args.bs, args.lr, args.beta), flush=True)
    for bits in args.bits:
        # C: whole-body, k = 2^bits
        kw = 1 << bits
        if kw <= 65536:
            ts = time.time()
            torch.manual_seed(0)
            m = WholeVQVAE(kw)
            print("  -- C  bits=%d k=%d --" % (bits, kw), flush=True)
            mm = train_one(m, Xtr, Xva, args.epochs, args.bs, args.lr,
                           args.beta, dev, args.log_every)
            rec("C", bits, kw, mm, time.time() - ts,
                dict(params=sum(p.numel() for p in m.parameters())))
        else:
            print("C          bits=%-3d  k=%d -> skipped (codebook too large)"
                  % (bits, kw), flush=True)

        if bits % 6:
            continue
        kp = 1 << (bits // 6)
        for cond, me, md in (("A", False, False),
                             ("B", True, True),
                             ("B_enc", True, False)):
            ts = time.time()
            torch.manual_seed(0)
            m = PartVQVAE(kp, mix_enc=me, mix_dec=md)
            print("  -- %s  bits=%d k=%d --" % (cond, bits, kp), flush=True)
            mm = train_one(m, Xtr, Xva, args.epochs, args.bs, args.lr,
                           args.beta, dev, args.log_every)
            rec(cond, bits, kp, mm, time.time() - ts,
                dict(params=sum(p.numel() for p in m.parameters()),
                     mix_enc=me, mix_dec=md))

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, args.tag + ".json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(train_frames=int(Xtr.shape[0]),
                       val_frames=int(Xva.shape[0]),
                       train_files=ntr, val_files=nva, files_total=nfiles,
                       epochs=args.epochs, bs=args.bs, lr=args.lr,
                       beta=args.beta, km_iters=args.km_iters, device=dev,
                       parts=PART6,
                       total_seconds=round(time.time() - t0, 1),
                       results=results), f, ensure_ascii=False, indent=2)
    print("\nwrote", out)
    print("total_seconds %.1f" % (time.time() - t0))


if __name__ == "__main__":
    main()
