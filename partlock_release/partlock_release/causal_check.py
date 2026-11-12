"""Plan 1 post-training checks for causal H (no comparisons with other models).
 2) streaming == whole: decode tokens one-by-one-growing vs all at once
 3) whole-clip numbers (val clips passed through in one piece)
 4) locality: swap right-arm tokens, other parts must not change
"""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq, tvq272 as B, tvq272c as C

dev = "cuda"
mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
Wtr, _ = B.load_split("train", 30000, 16)
mt = B.Metrics(mean, std, torch.from_numpy(Wtr).to(dev))
m = C.CHierTok().to(dev)
m.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location=dev))
m.eval()
out = {}

@torch.no_grad()
def codes(z):                     # z (1,T,272) normalized
    qg = tvq.vq_seq(m.gvq, m.genc(z.permute(0, 2, 1)))[0]
    qs = [tvq.vq_seq(m.vq[p], m.enc[p](z[..., c].permute(0, 2, 1)))[0] for p, c in enumerate(B.PD)]
    return qg, qs

@torch.no_grad()
def decode(qg, qs):
    T = qg.shape[-1] * 4
    y = torch.zeros(1, T, 272, device=dev)
    for p, c in enumerate(B.PD):
        y[..., c] = m.dec[p](torch.cat([qg, qs[p]], 1)).permute(0, 2, 1)
    return y

# val clips, whole
mdir = B.find_motion_dir()
ids = open(os.path.join(B.ROOT, "split", "val.txt")).read().split()
clips = []
for i in ids:
    p = os.path.join(mdir, i + ".npy")
    if os.path.exists(p):
        a = np.load(p).astype(np.float32)
        if a.ndim == 2 and a.shape[1] == 272 and a.shape[0] >= 64:
            clips.append(a[: a.shape[0] // 4 * 4])
    if len(clips) >= 200:
        break

# 2) streaming equality on 5 clips
worst = 0.0
for a in clips[:5]:
    z = (torch.from_numpy(a).to(dev) - mean) / std
    qg, qs = codes(z[None])
    full = decode(qg, qs)
    n = qg.shape[-1]
    for k in range(1, n + 1):
        part = decode(qg[..., :k], [q[..., :k] for q in qs])
        worst = max(worst, float((part[:, 4 * k - 4: 4 * k] - full[:, 4 * k - 4: 4 * k]).abs().max()))
out["stream_vs_whole_maxdiff"] = worst
print("2) streaming vs whole max diff: %.2e" % worst, flush=True)

# 3) whole-clip metrics
ys, xs = [], []
with torch.no_grad():
    for a in clips:
        x = torch.from_numpy(a).to(dev)
        y = m(((x - mean) / std)[None])[0][0] * std + mean
        ys.append(y); xs.append(x)
Y, X = torch.cat(ys)[None], torch.cat(xs)[None]
r = mt.all(Y, X)
out["whole_clip"] = r
out["n_clips"] = len(clips); out["n_frames"] = int(X.shape[1])
print("3) whole clips (%d clips, %d frames): pos %.2f mm | rot global %.2f deg | seam %.2f mm | consist %.2f (GT %.2f)"
      % (len(clips), X.shape[1], r["mpjpe_mm"], r["rot_global_deg"], r["seam_bone_mm"], r["consistency_deg"],
         r["consistency_gt_deg"]), flush=True)
for pn, v in r["per_part"].items():
    print("    %-6s pos %7.2f  rot %6.2f" % (pn, v["pos_mm"], v["rot_global_deg"]), flush=True)

# 4) locality: right-arm tokens from another clip
ar = B.PNAMES.index("arm_R")
worst_other, arm_change = 0.0, 0.0
for a, b in zip(clips[:20], clips[20:40]):
    n = min(len(a), len(b))
    za = (torch.from_numpy(a[:n]).to(dev) - mean) / std
    zb = (torch.from_numpy(b[:n]).to(dev) - mean) / std
    qg, qs = codes(za[None]); _, qsb = codes(zb[None])
    y0 = decode(qg, qs)
    qs2 = list(qs); qs2[ar] = qsb[ar]
    y1 = decode(qg, qs2)
    other = [d for p, c in enumerate(B.PD) if p != ar for d in c]
    worst_other = max(worst_other, float((y1[..., other] - y0[..., other]).abs().max()))
    arm_change = max(arm_change, float((y1[..., B.PD[ar]] - y0[..., B.PD[ar]]).abs().max()))
out["locality_other_maxdiff"] = worst_other
out["locality_arm_change"] = arm_change
print("4) swap right arm: other parts max diff %.2e (arm itself changed by %.2f)" % (worst_other, arm_change))
json.dump(out, open(os.path.join(B.OUTDIR, "t272_causal_v1_checks.json"), "w", encoding="utf-8"), indent=2)
