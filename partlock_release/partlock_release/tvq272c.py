"""
Stage 1: causal version of H (sees only current + past frames), for endless streaming.
Same layers / same parameter count as tvq272.HierTok; only padding moves to the left.
Compared against the already-measured H (runs/t272_v1.json) - H is NOT retrained.

Pass rule (fixed before running):
  pos <= 61 mm, seam <= 12 mm, consistency <= 5.5 deg,
  causality check: changing frames >= t never changes outputs before the token boundary.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vq_mix as V
import tvq
import tvq272 as B


class CConv(nn.Module):
    """Conv1d with all padding on the left -> output i depends on inputs <= i*stride + (stride-1)."""
    def __init__(self, cin, cout, k, s=1, dil=1):
        super().__init__()
        self.pad = (k - 1) * dil - (s - 1)
        self.conv = nn.Conv1d(cin, cout, k, s, 0, dilation=dil)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class CRes(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = CConv(ch, ch, 3, 1, dil)
        self.c2 = nn.Conv1d(ch, ch, 1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


def cenc(cin, ch, d):
    L = [CConv(cin, ch, 3), nn.ReLU()]
    for _ in range(2):
        L += [CConv(ch, ch, 4, 2), CRes(ch, 1), CRes(ch, 3)]
    L += [CConv(ch, d, 3)]
    return nn.Sequential(*L)


def cdec(d, ch, cout):
    L = [CConv(d, ch, 3), nn.ReLU()]
    for _ in range(2):
        L += [CRes(ch, 3), CRes(ch, 1), nn.Upsample(scale_factor=2, mode="nearest"), CConv(ch, ch, 3)]
    L += [nn.ReLU(), CConv(ch, cout, 3)]
    return nn.Sequential(*L)


class CHierTok(nn.Module):
    def __init__(self, kg=4096, kp=128, d=128, ch=128, gch=256):
        super().__init__()
        self.genc = cenc(272, gch, d)
        self.gvq = V.VQ(kg, d)
        self.enc = nn.ModuleList([cenc(len(c), ch, d) for c in B.PD])
        self.vq = nn.ModuleList([V.VQ(kp, d) for _ in B.PD])
        self.dec = nn.ModuleList([cdec(2 * d, ch, len(c)) for c in B.PD])

    def forward(self, x):
        qg, cg, _ = tvq.vq_seq(self.gvq, self.genc(x.permute(0, 2, 1)))
        y = torch.empty_like(x)
        commit = cg
        for p, c in enumerate(B.PD):
            q, cm, _ = tvq.vq_seq(self.vq[p], self.enc[p](x[..., c].permute(0, 2, 1)))
            commit = commit + cm
            y[..., c] = self.dec[p](torch.cat([qg, q], 1)).permute(0, 2, 1)
        return y, commit / 7


@torch.no_grad()
def causality_check(m, Zva):
    """perturb frames from cut onward; outputs before the token block containing cut must not change."""
    m.eval()
    x = Zva[:32].clone()
    y0 = m(x)[0]
    worst = 0.0
    for cut in (8, 20, 33, 47, 60):
        x2 = x.clone()
        x2[:, cut:] = torch.randn_like(x2[:, cut:]) * 3
        y2 = m(x2)[0]
        safe = (cut // 4) * 4          # frames of tokens fully before cut
        worst = max(worst, float((y2[:, :safe] - y0[:, :safe]).abs().max()))
    m.train()
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=500000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tag", default="t272_causal_v1")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
    std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
    Wtr, ntr = B.load_split("train", a.frames, 16)      # same data / seed as t272_v1
    Wva, nva = B.load_split("val", max(40000, a.frames // 10), B.T)
    Xtr = torch.from_numpy(Wtr).to(dev); Xva = torch.from_numpy(Wva).to(dev)
    print("train %d win  val %d win  load %.0fs" % (Xtr.shape[0], Xva.shape[0], time.time() - t0), flush=True)
    Ztr, Zva = (Xtr - mean) / std, (Xva - mean) / std
    mt = B.Metrics(mean, std, Xtr)
    del Xtr

    torch.manual_seed(0)
    m = CHierTok().to(dev)
    print("params causal H %d  (non-causal H %d)" % (B.nparams(m), B.nparams(B.HierTok())), flush=True)
    print("causality (untrained) max diff %.2e" % causality_check(m, Zva), flush=True)
    if a.check_only:
        return
    secs = B.train(m, Ztr, Zva, a.steps, 256, 4e-4, 0.25, 1000, mt)
    r = B.evaluate(m, Zva, mt)
    r.update(params=B.nparams(m), train_seconds=round(secs, 1), causality_maxdiff=causality_check(m, Zva))
    torch.save(m.state_dict(), os.path.join(B.CKPT_DIR, a.tag + "_H.pt"))
    ref = json.load(open(os.path.join(B.OUTDIR, "t272_v1.json"), encoding="utf-8"))["results"]["H"]
    r["pass"] = dict(pos=r["mpjpe_mm"] <= 61, seam=r["seam_bone_mm"] <= 12,
                     consistency=r["consistency_deg"] <= 5.5, causal=r["causality_maxdiff"] == 0.0)
    json.dump(dict(causal_H=r, ref_H=ref), open(os.path.join(B.OUTDIR, a.tag + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("causal H: pos %.2f (ref %.2f) | rot g %.2f (ref %.2f) | seam %.2f (ref %.2f) | consist %.2f (ref %.2f) | causal diff %.1e | %.0fs"
          % (r["mpjpe_mm"], ref["mpjpe_mm"], r["rot_global_deg"], ref["rot_global_deg"], r["seam_bone_mm"],
             ref["seam_bone_mm"], r["consistency_deg"], ref["consistency_deg"], r["causality_maxdiff"], secs), flush=True)
    for pn, v in r["per_part"].items():
        print("    %-6s pos %7.2f  rot %6.2f   (ref %7.2f / %6.2f)" % (pn, v["pos_mm"], v["rot_global_deg"],
              ref["per_part"][pn]["pos_mm"], ref["per_part"][pn]["rot_global_deg"]), flush=True)
    print("PASS" if all(r["pass"].values()) else "FAIL", r["pass"])


if __name__ == "__main__":
    main()
