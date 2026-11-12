"""Plan 2 / prep: (a) whole clips -> causal-H tokens (n_steps, 7) ; (b) captions -> CLIP ViT-B/32 text embeddings.
Only captions with whole-clip timing (start=end=0) are used.
Output: $PARTLOCK_DIR/data/gen/{train,val}.pt
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq, tvq272 as B, tvq272c as C

dev = "cuda"
BASE = os.environ.get("PARTLOCK_DIR", ".")   # folder that holds data/, ckpt/, runs/
OUT = os.path.join(BASE, "data", "gen")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

from transformers import CLIPTokenizer, CLIPTextModelWithProjection
name = "openai/clip-vit-base-patch32"
tok = CLIPTokenizer.from_pretrained(name)
clip = CLIPTextModelWithProjection.from_pretrained(name, use_safetensors=True).to(dev).eval()
print("clip loaded %.0fs" % (time.time() - t0), flush=True)

mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
m = C.CHierTok().to(dev)
m.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location=dev))
m.eval()
mdir = B.find_motion_dir()
tdir = os.path.join(B.ROOT, "texts", "texts")


@torch.no_grad()
def codes(z):
    ig = tvq.vq_seq(m.gvq, m.genc(z.permute(0, 2, 1)))[2]
    ip = [tvq.vq_seq(m.vq[p], m.enc[p](z[..., c].permute(0, 2, 1)))[2] for p, c in enumerate(B.PD)]
    return torch.stack([ig] + ip, -1)[0]          # (n, 7)


@torch.no_grad()
def embed(caps):
    t = tok(caps, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    e = clip(**t).text_embeds
    return torch.nn.functional.normalize(e, dim=-1).half().cpu()


for split in ("train", "val"):
    ids = open(os.path.join(B.ROOT, "split", split + ".txt")).read().split()
    rec = []
    skipped = 0
    for i in ids:
        mp, tp = os.path.join(mdir, i + ".npy"), os.path.join(tdir, i + ".txt")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            skipped += 1; continue
        a = np.load(mp).astype(np.float32)
        if a.ndim != 2 or a.shape[1] != 272 or a.shape[0] < 16:
            skipped += 1; continue
        caps = []
        for line in open(tp, encoding="utf-8", errors="ignore"):
            f = line.strip().split("#")
            if len(f) >= 4 and f[0]:
                try:
                    s, e = float(f[2]), float(f[3])
                except ValueError:
                    continue
                if s == 0.0 and e == 0.0:
                    caps.append(f[0])
        if not caps:
            skipped += 1; continue
        a = a[: a.shape[0] // 4 * 4]
        z = (torch.from_numpy(a).to(dev) - mean) / std
        rec.append(dict(id=i, codes=codes(z[None]).short().cpu(), caps=caps, emb=embed(caps)))
        if len(rec) % 2000 == 0:
            print("  %s %d clips  %.0fs" % (split, len(rec), time.time() - t0), flush=True)
    torch.save(rec, os.path.join(OUT, split + ".pt"))
    steps = sum(r["codes"].shape[0] for r in rec)
    print("%s: %d clips (skipped %d), %d steps = %.1f h of motion, %d captions"
          % (split, len(rec), skipped, steps, steps * 4 / 30 / 3600, sum(len(r["caps"]) for r in rec)), flush=True)
print("done %.0fs" % (time.time() - t0))
