"""Plan 2 / generate: 6 prompts x 10 s, plus 1 prompt x 60 s (endless check). CFG s=4 (MoMask form).
Tokens decoded by causal H. Global trajectory recovered as in representation_272 recover_visualize.py.
Writes runs/gen_v1_samples.json (metrics) and gen_viewer_data.json next to this script (viewer)."""
import os, sys, json, time, base64, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq272 as B, tvq272c as C, gen_ar as G

dev = "cuda"
D = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [("walk", "a person walks forward."), ("wave", "a person waves with their right hand."),
           ("kick", "a person kicks with their right leg."), ("jump", "a person jumps up and down."),
           ("dance", "a person dances happily."), ("sit", "a person sits down on a chair.")]
LONG = ("walk60", "a person walks forward.")


def load():
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    name = "openai/clip-vit-base-patch32"
    tok = CLIPTokenizer.from_pretrained(name)
    clip = CLIPTextModelWithProjection.from_pretrained(name, use_safetensors=True).to(dev).eval()
    return tok, clip


@torch.no_grad()
def embed(tok, clip, caps):
    t = tok(caps, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    return F.normalize(clip(**t).text_embeds, dim=-1)


@torch.no_grad()
def generate(g, emb, steps, s=4.0, seed=0):
    """emb (N,512). returns codes (N, steps, 7). context = BOS + whole steps (<= 63 past + current)."""
    torch.manual_seed(seed)
    N = emb.shape[0]
    toks = torch.zeros(N, 0, dtype=torch.long, device=dev)
    cond = emb.float()
    unc = g.null[None].expand(N, -1)
    slots_all = (torch.arange(G.L, device=dev) % 7)
    for k in range(steps * 7):
        cur_step = k // 7
        first = max(0, cur_step - (G.S - 1)) * 7
        ctx = toks[:, first:]
        inp = torch.cat([torch.full((N, 1), G.NV, device=dev, dtype=torch.long), ctx], 1)
        T = inp.shape[1]
        txt = torch.cat([cond, unc])[:, None, :].expand(-1, T, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = g(torch.cat([inp, inp]), slots_all[:T], txt)[:, -1].float()
        lc, lu = lo[:N], lo[N:]
        valid = (g.mask[k % 7] == 0)[None]
        lc = torch.where(valid, lc, torch.zeros_like(lc)); lu = torch.where(valid, lu, torch.zeros_like(lu))
        lg = torch.where(valid, (1 + s) * lc - s * lu, torch.full_like(lc, float("-inf")))
        nxt = torch.multinomial(F.softmax(lg, -1), 1)
        toks = torch.cat([toks, nxt], 1)
    loc = toks.view(N, steps, 7) - torch.tensor(G.OFF, device=dev)
    return loc


def get_codebook(vq):
    for n in ("embed", "codebook", "emb", "weight"):
        if hasattr(vq, n):
            t = getattr(vq, n)
            t = t.weight if hasattr(t, "weight") else t
            return t
    raise AttributeError("codebook not found: " + str([k for k, _ in vq.named_buffers()] + [k for k, _ in vq.named_parameters()]))


@torch.no_grad()
def decode_codes(h, codes, mean, std):
    cbg = get_codebook(h.gvq)
    qg = cbg[codes[..., 0]].permute(0, 2, 1)
    N, n = codes.shape[:2]
    y = torch.zeros(N, n * 4, 272, device=dev)
    for p, c in enumerate(B.PD):
        q = get_codebook(h.vq[p])[codes[..., p + 1]].permute(0, 2, 1)
        y[..., c] = h.dec[p](torch.cat([qg, q], 1)).permute(0, 2, 1)
    return y * std + mean


def to_world(x):
    """x (T,272) raw -> world positions (T,22,3), world rotations (T,22,3,3)"""
    T = x.shape[0]
    Rd = B.rot6d_to_mat(x[:, 2:8])
    tot = [Rd[0]]
    for t in range(1, T):
        tot.append(Rd[t] @ tot[-1])
    inv = torch.stack(tot).transpose(-1, -2)
    P = B.positions(x)
    Pw = (inv[:, None] @ P[..., None]).squeeze(-1)
    v = torch.zeros(T, 3, device=x.device)
    v[:, 0], v[:, 2] = x[:, 0], x[:, 1]
    v[1:] = (inv[:-1] @ v[1:, :, None]).squeeze(-1)
    tr = torch.cumsum(v, 0)
    Pw[..., 0] += tr[:, None, 0]
    Pw[..., 2] += tr[:, None, 2]
    Rl = B.local_rots(x).clone()
    Rl[:, 0] = inv @ Rl[:, 0]
    return Pw, B.global_rots(Rl), tr


def bone_dev_mm(P):
    out = []
    for j, p in enumerate(B.PARENTS):
        if p < 0:
            continue
        l = (P[:, j] - P[:, p]).norm(dim=-1)
        out.append((l - l.median()).abs().mean())
    return float(torch.stack(out).mean()) * 1000


def pack(a, scale=1000):
    v = np.round(np.asarray(a, np.float64) * scale).clip(-32767, 32767).astype(np.int16)
    return base64.b64encode(v.tobytes()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gen_v1")
    a = ap.parse_args()
    t0 = time.time()
    mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
    std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
    Wtr, _ = B.load_split("train", 30000, 16)
    mt = B.Metrics(mean, std, torch.from_numpy(Wtr).to(dev))
    h = C.CHierTok().to(dev)
    h.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location=dev)); h.eval()
    g = G.Gen().to(dev)
    g.load_state_dict(torch.load(os.path.join(G.CKPT, a.tag + ".pt"), map_location=dev)); g.eval()
    tok, clip = load()

    runs = []
    e = embed(tok, clip, [p for _, p in PROMPTS])
    codes = generate(g, e, 75)                       # 75 steps = 300 frames = 10 s
    for i, (k, p) in enumerate(PROMPTS):
        runs.append((k, p, codes[i]))
    print("6 prompts done %.0fs" % (time.time() - t0), flush=True)
    e = embed(tok, clip, [LONG[1]])
    codes = generate(g, e, 450)                      # 60 s
    runs.append((LONG[0], LONG[1], codes[0]))
    print("60 s done %.0fs" % (time.time() - t0), flush=True)

    res, out = {}, dict(fps=30, parents=B.PARENTS, clips=[])
    for k, p, c in runs:
        x = decode_codes(h, c[None], mean, std)[0]
        Pw, Gw, tr = to_world(x)
        r = dict(prompt=p, frames=int(x.shape[0]), consistency_deg=mt.consistency(x[None]),
                 bone_dev_mm=bone_dev_mm(Pw), travel_m=float((tr[-1] - tr[0]).norm()),
                 distinct_global_codes=int(c[:, 0].unique().numel()))
        if k == "walk60":
            r["per_10s"] = [dict(consistency_deg=round(mt.consistency(x[None, i:i + 300]), 2),
                                 bone_dev_mm=round(bone_dev_mm(Pw[i:i + 300]), 2),
                                 pelvis_height_m=round(float(Pw[i:i + 300, 0, 1].mean()), 3))
                            for i in range(0, x.shape[0], 300)]
        res[k] = r
        print(k, json.dumps({kk: (round(v, 2) if isinstance(v, float) else v) for kk, v in r.items() if kk != "per_10s"}), flush=True)
        if "per_10s" in r:
            for i, row in enumerate(r["per_10s"]):
                print("   %2d-%2ds %s" % (i * 10, i * 10 + 10, row), flush=True)
        rel = Pw.clone(); rel[..., 0] -= tr[:, None, 0]; rel[..., 2] -= tr[:, None, 2]
        out["clips"].append(dict(key=k, caption=p, frames=int(x.shape[0]),
                                 p=pack(rel.cpu().numpy().ravel()), r=pack(Gw.reshape(-1, 22, 9).cpu().numpy().ravel()),
                                 tr=[round(float(v), 3) for v in tr[:, [0, 2]].cpu().numpy().ravel()],
                                 m=dict(cons=round(r["consistency_deg"], 2), bone=round(r["bone_dev_mm"], 1),
                                        travel=round(r["travel_m"], 2))))
    json.dump(res, open(os.path.join(G.RUNS, a.tag + "_samples.json"), "w", encoding="utf-8"), indent=1)
    json.dump(out, open(os.path.join(D, "gen_viewer_data.json"), "w", encoding="utf-8"), separators=(",", ":"))
    print("viewer data %.2f MB  total %.0fs" % (os.path.getsize(os.path.join(D, "gen_viewer_data.json")) / 1024**2, time.time() - t0))


if __name__ == "__main__":
    main()
