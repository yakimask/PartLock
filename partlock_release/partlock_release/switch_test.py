"""Plan 4 (early): switch the text every 6 s without resetting. gen_v1 best (step 9000), CFG s=4, seed 0.
Rules fixed beforehand:
 J1 seam: max joint speed within +-0.5 s of a switch / 99th pct speed inside segments (excluding those windows), per part <= 1.5, all 9 switches
 J2 follow: the new action's sign appears within 2 s after the switch, >= 7 of 9
 J3 per segment: consistency <= 8 deg, bone dev <= 20 mm
Writes runs/switch_v1.json and <tag>_viewer_data.json next to this script"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq272 as B, tvq272c as C, gen_ar as G, gen_sample as GS

dev = "cuda"
D = os.path.dirname(os.path.abspath(__file__))
TXT = {"walk": "a person walks forward.", "jump": "a person jumps up and down.",
       "sit": "a person sits down on a chair.", "wave": "a person waves with their right hand.",
       "kick": "a person kicks with their right leg."}
SEQ = [("S1", ["walk", "jump", "walk", "sit"]), ("S2", ["walk", "wave", "walk", "kick"]),
       ("S3", ["jump", "walk", "wave", "sit"])]
SEG = 45                     # steps per segment = 180 frames = 6 s
FPS = 30


@torch.no_grad()
def generate_sched(g, embs, seg, s=4.0, seed=0, reset=False, keep=0, periodic=False):
    """periodic=True: the prompt is re-sent at every segment boundary, so every boundary counts as a switch."""
    """embs (N, nseg, 512): text for each segment. reset=False: context never reset.
    reset=True: at each text switch the generator forgets everything older than `keep` steps before the switch."""
    torch.manual_seed(seed)
    N, nseg = embs.shape[:2]
    steps = nseg * seg
    starts = [j for j in range(nseg) if j == 0 or periodic or not torch.equal(embs[:, j], embs[:, j - 1])]
    toks = torch.zeros(N, 0, dtype=torch.long, device=dev)
    unc = g.null[None].expand(N, -1)
    slots_all = torch.arange(G.L, device=dev) % 7
    for k in range(steps * 7):
        cur = k // 7
        cond = embs[:, min(cur // seg, nseg - 1)].float()
        first = max(0, cur - (G.S - 1)) * 7
        if reset:
            st = max(j for j in starts if j <= cur // seg) * seg      # start of the current text (only real changes count)
            first = max(first, max(0, st - (keep if st > 0 else 0)) * 7)
        inp = torch.cat([torch.full((N, 1), G.NV, device=dev, dtype=torch.long), toks[:, first:]], 1)
        T = inp.shape[1]
        txt = torch.cat([cond, unc])[:, None, :].expand(-1, T, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = g(torch.cat([inp, inp]), slots_all[:T], txt)[:, -1].float()
        lc, lu = lo[:N], lo[N:]
        valid = (g.mask[k % 7] == 0)[None]
        lc = torch.where(valid, lc, torch.zeros_like(lc)); lu = torch.where(valid, lu, torch.zeros_like(lu))
        lg = torch.where(valid, (1 + s) * lc - s * lu, torch.full_like(lc, float("-inf")))
        toks = torch.cat([toks, torch.multinomial(F.softmax(lg, -1), 1)], 1)
    return toks.view(N, steps, 7) - torch.tensor(G.OFF, device=dev)


def signs(Pw, tr):
    """per-frame booleans for each action sign. Pw (T,22,3) world, y up."""
    P = Pw.cpu().numpy(); T = P.shape[0]
    feet = P[:, [7, 8, 10, 11], 1]
    floor = np.percentile(feet.min(1), 5)
    t = tr.cpu().numpy()
    sp = np.zeros(T); sp[1:] = np.linalg.norm(np.diff(t, axis=0), axis=1) * FPS
    sp1 = np.convolve(sp, np.ones(FPS) / FPS, mode="same")
    return {"walk": sp1 > 0.5,
            "jump": feet.min(1) > floor + 0.08,
            "wave": P[:, 21, 1] > P[:, 17, 1] + 0.05,
            "kick": P[:, 8, 1] > floor + 0.30,
            "sit": P[:, 0, 1] < 0.6}, float(floor)


def main():
    t0 = time.time()
    mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
    std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
    Wtr, _ = B.load_split("train", 30000, 16)
    mt = B.Metrics(mean, std, torch.from_numpy(Wtr).to(dev))
    h = C.CHierTok().to(dev)
    h.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location=dev)); h.eval()
    g = G.Gen().to(dev)
    g.load_state_dict(torch.load(os.path.join(G.CKPT, "gen_v1.pt"), map_location=dev)); g.eval()
    tok, clip = GS.load()
    keys = list(TXT)
    E = GS.embed(tok, clip, [TXT[k] for k in keys])
    embs = torch.stack([torch.stack([E[keys.index(a)] for a in seq]) for _, seq in SEQ])
    # --keep1s: keep last 8 steps (1.07 s); --keep1step: keep last 1 step (4 frames); --forget: keep nothing
    KEEP = 8 if "--keep1s" in sys.argv else 1 if "--keep1step" in sys.argv else 0
    RESET = any(f in sys.argv for f in ("--keep1s", "--keep1step", "--forget"))
    TAG = {8: "switch_keep1s_v1", 1: "switch_keep1step_v1", 0: "switch_forget_v1"}[KEEP] if RESET else "switch_v1"
    codes = generate_sched(g, embs, SEG, reset=RESET, keep=KEEP)
    print("generated %.0fs" % (time.time() - t0), flush=True)

    res, out = {}, dict(fps=FPS, parents=B.PARENTS, clips=[])
    j1_ok, j2_hits, j2_n, j3_ok = True, 0, 0, True
    for i, (name, seq) in enumerate(SEQ):
        x = GS.decode_codes(h, codes[i][None], mean, std)[0]
        Pw, Gw, tr = GS.to_world(x)
        T = x.shape[0]; F_ = SEG * 4
        V = torch.zeros(T, 22, device=dev); V[1:] = (Pw[1:] - Pw[:-1]).norm(dim=-1) * FPS      # joint speed m/s
        sw = [F_ * k for k in range(1, len(seq))]
        win = torch.zeros(T, dtype=torch.bool, device=dev)
        for f in sw:
            win[max(0, f - 15):f + 15] = True
        sg, floor = signs(Pw, tr)
        segs, sws = [], []
        for k, a in enumerate(seq):
            a0, a1 = k * F_, (k + 1) * F_
            segs.append(dict(text=a, consistency_deg=round(mt.consistency(x[None, a0:a1]), 2),
                             bone_dev_mm=round(GS.bone_dev_mm(Pw[a0:a1]), 2),
                             sign_frac=round(float(sg[a][a0:a1].mean()), 2),
                             pelvis_h=round(float((Pw[a0:a1, 0, 1] - Pw[a0:a1][:, [7, 8, 10, 11], 1].min(1).values).mean()), 2)))
            j3_ok &= segs[-1]["consistency_deg"] <= 8 and segs[-1]["bone_dev_mm"] <= 20
        for k, f in enumerate(sw):
            per = {}
            for p, js in B.PART_J.items():
                ref = torch.quantile(V[~win][:, js].flatten(), 0.99)
                per[p] = round(float(V[max(0, f - 15):f + 15, js].max() / ref), 2)
            new = seq[k + 1]
            hit = np.nonzero(sg[new][f:f + 2 * FPS])[0]
            lat = round(float(hit[0]) / FPS, 2) if len(hit) else None
            sws.append(dict(frm=seq[k], to=new, sec=f / FPS, seam_ratio=per, seam_max=max(per.values()), sign_after_s=lat))
            j1_ok &= max(per.values()) <= 1.5
            j2_n += 1; j2_hits += lat is not None
        res[name] = dict(seq=seq, floor_m=round(floor, 3), segments=segs, switches=sws)
        print(name, json.dumps(res[name]), flush=True)
        rel = Pw.clone(); rel[..., 0] -= tr[:, None, 0]; rel[..., 2] -= tr[:, None, 2]
        out["clips"].append(dict(key=name, caption=" → ".join(seq), frames=int(T),
                                 seg=[dict(f=k * F_, text=TXT[a], key=a) for k, a in enumerate(seq)],
                                 p=GS.pack(rel.cpu().numpy().ravel()), r=GS.pack(Gw.reshape(-1, 22, 9).cpu().numpy().ravel()),
                                 tr=[round(float(v), 3) for v in tr[:, [0, 2]].cpu().numpy().ravel()],
                                 m=dict(cons=round(mt.consistency(x[None]), 2), bone=round(GS.bone_dev_mm(Pw), 1),
                                        travel=round(float((tr[-1] - tr[0]).norm()), 2))))
    res["judge"] = dict(J1_seam_all_le_1p5=bool(j1_ok), J2_follow="%d/%d" % (j2_hits, j2_n), J2_pass=j2_hits >= 7,
                        J3_segments_ok=bool(j3_ok))
    print("JUDGE", json.dumps(res["judge"]), "total %.0fs" % (time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(G.RUNS, TAG + ".json"), "w", encoding="utf-8"), indent=1)
    json.dump(out, open(os.path.join(D, TAG + "_viewer_data.json"), "w", encoding="utf-8"), separators=(",", ":"))


if __name__ == "__main__":
    main()
