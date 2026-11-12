"""Stop test with new random draws: 4 x run->stop, 4 x walk->stop (torch seed 1, batched). gen_v1, CFG 4.
Pass per sample, every 10 s bin from 12 s: all parts < 0.1 m/s, pelvis travel < 0.5 m, pelvis height above lowest foot >= 0.8 m."""
import os, sys, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq272 as B, tvq272c as C, gen_ar as G, gen_sample as GS
from switch_test import generate_sched
dev = "cuda"; D = os.path.dirname(os.path.abspath(__file__))
STOP = "a person stops and stands up straight."
MOV = [("run", "a person keeps running forward."), ("walk", "a person keeps walking forward.")]
K = 5
PJ = {"胴": [3, 6, 9], "頭": [12, 15], "右腕": [17, 19, 21], "左腕": [16, 18, 20], "右脚": [5, 8, 11], "左脚": [4, 7, 10]}
mean = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Mean.npy")).astype(np.float32)).to(dev)
std = torch.from_numpy(np.load(os.path.join(B.ROOT, "mean_std", "Std.npy")).astype(np.float32)).to(dev)
h = C.CHierTok().to(dev); h.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location=dev)); h.eval()
g = G.Gen().to(dev); g.load_state_dict(torch.load(os.path.join(G.CKPT, "gen_v1.pt"), map_location=dev)); g.eval()
tok, clip = GS.load()
E = GS.embed(tok, clip, [m[1] for m in MOV] + [STOP])
rows = [(m, k) for m in range(2) for k in range(K)]
embs = torch.stack([torch.stack([E[m]] + [E[2]] * 5) for m, _ in rows])
codes = generate_sched(g, embs, 75, seed=1, reset=True, keep=1)
bins = [(0, 300), (300, 360)] + [(a, a + 300) for a in range(360, 1800, 300)]
lab = ["0–10秒", "10–12秒"] + ["%d–%d秒" % (a // 30, min(60, a // 30 + 10)) for a in range(360, 1800, 300)]
res, out = dict(bins=lab), dict(fps=30, parents=B.PARENTS, clips=[])
for i, (m, k) in enumerate(rows):
    name = "%s_%d" % (MOV[m][0], k + 1)
    x = GS.decode_codes(h, codes[i][None], mean, std)[0]
    Pw, Gw, tr = GS.to_world(x)
    P, R = Pw.cpu().numpy(), Gw.cpu().numpy()
    Lc = np.einsum("tji,tkj->tki", R[:, 0], P - P[:, :1])
    V = np.zeros(Lc.shape[:2]); V[1:] = np.linalg.norm(np.diff(Lc, axis=0), axis=-1) * 30
    t = tr[:, [0, 2]].cpu().numpy()
    hy = P[:, 0, 1] - P[:, [7, 8, 10, 11], 1].min(1)
    per = {pn: [round(float(V[a:b, js].mean()), 3) for a, b in bins] for pn, js in PJ.items()}
    trav = [round(float(np.linalg.norm(t[min(b, len(t) - 1)] - t[a])), 2) for a, b in bins]
    hgt = [round(float(hy[a:b].mean()), 2) for a, b in bins]
    still = all(per[pn][j] < 0.1 for pn in PJ for j in range(2, len(bins))) and all(v < 0.5 for v in trav[2:])
    stand = all(v >= 0.8 for v in hgt[2:])
    res[name] = dict(part_speed=per, travel_m=trav, pelvis_h=hgt, still=bool(still), standing=bool(stand), pass_=bool(still and stand))
    print(name, "still", still, "standing", stand, "h", hgt, "legs", per["右脚"][2:], per["左脚"][2:], flush=True)
    rel = Pw.clone(); rel[..., 0] -= tr[:, None, 0]; rel[..., 2] -= tr[:, None, 2]
    out["clips"].append(dict(key=name, caption=MOV[m][1] + " → " + STOP, frames=int(x.shape[0]),
                             seg=[dict(f=0, text=MOV[m][1]), dict(f=300, text=STOP)],
                             p=GS.pack(rel.cpu().numpy().ravel()), r=GS.pack(R.reshape(-1, 22, 9).ravel()),
                             tr=[round(float(v), 3) for v in t.ravel()],
                             m=dict(cons=0, bone=round(GS.bone_dev_mm(Pw), 1), travel=round(float(np.linalg.norm(t[-1] - t[0])), 2))))
json.dump(res, open(os.path.join(G.RUNS, "stop_test.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(out, open(os.path.join(D, "stop_test_data.json"), "w"), separators=(",", ":"))
