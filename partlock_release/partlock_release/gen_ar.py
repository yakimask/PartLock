"""Plan 2: text -> causal-H tokens, one token at a time.
Per 4 frames: 7 tokens in order [global(4096), torso, head, arm_R, arm_L, leg_R, leg_L (128 each)].
- decoder-only Transformer, 8 layers, d=512, RoPE over flattened position (slides forever)
- text (CLIP ViT-B/32, 512) added to EVERY position (per-slot text possible later)
- corruption: replace tau*100% input tokens with random ones, tau~U[0,1] per sample (T2M-GPT)
- CFG: 10% of samples get the null text (MoMask / MotionStreamer)
- training stops by wall-clock (--seconds), cosine schedule on elapsed/budget
"""
import os, sys, math, time, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
BASE = os.environ.get("PARTLOCK_DIR", ".")   # folder that holds data/, ckpt/, runs/

GEN = os.path.join(BASE, "data", "gen")
CKPT = os.path.join(BASE, "ckpt")
RUNS = os.path.join(BASE, "runs")
VOC = [4096] + [128] * 6
OFF = [0]
for v in VOC[:-1]:
    OFF.append(OFF[-1] + v)
NV = sum(VOC)                       # 4864
S = 64                              # steps of context (= 8.5 s)
L = S * 7


def rope(x, pos):                   # x (B,h,T,dh)
    dh = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=x.device).float() / dh))
    a = pos[:, None].float() * inv[None]
    c, s = a.cos()[None, None], a.sin()[None, None]
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h = h
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = drop

    def forward(self, x, pos):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0)
        x = x + F.dropout(self.o(a.transpose(1, 2).reshape(B, T, D)), self.drop, self.training)
        return x + F.dropout(self.ff(self.n2(x)), self.drop, self.training)


class Gen(nn.Module):
    def __init__(self, d=512, layers=8, heads=8, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(NV + 1, d)          # last = BOS
        self.slot = nn.Embedding(7, d)              # slot of the token being PREDICTED
        self.txt = nn.Linear(512, d)
        self.null = nn.Parameter(torch.zeros(512))
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.nf = nn.LayerNorm(d)
        self.head = nn.Linear(d, NV)
        mask = torch.full((7, NV), float("-inf"))
        for s in range(7):
            mask[s, OFF[s]:OFF[s] + VOC[s]] = 0
        self.register_buffer("mask", mask)

    def forward(self, inp, slots, text):
        """inp (B,T) flat ids (BOS=NV); slots (T,) slot predicted at each position; text (B,T,512)"""
        x = self.tok(inp) + self.slot(slots)[None] + self.txt(text)
        pos = torch.arange(inp.shape[1], device=inp.device)
        for b in self.blocks:
            x = b(x, pos)
        return self.head(self.nf(x)) + self.mask[slots][None]


def flat(codes):                    # (n,7) local ids -> (n*7,) global ids
    return (codes.long() + torch.tensor(OFF)).reshape(-1)


def batch(rec, bs, dev, g, corrupt=True, drop=0.1, fixed=False):
    inps, tgts, txts = [], [], []
    lens = torch.tensor([r["codes"].shape[0] for r in rec], dtype=torch.float)
    idx = range(bs) if fixed else torch.multinomial(lens, bs, replacement=True, generator=g).tolist()
    for i in idx:
        r = rec[i]
        n = r["codes"].shape[0]
        st = 0 if fixed else int(torch.randint(0, max(1, n - S + 1), (1,), generator=g))
        c = flat(r["codes"][st:st + S])
        e = r["emb"][0 if fixed else int(torch.randint(0, r["emb"].shape[0], (1,), generator=g))].float()
        if not fixed and torch.rand(1, generator=g).item() < drop:
            e = None
        tgt = torch.full((L,), -100, dtype=torch.long); tgt[:len(c)] = c
        inp = torch.full((L,), NV, dtype=torch.long); inp[1:len(c)] = c[:-1]
        if corrupt and not fixed:
            tau = torch.rand(1, generator=g).item()
            sel = torch.rand(L, generator=g) < tau
            sel[0] = False
            slot_of_inp = (torch.arange(L) - 1) % 7
            rnd = torch.tensor(OFF)[slot_of_inp] + (torch.rand(L, generator=g) * torch.tensor(VOC)[slot_of_inp]).long()
            inp = torch.where(sel, rnd, inp)
        inps.append(inp); tgts.append(tgt); txts.append(e)
    return torch.stack(inps).to(dev), torch.stack(tgts).to(dev), txts


def text_tensor(m, txts, dev):
    rows = [m.null if t is None else t.to(dev) for t in txts]
    return torch.stack(rows)[:, None, :].expand(-1, L, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=7200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--overfit", type=int, default=0, help="N clips, no corruption/drop: memorization check")
    ap.add_argument("--tag", default="gen_v1")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    tr = torch.load(os.path.join(GEN, "train.pt"))
    va = torch.load(os.path.join(GEN, "val.pt"))
    if a.overfit:
        tr = tr[: a.overfit]
    m = Gen().to(dev)
    print("params %.1fM  train clips %d  val clips %d" % (sum(p.numel() for p in m.parameters()) / 1e6, len(tr), len(va)), flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    slots = (torch.arange(L) % 7).to(dev)
    vin, vtg, vtx = batch(va, min(256, len(va)), dev, g, fixed=True)
    log = []

    @torch.no_grad()
    def val_loss():
        m.eval()
        tot, per = 0.0, torch.zeros(7, device=dev); cnt = torch.zeros(7, device=dev)
        for i in range(0, vin.shape[0], 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(vin[i:i + 64], slots, text_tensor(m, vtx[i:i + 64], dev)).float()
            ce = F.cross_entropy(lo.reshape(-1, NV), vtg[i:i + 64].reshape(-1), reduction="none").view(-1, L)
            ok = (vtg[i:i + 64] != -100)
            for s in range(7):
                per[s] += (ce[:, s::7] * ok[:, s::7]).sum(); cnt[s] += ok[:, s::7].sum()
        m.train()
        p = (per / cnt)
        return float(per.sum() / cnt.sum()), [round(float(v), 3) for v in p]

    n_win = sum(r["codes"].shape[0] for r in tr) / S        # data size in 64-step windows (1 epoch)
    best, bad = [float("inf"), None, None], [0]
    t0, step, warm = time.time(), 0, 300
    if a.resume:
        ck = torch.load(os.path.join(CKPT, a.tag + "_last.pt"), weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); g.set_state(ck["rng"])
        step, best, bad, log = ck["step"], ck["best"], ck["bad"], ck["log"]
        t0 -= log[-1]["sec"] if log else 0
        print("resumed at step %d (best val %.4f)" % (step, best[0]), flush=True)
    while True:
        el = time.time() - t0
        if el > a.seconds:
            break
        lr = a.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, el / a.seconds)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        inp, tgt, txt = batch(tr, a.bs, dev, g, corrupt=not a.overfit, drop=0 if a.overfit else 0.1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = m(inp, slots, text_tensor(m, txt, dev))
        loss = F.cross_entropy(lo.float().reshape(-1, NV), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        step += 1
        if step == 20:
            torch.cuda.synchronize()
            print("  speed %.2f step/s -> ~%d steps in budget" % (20 / (time.time() - t0), a.seconds * 20 / (time.time() - t0)), flush=True)
        if step % 100 == 0:
            row = dict(step=step, sec=round(time.time() - t0), epochs=round(step * a.bs / n_win, 2),
                       train_loss=round(loss.item(), 4), lr=lr)
            if step % a.val_every == 0 or a.overfit:
                vl, per = val_loss(); row.update(val_loss=round(vl, 4), val_per_slot=per)
                if vl < best[0]:
                    best[:] = [vl, step, per]; bad[0] = 0
                    torch.save(m.state_dict(), os.path.join(CKPT, a.tag + ".pt"))
                    row["saved"] = True
                else:
                    bad[0] += 1
                torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step, best=best, bad=bad,
                                log=log, rng=g.get_state()), os.path.join(CKPT, a.tag + "_last.pt"))
            log.append(row)
            print("  " + json.dumps(row), flush=True)
            if not a.overfit and bad[0] >= a.patience:
                print("  early stop: val not improved for %d checks" % a.patience, flush=True)
                break
    if a.overfit:
        torch.save(m.state_dict(), os.path.join(CKPT, a.tag + ".pt"))
    json.dump(dict(args=vars(a), steps=step, best_val_loss=best[0], best_step=best[1], best_val_per_slot=best[2],
                   log=log), open(os.path.join(RUNS, a.tag + ".json"), "w"), indent=1)
    print("done steps %d (%.2f epochs)  best val %.4f at step %s  per-slot %s"
          % (step, step * a.bs / n_win, best[0], best[1], best[2]), flush=True)


if __name__ == "__main__":
    main()
