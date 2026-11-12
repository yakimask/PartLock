import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq272c as C, tvq272 as B
m = C.CHierTok().double()
m.load_state_dict(torch.load(os.path.join(B.CKPT_DIR, "t272_causal_v1_H.pt"), map_location="cpu")); m.eval()
torch.manual_seed(0)
qg = torch.randn(1,128,40,dtype=torch.float64); q = torch.randn(1,128,40,dtype=torch.float64)
d = m.dec[2]
with torch.no_grad():
    full = d(torch.cat([qg,q],1)); w = 0
    for k in range(1,41):
        part = d(torch.cat([qg[...,:k],q[...,:k]],1))
        w = max(w, float((part[...,4*k-4:4*k]-full[...,4*k-4:4*k]).abs().max()))
print("float64 CPU stream vs whole max diff", w)
