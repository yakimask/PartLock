"""How many frames ahead / behind does each H look? (empirical, via gradients, untrained weights)"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tvq272 as B, tvq272c as C

def reach(m, name):
    torch.manual_seed(0)
    x = torch.randn(1, 256, 272, requires_grad=True)
    # encoder: which input frames influence global token at index 32 (frames 128..131)
    z = m.genc(x.permute(0, 2, 1))
    g, = torch.autograd.grad(z[0, :, 32].abs().sum(), x)
    f = (g[0].abs().sum(-1) > 0).nonzero().flatten()
    print("%s encoder token(frames128-131): sees frames %d..%d -> past %d, future %d"
          % (name, f.min(), f.max(), 128 - f.min(), f.max() - 131))
    # decoder: which tokens influence output frame 128
    q = torch.randn(1, 256, 64, requires_grad=True)
    y = m.dec[0](q)
    g, = torch.autograd.grad(y[0, :, 128].abs().sum(), q)
    t = (g[0].abs().sum(0) > 0).nonzero().flatten()
    print("%s decoder frame128 (token 32): uses tokens %d..%d -> past %d, future %d"
          % (name, t.min(), t.max(), 32 - t.min(), t.max() - 32))

reach(B.HierTok(), "old H   ")
reach(C.CHierTok(), "causal H")
