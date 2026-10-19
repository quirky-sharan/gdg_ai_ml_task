import torch; torch.set_num_threads(4)
from explore import *
S3=(0,1,2)
cfgs=[
 ("base(orig HP)", {}, dict(epochs=60)),
 ("small 64/8", {}, dict(epochs=60, hid=64, lat=8)),
 ("small 64/8 drop.3 wd1e-3", {}, dict(epochs=60, hid=64, lat=8, p=0.3, wd=1e-3)),
 ("base drop.3 wd1e-3", {}, dict(epochs=60, p=0.3, wd=1e-3)),
 ("base lr3e-4", {}, dict(epochs=60, lr=3e-4)),
 ("base pw sqrt", {}, dict(epochs=60, pos_weight="sqrt")),
 ("base lam .3", {}, dict(epochs=60, lam=0.3)),
 ("base lam 3", {}, dict(epochs=60, lam=3.0)),
 ("base +mask ch", dict(use_mask_channels=True), dict(epochs=60)),
 ("base no legacy feats", dict(use_legacy_feats=False), dict(epochs=60)),
 ("base no clip", dict(clip=1e9), dict(epochs=60)),
 ("base clip3", dict(clip=3.0), dict(epochs=60)),
]
for n,a,b in cfgs: run(n,a,b,seeds=S3)
