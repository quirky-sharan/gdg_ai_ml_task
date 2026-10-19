from wg import *
from sklearn.linear_model import LogisticRegression
import time

df = load()
tr, va, te = temporal_split(df)
y = (df["label"].to_numpy() == 1).astype(np.int64)
print("split sizes", len(tr), len(va), len(te), "pos", y[tr].sum(), y[va].sum(), y[te].sum())
print("boundaries", df.timestamp.iloc[tr[-1]], df.timestamp.iloc[va[0]], df.timestamp.iloc[va[-1]], df.timestamp.iloc[te[0]])

def run(name, prep_kw={}, train_kw={}, seeds=(0, 1, 2, 3, 4)):
    t0 = time.time()
    P = Prep(**prep_kw).fit(df.iloc[tr])
    Xtr, Mtr, K = P.transform(df.iloc[tr]); Xva, _, _ = P.transform(df.iloc[va])
    H = []
    for s in seeds:
        _, h = train(Xtr, Mtr, y[tr], K, Xva, y[va], seed=s, **train_kw)
        H.append(h)
    H = np.array(H)  # seeds x epochs x 3
    ap = H[:, :, 1].mean(0); auc = H[:, :, 2].mean(0)
    eps = H[0, :, 0].astype(int)
    msg = " ".join(f"e{e}:{a:.3f}/{u:.3f}" for e, a, u in zip(eps, ap, auc) if e in (5, 10, 15, 20, 30, 40, 60, 80))
    print(f"{name:28s} K={K} dim={Xtr.shape[1]} | AP/AUC {msg} | best AP {ap.max():.3f}@{eps[ap.argmax()]} | {time.time()-t0:.0f}s", flush=True)

def lr_baseline(C=0.01, prep_kw={}):
    P = Prep(**prep_kw).fit(df.iloc[tr])
    Xtr, _, _ = P.transform(df.iloc[tr]); Xva, _, _ = P.transform(df.iloc[va])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Xtr, y[tr])
    p = m.predict_proba(Xva)[:, 1]
    print(f"LR C={C}: val AP {average_precision_score(y[va], p):.3f} AUC {roc_auc_score(y[va], p):.3f}")

if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("lr", "all"):
        for C in (0.001, 0.01, 0.1, 1.0):
            lr_baseline(C)
    if which in ("sae", "all"):
        run("base", {}, dict(epochs=60))
        run("pw sqrt", {}, dict(epochs=60, pos_weight="sqrt"))
        run("pw none", {}, dict(epochs=60, pos_weight="none"))
        run("lam 0.3", {}, dict(epochs=60, lam=0.3))
        run("lam 3", {}, dict(epochs=60, lam=3.0))
        run("mask channels", dict(use_mask_channels=True), dict(epochs=60))
        run("no legacy feats", dict(use_legacy_feats=False), dict(epochs=60))
        run("no clip", dict(clip=1e9), dict(epochs=60))
        run("wd 1e-3 drop .3", {}, dict(epochs=60, wd=1e-3, p=0.3))
        run("lr 3e-4", {}, dict(epochs=60, lr=3e-4))
