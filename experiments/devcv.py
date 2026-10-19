"""Rolling-origin CV inside the development period (first 80%) - never touches the test window."""
import torch; torch.set_num_threads(4)
import sys, time
from wg import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = load()
y = (df["label"].to_numpy() == 1).astype(np.int64)
N = len(df)
cuts = [int(N * f) for f in (0.4, 0.5, 0.6, 0.7, 0.8)]
folds = [(np.arange(0, a), np.arange(a, b)) for a, b in zip(cuts[:-1], cuts[1:])]
print("eval positives per fold", [int(y[b].sum()) for _, b in folds], flush=True)

def cv_avg(name, prep_kw={}, train_kw={}, seeds=(0, 1, 2)):
    t0 = time.time()
    H = []
    for a, b in folds:
        P = Prep(**prep_kw).fit(df.iloc[a])
        Xa, Ma, K = P.transform(df.iloc[a]); Xb, _, _ = P.transform(df.iloc[b])
        hs = []
        for s in seeds:
            _, h = train(Xa, Ma, y[a], K, Xb, y[b], seed=s, **train_kw)
            hs.append([[x[1], x[2]] for x in h])
        H.append(np.mean(hs, 0))  # epochs x 2
    H = np.array(H)  # folds x epochs x 2
    ap = H[:, :, 0].mean(0); auc = H[:, :, 1].mean(0)
    E = len(ap)
    msg = " ".join(f"e{e}:{ap[e-1]:.3f}/{auc[e-1]:.3f}" for e in (1, 3, 5, 10, 20, 30, 40, 60) if e <= E)
    print(f"{name:30s} | meanAP/meanAUC {msg} | best AUC {auc.max():.3f}@{auc.argmax()+1} | {time.time()-t0:.0f}s", flush=True)
    return H

def cv_sklearn(name, make):
    aps, aucs = [], []
    for a, b in folds:
        P = Prep().fit(df.iloc[a])
        Xa, _, _ = P.transform(df.iloc[a]); Xb, _, _ = P.transform(df.iloc[b])
        p = make().fit(Xa, y[a]).predict_proba(Xb)[:, 1]
        aps.append(average_precision_score(y[b], p)); aucs.append(roc_auc_score(y[b], p))
    print(f"{name:30s} | per-fold AUC {np.round(aucs,3)} | meanAP {np.mean(aps):.3f} meanAUC {np.mean(aucs):.3f}", flush=True)

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "sk":
        cv_sklearn("LR C=1e-3", lambda: LogisticRegression(C=1e-3, class_weight="balanced", max_iter=5000))
        cv_sklearn("LR C=1e-4", lambda: LogisticRegression(C=1e-4, class_weight="balanced", max_iter=5000))
        cv_sklearn("RF", lambda: RandomForestClassifier(500, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=8, random_state=0))
    elif which == "sae1":
        cv_avg("base(orig HP)", {}, dict(epochs=60))
        cv_avg("small 64/8 drop.3 wd1e-3", {}, dict(epochs=60, hid=64, lat=8, p=0.3, wd=1e-3))
    elif which == "sae2":
        cv_avg("base pw sqrt", {}, dict(epochs=60, pos_weight="sqrt"))
        cv_avg("base lam 3", {}, dict(epochs=60, lam=3.0))
    elif which == "sae3":
        cv_avg("base lr3e-4", {}, dict(epochs=60, lr=3e-4))
        cv_avg("base drop.3 wd1e-3", {}, dict(epochs=60, p=0.3, wd=1e-3))
    elif which == "q1":
        cv_avg("quantile orig HP", dict(scaling="quantile"), dict(epochs=60))
        cv_avg("quantile small drop.3 wd1e-3", dict(scaling="quantile"), dict(epochs=60, hid=64, lat=8, p=0.3, wd=1e-3))
    elif which == "q2":
        cv_avg("quantile lam3", dict(scaling="quantile"), dict(epochs=60, lam=3.0))
        cv_avg("quantile drop.3 wd1e-3", dict(scaling="quantile"), dict(epochs=60, p=0.3, wd=1e-3))
    elif which == "q3":
        cv_avg("quantile +mask ch", dict(scaling="quantile", use_mask_channels=True), dict(epochs=60))
        cv_avg("quantile pw sqrt", dict(scaling="quantile"), dict(epochs=60, pos_weight="sqrt"))
    elif which == "q4":
        cv_avg("quantile lr3e-4", dict(scaling="quantile"), dict(epochs=60, lr=3e-4))
        cv_avg("quantile 128/16", dict(scaling="quantile"), dict(epochs=60, hid=128, lat=16))
    elif which == "q5":
        cv_avg("quantile no legacy feats", dict(scaling="quantile", use_legacy_feats=False), dict(epochs=60))
        cv_avg("quantile maxmiss .2", dict(scaling="quantile", max_missing=0.2), dict(epochs=60))
