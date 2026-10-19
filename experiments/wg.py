import sys, random
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve

LEGACY_CODES = (100.0, 200.0, 300.0)

def load(path="secom_telemetry_modified.csv"):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df

def temporal_split(df, fr=(0.6, 0.2)):
    ts = df["timestamp"]
    n = len(df)
    def cut(i):
        # move boundary forward so identical timestamps stay in one partition
        while 0 < i < n and ts.iloc[i] == ts.iloc[i - 1]:
            i += 1
        return i
    a = cut(int(round(fr[0] * n)))
    b = cut(int(round((fr[0] + fr[1]) * n)))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)

class Prep:
    """Train-only fitted preprocessor with PS-L legacy handling."""
    def __init__(self, max_missing=0.5, clip=5.0, use_mask_channels=False, use_legacy_feats=True, scaling="standard"):
        self.max_missing, self.clip, self.scaling = max_missing, clip, scaling
        self.use_mask_channels, self.use_legacy_feats = use_mask_channels, use_legacy_feats

    @staticmethod
    def split_legacy(df):
        S = [c for c in df.columns if c.startswith("sensor_")]
        X = df[S].to_numpy(dtype=np.float64).copy()
        legacy = (df["sensor_0"].to_numpy() == 0.0)
        code = np.where(legacy, df["sensor_1"].to_numpy(), np.nan)
        X[legacy, 0] = np.nan  # sentinel -> not a measurement
        X[legacy, 1] = np.nan  # firmware code -> not a continuous measurement
        return X, legacy, code, S

    def fit(self, df):
        X, legacy, code, S = self.split_legacy(df)
        miss = np.isnan(X).mean(0)
        with np.errstate(all="ignore"):
            mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
        keep = (miss <= self.max_missing) & np.isfinite(sd) & (sd > 1e-12)
        self.keep, self.mu, self.sd = keep, mu[keep], sd[keep]
        self.cols = [s for s, k in zip(S, keep) if k]
        self.codes = sorted(set(code[legacy].tolist()))
        if self.scaling == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            self.qt = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution="normal", subsample=10**9, random_state=0).fit(X[:, keep])
        return self

    def transform(self, df):
        X, legacy, code, _ = self.split_legacy(df)
        X = X[:, self.keep]
        m = ~np.isnan(X)
        Z = self.qt.transform(X) if self.scaling == "quantile" else (X - self.mu) / self.sd
        Z = np.clip(Z, -self.clip, self.clip)
        Z = np.where(m, Z, 0.0)
        parts = [Z]
        if self.use_mask_channels:
            parts.append(m.astype(np.float64))
        if self.use_legacy_feats:
            onehot = np.stack([(legacy & (code == c)) for c in self.codes], 1).astype(np.float64)
            parts += [legacy[:, None].astype(np.float64), onehot]
        return np.concatenate(parts, 1).astype(np.float32), m.astype(np.float32), Z.shape[1]

class WaferSAE(nn.Module):
    def __init__(self, d_in, d_rec, d_hid=256, d_lat=32, p=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hid), nn.LayerNorm(d_hid), nn.GELU(), nn.Dropout(p),
            nn.Linear(d_hid, d_hid // 2), nn.LayerNorm(d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_lat))
        self.decoder = nn.Sequential(
            nn.Linear(d_lat, d_hid // 2), nn.GELU(),
            nn.Linear(d_hid // 2, d_hid), nn.GELU(),
            nn.Linear(d_hid, d_rec))
        self.head = nn.Linear(d_lat, 1)
    def forward(self, x):
        assert x.dim() == 2, f"expected [B, F], got {tuple(x.shape)}"
        z = self.encoder(x)
        return self.decoder(z), self.head(z).squeeze(-1), z

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def predict(model, X):
    model.eval()
    with torch.no_grad():
        _, lg, _ = model(torch.from_numpy(X))
    return torch.sigmoid(lg).numpy()

def train(Xtr, Mtr, ytr, d_rec, Xva=None, yva=None, seed=0, epochs=60, lr=1e-3, wd=1e-5, bs=64,
          lam=1.0, pos_weight="ratio", hid=256, lat=32, p=0.1, patience=None, verbose=False):
    seed_all(seed)
    model = WaferSAE(Xtr.shape[1], d_rec, hid, lat, p)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    npos = ytr.sum(); nneg = len(ytr) - npos
    pw = {"ratio": nneg / npos, "sqrt": np.sqrt(nneg / npos), "none": 1.0}[pos_weight]
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pw)))
    Xt, Mt, yt = map(torch.from_numpy, (Xtr, Mtr, ytr.astype(np.float32)))
    g = torch.Generator().manual_seed(seed)
    hist = []
    best = (-1, None, 0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            xb, mb, yb = Xt[idx], Mt[idx], yt[idx]
            rec, lg, _ = model(xb)
            tgt = xb[:, :d_rec]
            assert rec.shape == tgt.shape == mb.shape and lg.shape == yb.shape
            rl = ((rec - tgt) ** 2 * mb).sum() / mb.sum().clamp_min(1)
            cl = bce(lg, yb)
            loss = rl + lam * cl
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        if Xva is not None:
            pv = predict(model, Xva)
            ap = average_precision_score(yva, pv); auc = roc_auc_score(yva, pv)
            hist.append((ep, ap, auc))
            if verbose and (ep % 10 == 0 or ep == 1): print(ep, round(ap, 4), round(auc, 4))
    return model, hist

def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    i = np.nanargmax(f1[:-1])
    return th[i], f1[i]
