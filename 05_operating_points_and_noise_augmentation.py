"""
Threshold-free operating points of the unstable class, channel-level importance
and noise-augmented training (Figures 9 and 10).
"""
import os
import json
import time
import platform
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             recall_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ========================= SETTINGS =========================
ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
S4 = os.path.join(ROOT, "_STEP4")
OUT = os.path.join(ROOT, "_STEP6")
REF = "clear"
SEEDS = [0, 1, 2, 3, 4]
N_FOLD = 5
TAU_MAIN = 100
TRAIN_NOISE, TEST_NOISE = 1, 1
J_UNITS = 3            # shape function e sigmoid unit
H_HIDDEN = 8           # base branch er hidden unit
L2_GRID = [1e-3, 1e-2, 1e-1]
MAXITER = 400
PCR_DELTA = 0.5        # standardized unit e perturbation
EXTRA_PHYSORD = [("PhysOrd-Net-L", 6, 16)]   # (naam, J_UNITS, H_HIDDEN) boro version
# ============================================================

CLASSES = ["stable", "alert", "unstable"]
LEVELS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]

BASE_DROP = ["dV_min", "dV_min_cum", "I_BESS_rms_max", "I_BESS_abs_peak",
             "I_BESS_abs_peak_cum", "df_absmax", "df_absmax_cum"]


# ================================================================ PhysOrd-Net
def softplus(x):
    return np.logaddexp(0.0, x)


class PhysOrdNet:
    """Monotone ordinal model. Xp: physical indicator (direction diye gun kora), Xb: base."""

    def __init__(self, n_p, n_b, J=J_UNITS, H=H_HIDDEN, l2=1e-2, mono=True, seed=0):
        self.n_p, self.n_b, self.J, self.H, self.l2, self.mono, self.seed = n_p, n_b, J, H, l2, mono, seed

    # ---- parameter packing ----
    def _shapes(self):
        nb_in = self.n_b + (0 if self.mono else self.n_p)
        sh = {}
        if self.mono and self.n_p:
            sh.update(alpha=(self.n_p, self.J), beta=(self.n_p, self.J), c=(self.n_p, self.J))
        if nb_in:
            sh.update(W1=(nb_in, self.H), b1=(self.H,), W2=(self.H,), b2=(1,))
        sh.update(theta0=(1,), delta=(1,))
        return sh

    def _unpack(self, w):
        out, i = {}, 0
        for k, s in self._shapes().items():
            n = int(np.prod(s))
            out[k] = w[i:i + n].reshape(s)
            i += n
        return out

    def _init(self):
        rng = np.random.default_rng(self.seed)
        p = {}
        for k, s in self._shapes().items():
            if k == "alpha":
                p[k] = np.full(s, -1.0) + 0.1 * rng.standard_normal(s)
            elif k == "beta":
                p[k] = np.full(s, 0.5) + 0.1 * rng.standard_normal(s)
            elif k == "c":
                p[k] = np.tile(np.linspace(-1, 1, s[1]), (s[0], 1)) + 0.1 * rng.standard_normal(s)
            elif k == "W1":
                p[k] = rng.standard_normal(s) / np.sqrt(s[0])
            elif k == "W2":
                p[k] = rng.standard_normal(s) / np.sqrt(s[0])
            elif k == "delta":
                p[k] = np.array([0.5])
            else:
                p[k] = np.zeros(s)
        return np.concatenate([p[k].ravel() for k in self._shapes()])

    # ---- forward ----
    def _score(self, p, Xp, Xb, keep=False):
        s = np.zeros(len(Xp) if Xp is not None and Xp.size else len(Xb))
        cache = {}
        if self.mono and self.n_p:
            a, b = softplus(p["alpha"]), softplus(p["beta"])
            U = b[None] * (Xp[:, :, None] - p["c"][None])          # n x K x J
            S = expit(U)
            s = s + np.einsum("kj,nkj->n", a, S)
            cache.update(a=a, b=b, S=S)
        xin = Xb if self.mono else np.hstack([Xp, Xb]) if Xb is not None and Xb.size else Xp
        if "W1" in p:
            Z = np.tanh(xin @ p["W1"] + p["b1"])
            s = s + Z @ p["W2"] + p["b2"][0]
            cache.update(Z=Z, xin=xin)
        return (s, cache) if keep else s

    def _thetas(self, p):
        return p["theta0"][0], p["theta0"][0] + softplus(p["delta"][0])

    def _loss(self, w, Xp, Xb, y, sw):
        p = self._unpack(w)
        s, C = self._score(p, Xp, Xb, keep=True)
        t0, t1 = self._thetas(p)
        z0, z1 = s - t0, s - t1
        y0, y1 = (y >= 1).astype(float), (y >= 2).astype(float)
        L = sw * (np.logaddexp(0, z0) - y0 * z0 + np.logaddexp(0, z1) - y1 * z1)
        n = sw.sum()
        loss = L.sum() / n
        r0, r1 = sw * (expit(z0) - y0) / n, sw * (expit(z1) - y1) / n
        ds = r0 + r1
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["theta0"][0] = -(r0.sum() + r1.sum())
        g["delta"][0] = -r1.sum() * expit(p["delta"][0])
        if "alpha" in p:
            a, b, S = C["a"], C["b"], C["S"]
            dS = S * (1 - S)                                        # n x K x J
            g["alpha"] = np.einsum("n,nkj->kj", ds, S) * expit(p["alpha"])
            xc = Xp[:, :, None] - p["c"][None]
            g["beta"] = np.einsum("n,kj,nkj,nkj->kj", ds, a, dS, xc) * expit(p["beta"])
            g["c"] = -np.einsum("n,kj,nkj->kj", ds, a * b, dS)
            loss += self.l2 * (np.sum(softplus(p["alpha"]) ** 2) * 0.1)
            g["alpha"] += self.l2 * 0.2 * softplus(p["alpha"]) * expit(p["alpha"])
        if "W1" in p:
            Z, xin = C["Z"], C["xin"]
            g["W2"] = Z.T @ ds
            g["b2"][0] = ds.sum()
            dZ = np.outer(ds, p["W2"]) * (1 - Z ** 2)
            g["W1"] = xin.T @ dZ
            g["b1"] = dZ.sum(0)
            loss += self.l2 * (np.sum(p["W1"] ** 2) + np.sum(p["W2"] ** 2))
            g["W1"] += 2 * self.l2 * p["W1"]
            g["W2"] += 2 * self.l2 * p["W2"]
        return loss, np.concatenate([g[k].ravel() for k in self._shapes()])

    def fit(self, Xp, Xb, y):
        cnt = np.bincount(y, minlength=3).astype(float)
        sw = (len(y) / (3 * np.maximum(cnt, 1)))[y]
        r = minimize(self._loss, self._init(), args=(Xp, Xb, y, sw), jac=True,
                     method="L-BFGS-B", options={"maxiter": MAXITER})
        self.w_ = r.x
        return self

    def proba(self, Xp, Xb):
        p = self._unpack(self.w_)
        s = self._score(p, Xp, Xb)
        t0, t1 = self._thetas(p)
        q0, q1 = expit(s - t0), expit(s - t1)
        return np.column_stack([1 - q0, q0 - q1, q1])

    def predict(self, Xp, Xb):
        return self.proba(Xp, Xb).argmax(1)

    def n_params(self):
        return int(sum(np.prod(s) for s in self._shapes().values()))

    def shape_curve(self, k, u):
        p = self._unpack(self.w_)
        a, b = softplus(p["alpha"][k]), softplus(p["beta"][k])
        return (a[None] * expit(b[None] * (u[:, None] - p["c"][k][None]))).sum(1)


# ================================================================ helpers
def scores(y, p):
    return {"accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
            "macro_F1": f1_score(y, p, average="macro"),
            **{f"recall_{c}": recall_score(y, p, labels=[i], average="macro", zero_division=0)
               for i, c in enumerate(CLASSES)}}


class Prep:
    """Train fold e fit: standardize; physical indicator direction diye gun (risk barle bare)."""

    def __init__(self, P_cols, B_cols, dirs):
        self.P_cols, self.B_cols, self.dirs = P_cols, B_cols, np.asarray(dirs, float)

    def fit(self, df):
        self.mp, self.sp = df[self.P_cols].mean().to_numpy(), df[self.P_cols].std().to_numpy() + 1e-9
        if self.B_cols:
            self.mb, self.sb = df[self.B_cols].mean().to_numpy(), df[self.B_cols].std().to_numpy() + 1e-9
        return self

    def __call__(self, df):
        Xp = ((df[self.P_cols].to_numpy() - self.mp) / self.sp) * self.dirs if self.P_cols \
            else np.zeros((len(df), 0))
        Xb = (df[self.B_cols].to_numpy() - self.mb) / self.sb if self.B_cols else np.zeros((len(df), 0))
        return Xp, Xb


def fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed, mono=True, J=J_UNITS, H=H_HIDDEN):
    """Inner 3-fold CV e L2 bachai, tarpor puro train fold e fit."""
    best, best_f1 = L2_GRID[0], -1
    inner = StratifiedKFold(3, shuffle=True, random_state=seed)
    for l2 in L2_GRID:
        f1s = []
        for a, b in inner.split(dtr, ytr):
            pr = Prep(P_cols, B_cols, dirs).fit(dtr.iloc[a])
            m = PhysOrdNet(len(P_cols), len(B_cols), J=J, H=H, l2=l2, mono=mono,
                           seed=seed).fit(*pr(dtr.iloc[a]), ytr[a])
            f1s.append(f1_score(ytr[b], m.predict(*pr(dtr.iloc[b])), average="macro"))
        if np.mean(f1s) > best_f1:
            best, best_f1 = l2, np.mean(f1s)
    pr = Prep(P_cols, B_cols, dirs).fit(dtr)
    m = PhysOrdNet(len(P_cols), len(B_cols), J=J, H=H, l2=best, mono=mono, seed=seed).fit(*pr(dtr), ytr)
    return m, pr, best


def baseline_models(n_physord_params):
    m = {
        "LR": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
               {"logisticregression__C": [0.1, 1, 10]}),
        "RF": (RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=1),
               {"max_depth": [None, 8], "min_samples_leaf": [1, 3]}),
    }
    try:
        from xgboost import XGBClassifier
        m["XGBoost"] = (XGBClassifier(n_estimators=300, eval_metric="mlogloss", tree_method="hist",
                                      n_jobs=1, verbosity=0),
                        {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]})
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m["HGB"] = (HistGradientBoostingClassifier(class_weight="balanced", min_samples_leaf=5),
                    {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]})
    m["MLP_pm"] = (None, {"mlpclassifier__alpha": [1e-4, 1e-2]})   # parameter-matched, nicher fn e banai
    return m


def make_mlp_pm(n_in, n_target):

    h = max(4, int(round((n_target - 3) / (n_in + 4))))
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(h,), max_iter=3000,
                                                         early_stopping=True)), h


def set_seed(est, seed):
    for k in est.get_params():
        if k.endswith("random_state"):
            est.set_params(**{k: seed})
    return est


def fit_baseline(name, base, grid, X, y, seed):
    est = set_seed(clone(base), seed)
    kw = {}
    if name == "XGBoost":
        kw = {"sample_weight": pd.Series(y).map(len(y) / (3 * pd.Series(y).value_counts())).to_numpy()}
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                      n_jobs=-1)
    gs.fit(X, y, **kw)
    return gs.best_estimator_


class MonoOrdGBM:
    """Monotone-constrained gradient boosting, ordinal: duita cumulative binary model
    (y>=1, y>=2). Physical indicator e direction onujayi monotone constraint, base e 0."""

    def __init__(self, cons, seed):
        self.cons, self.seed = cons, seed

    def _model(self):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=1, verbosity=0,
                                 eval_metric="logloss", random_state=self.seed,
                                 monotone_constraints="(" + ",".join(str(int(c)) for c in self.cons) + ")"), \
                {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(monotonic_cst=list(self.cons), min_samples_leaf=5,
                                                  random_state=self.seed), \
                {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]}

    def fit(self, X, y):
        self.m = []
        for k in (1, 2):
            base, grid = self._model()
            gs = GridSearchCV(base, grid, scoring="roc_auc", n_jobs=-1,
                              cv=StratifiedKFold(3, shuffle=True, random_state=self.seed))
            gs.fit(X, (y >= k).astype(int))
            self.m.append(gs.best_estimator_)
        return self

    def predict_proba(self, X):
        q0 = self.m[0].predict_proba(X)[:, 1]
        q1 = np.minimum(self.m[1].predict_proba(X)[:, 1], q0)
        return np.column_stack([1 - q0, q0 - q1, q1])


def risk(P):
    return P[:, 1] + 2 * P[:, 2]          # expected severity


# ================================================================== STEP 7 SETTINGS
OUT = os.path.join(ROOT, "_RESULTS_JOURNAL")
T_DIR, F_DIR, P_DIR = (os.path.join(OUT, d) for d in ("tables", "figures", "predictions"))
TAUS_SWEEP = [20, 50, 100, 200, 500]
MAIN_MODELS = ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF", "SVM", "LR", "MLP_pm"]
GEN_MODELS = ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"]      # generalization / robustness
N_BOOT = 2000
DPI = 300
CHANNELS = {
    "frequency (PLL)": ["df_", "f_std", "ROCOF", "U_F", "f_resid"],
    "BESS current": ["I_BESS", "dI_BESS", "I_ratio", "I_over"],
    "POI current": ["I_POI", "dI_POI"],
    "active/reactive power": ["dP_", "dQ_", "P_total_std", "P_BESS_std", "zeta_P", "growth_P", "P_resid"],
}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
LOG = []


def log(msg):
    print(msg)
    LOG.append(msg)


def save_table(df, name, title):
    df.to_csv(os.path.join(T_DIR, f"{name}.csv"), index=False)
    TABLES[name] = (title, df)
    log(f"  [table] {name}: {title}")


def save_fig(fig, name, title):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(F_DIR, f"{name}.{ext}"), dpi=DPI)
    plt.close(fig)
    FIGS[name] = title
    log(f"  [figure] {name}: {title}")


def ece(y_bin, p, bins=10):
    e, edges = 0.0, np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p > a) & (p <= b)
        if m.any():
            e += m.mean() * abs(y_bin[m].mean() - p[m].mean())
    return e


def full_metrics(y, P):
    from sklearn.metrics import matthews_corrcoef, precision_score
    p = P.argmax(1)
    yb = (y == 2).astype(float)
    return {"accuracy": accuracy_score(y, p), "balanced_acc": balanced_accuracy_score(y, p),
            "macro_F1": f1_score(y, p, average="macro"), "MCC": matthews_corrcoef(y, p),
            **{f"recall_{c}": recall_score(y, p, labels=[i], average="macro", zero_division=0)
               for i, c in enumerate(CLASSES)},
            "precision_unstable": precision_score(y == 2, p == 2, zero_division=0),
            "ordinal_MAE": float(np.mean(np.abs(y - p))),
            "severe_error_%": 100 * float(np.mean(np.abs(y - p) == 2)),
            "brier": float(np.mean(np.sum((P - np.eye(3)[y]) ** 2, 1))),
            "ECE_unstable": ece(yb, P[:, 2])}


def channel_cols(F, keys):
    return [c for c in F if any(c.startswith(k) or k in c for k in keys)]


# ================================================================== STEP 7b ADD-ON
AUG_NOISE = [0, 1, 2, 5]
CHANNEL_OF = [("voltage", ["dV_", "V_", "VI_", "U_LV", "U_HV", "sag_depth", "rec_index", "zeta_V", "growth_V"]),
              ("frequency", ["df_", "f_std", "ROCOF", "U_F", "f_resid"]),
              ("current", ["I_", "dI_"]),
              ("power", ["dP_", "dQ_", "P_total_std", "P_BESS_std", "zeta_P", "growth_P", "P_resid"]),
              ("other", ["osc_valid"])]


def channel(feat):
    for ch, keys in CHANNEL_OF:
        if any(feat.startswith(k) for k in keys):
            return ch
    return "other"


def main():
    global TABLES, FIGS
    TABLES, FIGS = {}, {}
    for d in (OUT, T_DIR, F_DIR, P_DIR):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    log("=" * 64 + "\n STEP 7b: ADD-ON (safety operating points, channel consistency, noise-aware training)\n" + "=" * 64)
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

    # ================================================================ A. threshold-free safety metrics
    log("\n[A] Unstable-class operating points from saved out-of-fold predictions")
    rows = []
    for name in MAIN_MODELS:
        for seed in SEEDS:
            pth = os.path.join(P_DIR, f"oof_{name}_seed{seed}.csv")
            if not os.path.exists(pth):
                continue
            o = pd.read_csv(pth)
            yb = (o.y_true == 2).astype(int).to_numpy()
            p = o.P_unstable.to_numpy()
            pr_, rc_, _ = precision_recall_curve(yb, p)
            r90 = rc_[pr_ >= 0.90].max() if (pr_ >= 0.90).any() else 0.0
            r95 = rc_[pr_ >= 0.95].max() if (pr_ >= 0.95).any() else 0.0
            p95 = pr_[rc_ >= 0.95].max() if (rc_ >= 0.95).any() else 0.0
            # stable run ke unstable bola (sobcheye bhul): P(unstable) >= 0.5 jokhon true = stable
            rows.append({"model": name, "seed": seed, "ROC_AUC_unstable": roc_auc_score(yb, p),
                         "AP_unstable": average_precision_score(yb, p), "recall_at_precision_0.90": r90,
                         "recall_at_precision_0.95": r95, "precision_at_recall_0.95": p95})
    A = pd.DataFrame(rows)
    As = A.drop(columns="seed").groupby("model").agg(["mean", "std"])
    t = pd.DataFrame({"model": As.index})
    for c in A.columns.drop(["model", "seed"]):
        t[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(As[(c, "mean")], As[(c, "std")])]
    t = t.set_index("model").loc[[m for m in MAIN_MODELS if m in t.model.values]].reset_index()
    save_table(t, "Table22_unstable_operating_points",
               "Threshold-free unstable-class metrics: ROC-AUC, AP, recall at fixed precision, precision at recall 0.95")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    mm = [m for m in MAIN_MODELS if m in As.index]
    for a, c, lab in [(ax[0], "recall_at_precision_0.90", "Unstable recall at precision ≥ 0.90"),
                      (ax[1], "precision_at_recall_0.95", "Precision at unstable recall ≥ 0.95")]:
        a.bar(mm, As.loc[mm, (c, "mean")], yerr=As.loc[mm, (c, "std")], capsize=3,
              color=["#1f5fa8", "#4a90d9"] + ["#9aa5b1"] * (len(mm) - 2))
        a.set_title(lab); a.tick_params(axis="x", rotation=40); a.set_ylim(0, 1.02)
    save_fig(fig, "Fig13_unstable_operating_points", "Safety-oriented operating points for the unstable class")

    # ================================================================ B. channel-level explanation consistency
    log("\n[B] Measurement-channel level importance and consistency")
    imp = pd.read_csv(os.path.join(T_DIR, "Table18_permutation_importance.csv"))
    models = [c for c in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"] if c in imp.columns]
    imp["channel"] = imp.feature.map(channel)
    ch = imp.groupby("channel")[models].apply(lambda d: d.clip(lower=0).sum())
    ch = 100 * ch / ch.sum()
    ch = ch.drop(index="other", errors="ignore")
    save_table(ch.reset_index(), "Table23_importance_by_channel",
               "Share of permutation importance per measurement channel (%)")
    from scipy.stats import spearmanr
    cr = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = ch[models[i]], ch[models[j]]
            cr.append({"model_A": models[i], "model_B": models[j], "spearman_channel": spearmanr(a, b).correlation,
                       "same_top_channel": a.idxmax() == b.idxmax(),
                       "top_channel_A": a.idxmax(), "top_channel_B": b.idxmax()})
    save_table(pd.DataFrame(cr), "Table24_channel_consistency",
               "Agreement across models at measurement-channel level (Spearman, top channel)")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ch[models].plot.bar(ax=ax); ax.set_ylabel("Share of importance (%)"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=7)
    save_fig(fig, "Fig14_importance_by_channel", "Importance share per measurement channel across models")

    # ================================================================ C. noise-aware (augmented) training
    log("\n[C] Noise-augmented training (train on noise levels %s, rep 0)" % AUG_NOISE)
    keys = ["run_id", "ref", "tau_ms", "noise", "rep"]
    D = pd.read_csv(os.path.join(S4, "features.csv")).merge(
        pd.read_csv(os.path.join(S4, "physics_features.csv")), on=keys, how="inner")
    D = D[(D.ref == REF) & (D.tau_ms == TAU_MAIN)].copy()
    ind = pd.read_csv(os.path.join(S4, "physics_indicators.txt"), sep="\t", header=None,
                      names=["name", "branch", "dir", "desc"])
    flist = pd.read_csv(os.path.join(S4, "feature_list.txt"), sep="\t", header=None, names=["name", "desc"])
    B_cols = [c for c in flist.name if c not in BASE_DROP] + ["osc_valid"]
    P_cols, dirs = ind.name.tolist(), ind.dir.tolist()
    F_ALL = B_cols + P_cols
    cons = [0] * len(B_cols) + [int(d) for d in dirs]
    runs = D.drop_duplicates("run_id")[["run_id", "label_code"]].reset_index(drop=True)
    n_par = PhysOrdNet(len(P_cols), len(B_cols)).n_params()
    BM = baseline_models(n_par)

    def get(ids, noise, rep):
        s = D[(D.noise == noise) & (D.rep == rep)].set_index("run_id").loc[list(ids)]
        return s.reset_index(), s.label_code.to_numpy().astype(int)

    def get_train(ids):
        parts = [get(ids, nz, 0) for nz in AUG_NOISE]
        return pd.concat([p[0] for p in parts], ignore_index=True), np.concatenate([p[1] for p in parts])

    def fit_model(name, dtr, ytr, seed):
        if name == "MonoOrd-GBM":
            m = MonoOrdGBM(cons, seed).fit(dtr[F_ALL].to_numpy(), ytr)
            return lambda d: m.predict_proba(d[F_ALL].to_numpy())
        if name == "PhysOrd-Net":
            m, pr, _ = fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed)
            return lambda d: m.proba(*pr(d))
        bname = "XGBoost" if (name == "XGBoost" and "XGBoost" in BM) else ("HGB" if name == "XGBoost" else name)
        base, grid = BM[bname]
        est = fit_baseline(bname, base, grid, dtr[F_ALL].to_numpy(), ytr, seed)
        return lambda d: est.predict_proba(d[F_ALL].to_numpy())

    rows, pcr = [], []
    for seed in SEEDS:
        folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).set_index("run_id").fold
        folds = folds.loc[runs.run_id].to_numpy()
        for k in range(N_FOLD):
            tr, te = runs.run_id[folds != k], runs.run_id[folds == k]
            dtr, ytr = get_train(tr)
            tests = {tn: get(te, tn, 1 if tn > 0 else 0) for tn in sorted(D.noise.unique())}
            for name in GEN_MODELS:
                f = fit_model(name, dtr, ytr, seed)
                for tn, (dte, yte) in tests.items():
                    P = f(dte)
                    rows.append({"test_noise_x": tn, "seed": seed, "fold": k, "model": name, "y": yte, "P": P})
                    if tn == TEST_NOISE:
                        sd = dtr[P_cols].std().to_numpy() + 1e-9
                        v = n = 0
                        for j, c in enumerate(P_cols):
                            d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
                            v += int((risk(f(d2)) < risk(P) - 1e-9).sum()); n += len(d2)
                        pcr.append({"seed": seed, "fold": k, "model": name, "v": v, "n": n})
        log(f"  seed {seed} done")
    out = []
    df = pd.DataFrame(rows)
    for (tn, seed, n), g in df.groupby(["test_noise_x", "seed", "model"]):
        y, P = np.concatenate(g.y.values), np.vstack(g.P.values)
        p = P.argmax(1)
        out.append({"test_noise_x": tn, "seed": seed, "model": n, "macro_F1": f1_score(y, p, average="macro"),
                    "recall_unstable": recall_score(y == 2, p == 2)})
    AUG = pd.DataFrame(out).groupby(["test_noise_x", "model"])[["macro_F1", "recall_unstable"]].agg(["mean", "std"])
    AUG.columns = ["_".join(c) for c in AUG.columns]
    AUG = AUG.reset_index()
    pc = pd.DataFrame(pcr).groupby(["seed", "model"])[["v", "n"]].sum()
    pc = (100 * pc.v / pc.n).groupby("model").agg(["mean", "std"]).reset_index()
    pc.columns = ["model", "PCR_violation_%_mean", "PCR_violation_%_std"]
    std = pd.read_csv(os.path.join(T_DIR, "Table13_noise_robustness.csv"))
    cmp_ = AUG.merge(std, on=["test_noise_x", "model"], suffixes=("_augmented", "_standard"))
    save_table(cmp_, "Table25_noise_augmented_training",
               f"Noise-augmented training (train noise {AUG_NOISE}) vs standard (train 1x), test at 0/1/2/5x")
    save_table(pc, "Table26_PCR_noise_augmented", "Physics-violation rate after noise-augmented training (test 1x)")
    fig, ax = plt.subplots(1, len(GEN_MODELS), figsize=(3.2 * len(GEN_MODELS), 3.2), sharey=True)
    for a, n in zip(ax, GEN_MODELS):
        s = cmp_[cmp_.model == n]
        a.errorbar(s.test_noise_x, s.macro_F1_mean_standard, yerr=s.macro_F1_std_standard, marker="o",
                   capsize=3, label="standard (train 1x)")
        a.errorbar(s.test_noise_x, s.macro_F1_mean_augmented, yerr=s.macro_F1_std_augmented, marker="s",
                   capsize=3, label="noise-augmented")
        a.set_title(n); a.set_xlabel("Test noise (x base)")
    ax[0].set_ylabel("Macro F1"); ax[0].legend(fontsize=7)
    save_fig(fig, "Fig15_noise_augmented_training", "Effect of noise-augmented training on robustness")

    # ================================================================ README append
    with open(os.path.join(OUT, "README.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n\nADD-ON (step7b_addon.py)\n")
        for k, (title, _) in TABLES.items():
            fh.write(f"  {k}: {title}\n")
        for k, t_ in FIGS.items():
            fh.write(f"  {k}: {t_}\n")
    open(os.path.join(OUT, "log_addon.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("\n================ SUMMARY ================")
    print(t[["model", "AP_unstable", "recall_at_precision_0.90", "precision_at_recall_0.95"]].to_string(index=False))
    print("\nChannel shares (%):"); print(ch.round(1).to_string())
    print("\nNoise-augmented vs standard (macro F1):")
    print(cmp_[["test_noise_x", "model", "macro_F1_mean_standard", "macro_F1_mean_augmented"]].round(4).to_string(index=False))
    print("\nPCR after augmentation:"); print(pc.round(3).to_string(index=False))
    log(f"\nRuntime: {(time.time() - t_start) / 60:.1f} min\nDONE.")


if __name__ == "__main__":
    main()
