"""
Extended hyperparameter grids, target observability at the decision time and the 500 ms ROCOF
(Tables 11 and 16).
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


# ================================================================== STEP 10
EXT_GRIDS = {   # extended grids for hyperparameters that hit a boundary in Table31
    "LR": {"logisticregression__C": [0.1, 1, 10, 100, 1000]},
    "SVM": {"svc__C": [1, 10, 100, 1000], "svc__gamma": ["scale", 0.1, 0.01]},
    "MLP_pm": {"mlpclassifier__alpha": [1e-6, 1e-5, 1e-4, 1e-2]},
    "XGBoost": {"max_depth": [2, 3, 5], "learning_rate": [0.02, 0.05, 0.1, 0.2]},
    "MonoOrd-GBM": {"max_depth": [2, 3, 5], "learning_rate": [0.02, 0.05, 0.1]},
    "PhysOrd-Net": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
}
ROCOF_WIN = 0.5      # s, alternative ROCOF window


class MonoOrdGBMExt(MonoOrdGBM):
    def _model(self):
        base, _ = super()._model()
        g = EXT_GRIDS["MonoOrd-GBM"]
        if "HistGradient" in type(base).__name__:
            g = {"max_depth": [2, 3, None], "learning_rate": g["learning_rate"]}
        return base, g


def main():
    global TABLES, FIGS, L2_GRID
    TABLES, FIGS = {}, {}
    for d in (OUT, T_DIR, F_DIR, P_DIR):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    log("=" * 64 + "\n STEP 10: EXTENDED-GRID CHECK, TARGET OBSERVABILITY, ROCOF DEFINITION\n" + "=" * 64)
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
    runs = D.drop_duplicates("run_id")[["run_id", "label_code", "scenario", "scenario_name", "fault_dur_s"]].reset_index(drop=True)
    n_par = PhysOrdNet(len(P_cols), len(B_cols)).n_params()
    BM = baseline_models(n_par)
    BM["SVM"] = (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True)), None)
    xgb_name = "XGBoost" if "XGBoost" in BM else "HGB"

    def get(ids, noise, rep):
        s = D[(D.noise == noise) & (D.rep == rep)].set_index("run_id").loc[list(ids)]
        return s.reset_index(), s.label_code.to_numpy().astype(int)

    # ================================================================ A. extended-grid check
    log("\n[A] Extended hyperparameter grids for boundary-selected values (same folds, seeds, noise)")
    L2_GRID = EXT_GRIDS["PhysOrd-Net"]
    rows, sel = [], []
    models = ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "SVM", "LR", "MLP_pm"]
    for seed in SEEDS:
        folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).set_index("run_id").fold.loc[runs.run_id].to_numpy()
        res = {m: ([], [], [0, 0]) for m in models}
        for k in range(N_FOLD):
            dtr, ytr = get(runs.run_id[folds != k], TRAIN_NOISE, 0)
            dte, yte = get(runs.run_id[folds == k], TEST_NOISE, 1)
            sd = dtr[P_cols].std().to_numpy() + 1e-9
            pert = []
            for j, c in enumerate(P_cols):
                d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]; pert.append(d2)
            for name in models:
                if name == "MonoOrd-GBM":
                    gb = MonoOrdGBMExt(cons, seed).fit(dtr[F_ALL].to_numpy(), ytr)
                    f = lambda d, gb=gb: gb.predict_proba(d[F_ALL].to_numpy())
                    for j, m in enumerate(gb.m):
                        p = m.get_params()
                        for hp in ("max_depth", "learning_rate"):
                            sel.append({"model": name, "part": f"y>={j+1}", "hyperparameter": hp, "value": str(p.get(hp))})
                elif name == "PhysOrd-Net":
                    m, pr, l2 = fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed)
                    f = lambda d, m=m, pr=pr: m.proba(*pr(d))
                    sel.append({"model": name, "part": "", "hyperparameter": "lambda", "value": str(l2)})
                else:
                    bname = xgb_name if name == "XGBoost" else name
                    base = BM[bname][0]
                    if bname == "MLP_pm":
                        base, _ = make_mlp_pm(len(F_ALL), n_par)
                    grid = EXT_GRIDS[name]
                    if bname == "HGB":
                        grid = {"max_depth": [2, 3, None], "learning_rate": EXT_GRIDS["XGBoost"]["learning_rate"]}
                    est = set_seed(clone(base), seed)
                    kw = {}
                    if bname == "XGBoost":
                        kw = {"sample_weight": pd.Series(ytr).map(len(ytr) / (3 * pd.Series(ytr).value_counts())).to_numpy()}
                    gs = GridSearchCV(est, grid, scoring="f1_macro", n_jobs=-1,
                                      cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(dtr[F_ALL].to_numpy(), ytr, **kw)
                    f = lambda d, gs=gs: gs.predict_proba(d[F_ALL].to_numpy())
                    for hp, v in gs.best_params_.items():
                        sel.append({"model": name, "part": "", "hyperparameter": hp.split("__")[-1], "value": str(v)})
                P0 = f(dte)
                res[name][0].append(yte); res[name][1].append(P0.argmax(1))
                for d2 in pert:
                    res[name][2][0] += int((risk(f(d2)) < risk(P0) - 1e-9).sum()); res[name][2][1] += len(d2)
        for name, (ys, ps, v) in res.items():
            y, p = np.concatenate(ys), np.concatenate(ps)
            rows.append({"model": name, "seed": seed, "macro_F1": f1_score(y, p, average="macro"),
                         "recall_unstable": recall_score(y == 2, p == 2), "PCR_%": 100 * v[0] / v[1]})
        log(f"  seed {seed} done")
    EXT = pd.DataFrame(rows)
    orig = pd.read_csv(os.path.join(T_DIR, "Table06b_main_comparison_per_seed.csv"))
    comp = []
    for name in models:
        e = EXT[EXT.model == name]; o = orig[orig.model == name]
        comp.append({"model": name,
                     "macro_F1_original": f"{o.macro_F1.mean():.4f} ± {o.macro_F1.std():.4f}",
                     "macro_F1_extended": f"{e.macro_F1.mean():.4f} ± {e.macro_F1.std():.4f}",
                     "delta_F1": round(e.macro_F1.mean() - o.macro_F1.mean(), 4),
                     "recall_unstable_extended": round(e.recall_unstable.mean(), 4),
                     "PCR_%_original": round(o["PCR_violation_%"].mean(), 3), "PCR_%_extended": round(e["PCR_%"].mean(), 3)})
    save_table(pd.DataFrame(comp), "Table33_extended_grid_check",
               "Main comparison repeated with extended grids for hyperparameters selected at a grid boundary (5 seeds x 5 folds)")
    S = pd.DataFrame(sel)
    T = S.groupby(["model", "part", "hyperparameter", "value"]).size().rename("times_selected").reset_index()
    T["share_%"] = (100 * T.times_selected / (len(SEEDS) * N_FOLD)).round(1)
    save_table(T.sort_values(["model", "part", "hyperparameter", "times_selected"], ascending=[True, True, True, False]),
               "Table34_selected_hyperparameters_extended", "Values selected with the extended grids (25 outer partitions)")

    # ================================================================ B/C. target observability and ROCOF
    log("\n[B] Time of each severity extremum relative to the decision time; [C] ROCOF definitions")
    lab = pd.read_csv(os.path.join(ROOT, "_STEP3", "labels_metrics.csv")).set_index("run_id")
    lg = pd.read_csv(os.path.join(ROOT, "run_log.csv"))
    lg = lg[lg.status.astype(str).str.strip() == "ok"].drop_duplicates("run_id", keep="last")
    t_event = dict(zip(lg.run_id, lg.t_event))

    def mavg(x, n):
        n = max(1, int(n)); c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
        y = np.empty_like(x, dtype=float); y[n - 1:] = (c[n:] - c[:-n]) / n; y[:n - 1] = c[1:n] / np.arange(1, n); return y
    obs = []
    for i, r in runs.iterrows():
        df = pd.read_csv(os.path.join(ROOT, "runs", f"{r.run_id}.csv"), usecols=["Time_s", "Vmag_POI_pu", "f_POI_Hz"])
        t = df.Time_s.to_numpy(); dt = float(np.median(np.diff(t))); V = df.Vmag_POI_pu.to_numpy()
        fb = mavg(df.f_POI_Hz.to_numpy(), 0.1 / dt); fb[V < 0.1] = np.nan
        te = float(t_event.get(r.run_id, 4.0))
        fd = r.fault_dur_s
        tref = te + (fd if r.scenario == 4 and np.isfinite(fd) else (0.5 if r.scenario == 5 else 0.0))
        td = tref + TAU_MAIN / 1000.0
        post = t >= te
        iv = np.argmin(np.where(post, V, np.inf)); t_v = t[iv]
        dev = np.where(post & np.isfinite(fb), np.abs(fb - 60.0), -1); t_f = t[np.argmax(dev)] if (dev >= 0).any() else np.nan
        roc = np.gradient(fb, dt); rm = post & (t <= te + 1.0) & np.isfinite(roc)
        t_r = t[np.where(rm)[0][np.argmax(np.abs(roc[rm]))]] if rm.any() else np.nan
        lag = int(round(ROCOF_WIN / dt)); r5 = np.full_like(fb, np.nan); r5[lag:] = (fb[lag:] - fb[:-lag]) / ROCOF_WIN
        m5 = (t >= te + ROCOF_WIN) & (t <= te + 1.5) & np.isfinite(r5)
        rocof5 = float(np.nanmax(np.abs(r5[m5]))) if m5.any() else np.nan
        trec = lab.at[r.run_id, "T_rec"]; t0 = te + (fd if r.scenario == 4 and np.isfinite(fd) else 0.0)
        obs.append({"run_id": r.run_id, "scenario": r.scenario_name, "t_d_after_event_ms": 1000 * (td - te),
                    "dip_observed_by_td": t_v <= td, "fdev_observed_by_td": bool(t_f <= td) if np.isfinite(t_f) else np.nan,
                    "rocof_observed_by_td": bool(t_r <= td) if np.isfinite(t_r) else np.nan,
                    "trec_observed_by_td": bool(t0 + trec <= td) if np.isfinite(trec) else np.nan,
                    "ROCOF_max_original": lab.at[r.run_id, "ROCOF_max"], "ROCOF_500ms": rocof5})
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{len(runs)} runs")
    O = pd.DataFrame(obs)
    O.to_csv(os.path.join(P_DIR, "target_observability_per_run.csv"), index=False)
    cols = ["dip_observed_by_td", "fdev_observed_by_td", "rocof_observed_by_td", "trec_observed_by_td"]
    ob = O.groupby("scenario")[cols].apply(lambda d: 100 * d.astype(float).mean()).round(1)
    ob.loc["ALL"] = (100 * O[cols].astype(float).mean()).round(1)
    save_table(ob.reset_index(), "Table35_target_observability",
               f"Percentage of runs whose severity extremum (or recovery) has already occurred by the decision time (tau = {TAU_MAIN} ms)")
    q = O[["ROCOF_max_original", "ROCOF_500ms"]].describe(percentiles=[0.5, 0.9, 0.99]).T.round(3).reset_index()
    save_table(q, "Table36_ROCOF_distribution", "Distribution of ROCOF (Hz/s): original (gradient of 100 ms-filtered f) vs 500 ms-window ROCOF")

    # regression accuracy on the not-yet-observed subset (stored seed-0 predictions)
    pth = os.path.join(P_DIR, "oof_regression_seed0.csv")
    sub_rows = []
    if os.path.exists(pth):
        RR = pd.read_csv(pth).merge(O, on="run_id")
        cmap = {"voltage_dip_pu": "dip_observed_by_td", "max_freq_dev_Hz": "fdev_observed_by_td",
                "ROCOF_max_Hz_s": "rocof_observed_by_td", "T_rec_s": "trec_observed_by_td"}
        for (tg, mo), g in RR.groupby(["target", "model"]):
            for lab_, m in [("all", np.ones(len(g), bool)), ("not yet observed at t_d", g[cmap[tg]] == False),
                            ("already observed", g[cmap[tg]] == True)]:
                gg = g[m]
                if len(gg) >= 10 and gg.y_true.var() > 0:
                    sub_rows.append({"target": tg, "model": mo, "subset": lab_, "n": len(gg),
                                     "R2": r2_score(gg.y_true, gg.y_pred),
                                     "RMSE": float(np.sqrt(mean_squared_error(gg.y_true, gg.y_pred))),
                                     "MAE": mean_absolute_error(gg.y_true, gg.y_pred)})
        save_table(pd.DataFrame(sub_rows), "Table37_regression_by_observability",
                   "Seed-0 out-of-fold regression accuracy split by whether the target was already observed at the decision time")

    # regression of the 500 ms ROCOF
    yv_all = O.set_index("run_id").ROCOF_500ms
    ok = np.isfinite(yv_all.loc[runs.run_id].to_numpy())
    R_ids = runs.run_id[ok].reset_index(drop=True); yv = yv_all.loc[R_ids].to_numpy()
    try:
        from xgboost import XGBRegressor
        mk = lambda c, s: XGBRegressor(n_estimators=300, tree_method="hist", n_jobs=1, verbosity=0, random_state=s,
                                       **({"monotone_constraints": "(" + ",".join(map(str, c)) + ")"} if c else {}))
        g_ = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        mk = lambda c, s: HistGradientBoostingRegressor(random_state=s, min_samples_leaf=5, **({"monotonic_cst": c} if c else {}))
        g_ = {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]}
    rr = []
    for seed in SEEDS:
        folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).set_index("run_id").fold.loc[R_ids].to_numpy()
        for name, base, grid in [("MonoReg-GBM", mk(cons, seed), g_), ("XGBoost-R", mk(None, seed), g_),
                                 ("RF-R", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
                                  {"max_depth": [None, 8], "min_samples_leaf": [1, 3]})]:
            ys, ps, v, n = [], [], 0, 0
            for k in range(N_FOLD):
                tr, te_ = folds != k, folds == k
                dtr, _ = get(R_ids[tr], TRAIN_NOISE, 0); dte, _ = get(R_ids[te_], TEST_NOISE, 1)
                gs = GridSearchCV(clone(base), grid, scoring="neg_root_mean_squared_error", n_jobs=-1,
                                  cv=KFold(3, shuffle=True, random_state=seed)).fit(dtr[F_ALL].to_numpy(), yv[tr])
                p0 = gs.predict(dte[F_ALL].to_numpy()); ys.append(yv[te_]); ps.append(p0)
                sd = dtr[P_cols].std().to_numpy() + 1e-9
                for j, c in enumerate(P_cols):
                    d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
                    v += int((gs.predict(d2[F_ALL].to_numpy()) < p0 - 1e-9).sum()); n += len(d2)
            y, p = np.concatenate(ys), np.concatenate(ps)
            rr.append({"model": name, "seed": seed, "R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
                       "MAE": mean_absolute_error(y, p), "PCR_%": 100 * v / n})
    RR5 = pd.DataFrame(rr).groupby("model")[["R2", "RMSE", "MAE", "PCR_%"]].agg(["mean", "std"])
    RR5.columns = ["_".join(c) for c in RR5.columns]
    save_table(RR5.reset_index(), "Table38_ROCOF500_regression", f"Regression of the 500 ms-window ROCOF (n = {len(R_ids)} runs, 5 seeds x 5 folds)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a, c in zip(ax, ["ROCOF_max_original", "ROCOF_500ms"]):
        a.hist(O[c].dropna(), bins=40, color="#4a90d9"); a.set_xlabel(f"{c} (Hz/s)"); a.set_ylabel("Runs"); a.set_yscale("log")
    save_fig(fig, "Fig20_ROCOF_distribution", "ROCOF distribution: gradient of 100 ms-filtered frequency vs 500 ms-window ROCOF")

    with open(os.path.join(OUT, "README.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n\nSTEP 10 (step10_checks.py)\n")
        for k_, (title, _) in TABLES.items():
            fh.write(f"  {k_}: {title}\n")
        for k_, t_ in FIGS.items():
            fh.write(f"  {k_}: {t_}\n")
    open(os.path.join(OUT, "log_step10.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("\n================ SUMMARY ================")
    print(pd.DataFrame(comp).to_string(index=False)); print()
    print(T.to_string(index=False)); print()
    print(ob.to_string()); print()
    print(q.to_string(index=False)); print()
    if sub_rows:
        print(pd.DataFrame(sub_rows).round(4).to_string(index=False)); print()
    print(RR5.round(4).to_string())
    log(f"\nRuntime: {(time.time() - t_start) / 60:.1f} min\nDONE.")


if __name__ == "__main__":
    main()
