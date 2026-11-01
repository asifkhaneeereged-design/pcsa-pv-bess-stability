"""
Hyperparameters selected by the inner cross-validation in every outer partition (Table 11).
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


# ================================================================== STEP 9
SEEDS_REG = [0, 1, 2, 3, 4]
REG_TARGETS = {   # name: (column expression, unit, keep only recovered runs?)
    "voltage_dip_pu": ("dip", "pu", False),
    "max_freq_dev_Hz": ("fdev", "Hz", False),
    "ROCOF_max_Hz_s": ("ROCOF_max", "Hz/s", False),
    "T_rec_s": ("T_rec", "s", True),
}
REG_MODELS = ["MonoReg-GBM", "XGBoost-R", "RF-R", "SVR", "Ridge", "MLP-R"]


def main():
    global TABLES, FIGS
    TABLES, FIGS = {}, {}
    for d in (OUT, T_DIR, F_DIR, P_DIR):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    log("=" * 64 + "\n STEP 9: SELECTED HYPERPARAMETERS + SEVERITY REGRESSION (R2, RMSE, MAE)\n" + "=" * 64)
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.compose import TransformedTargetRegressor
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
    runs = D.drop_duplicates("run_id")[["run_id", "label_code"]].reset_index(drop=True)
    n_par = PhysOrdNet(len(P_cols), len(B_cols)).n_params()
    BM = baseline_models(n_par)
    BM["SVM"] = (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True)),
                 {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.1]})
    xgb_name = "XGBoost" if "XGBoost" in BM else "HGB"

    def get(ids, noise, rep):
        s = D[(D.noise == noise) & (D.rep == rep)].set_index("run_id").loc[list(ids)]
        return s.reset_index(), s.label_code.to_numpy().astype(int)

    def folds_of(seed):
        f = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).set_index("run_id").fold
        return f.loc[runs.run_id].to_numpy()

    # ================================================================ A. selected hyperparameters
    log("\n[A] Hyperparameters selected by the inner CV in every outer fold (classification)")
    sel = []
    for seed in SEEDS:
        folds = folds_of(seed)
        for k in range(N_FOLD):
            dtr, ytr = get(runs.run_id[folds != k], TRAIN_NOISE, 0)
            X = dtr[F_ALL].to_numpy()
            # MonoOrd-GBM
            gb = MonoOrdGBM(cons, seed).fit(X, ytr)
            for j, m in enumerate(gb.m):
                p = m.get_params()
                for hp in ("max_depth", "learning_rate"):
                    sel.append({"model": "MonoOrd-GBM", "part": f"ensemble y>={j+1}", "hyperparameter": hp,
                                "value": str(p.get(hp)), "seed": seed, "fold": k})
            # PhysOrd-Net
            _, _, l2 = fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed)
            sel.append({"model": "PhysOrd-Net", "part": "", "hyperparameter": "lambda (L2)", "value": str(l2),
                        "seed": seed, "fold": k})
            # baselines
            for name, bname in [("XGBoost", xgb_name), ("RF", "RF"), ("SVM", "SVM"), ("LR", "LR"), ("MLP_pm", "MLP_pm")]:
                base, grid = BM[bname]
                if bname == "MLP_pm":
                    base, _ = make_mlp_pm(len(F_ALL), n_par)
                est = set_seed(clone(base), seed)
                kw = {}
                if bname == "XGBoost":
                    kw = {"sample_weight": pd.Series(ytr).map(len(ytr) / (3 * pd.Series(ytr).value_counts())).to_numpy()}
                gs = GridSearchCV(est, grid, scoring="f1_macro", n_jobs=-1,
                                  cv=StratifiedKFold(3, shuffle=True, random_state=seed)).fit(X, ytr, **kw)
                for hp, v in gs.best_params_.items():
                    sel.append({"model": name, "part": "", "hyperparameter": hp.split("__")[-1], "value": str(v),
                                "seed": seed, "fold": k})
        log(f"  seed {seed} done")
    S = pd.DataFrame(sel)
    S.to_csv(os.path.join(P_DIR, "selected_hyperparameters_per_fold.csv"), index=False)
    n_fit = len(SEEDS) * N_FOLD
    T = (S.groupby(["model", "part", "hyperparameter", "value"]).size().rename("times_selected").reset_index())
    T["share_%"] = (100 * T.times_selected / n_fit).round(1)
    T = T.sort_values(["model", "part", "hyperparameter", "times_selected"], ascending=[True, True, True, False])
    save_table(T, "Table31_selected_hyperparameters",
               f"Hyperparameter values selected by the inner 3-fold CV across the {n_fit} outer training partitions")

    # ================================================================ B. severity regression
    log("\n[B] Severity regression: same early window, continuous IEEE-2800-related targets")
    meta = D.drop_duplicates("run_id").set_index("run_id")
    tgt = pd.DataFrame(index=runs.run_id)
    tgt["dip"] = 1.0 - meta.V_min.reindex(runs.run_id).to_numpy()
    fn, fp = meta.f_nadir.reindex(runs.run_id), meta.f_peak.reindex(runs.run_id)
    tgt["fdev"] = np.maximum((60.0 - fn).abs(), (fp - 60.0).abs()).to_numpy()
    tgt["ROCOF_max"] = meta.ROCOF_max.reindex(runs.run_id).to_numpy()
    tgt["T_rec"] = meta.T_rec.reindex(runs.run_id).to_numpy()

    def reg_models(seed):
        sc = lambda est: TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), est),
                                                    transformer=StandardScaler())
        m = {}
        try:
            from xgboost import XGBRegressor
            mk = lambda c: XGBRegressor(n_estimators=300, tree_method="hist", n_jobs=1, verbosity=0, random_state=seed,
                                        **({"monotone_constraints": "(" + ",".join(map(str, c)) + ")"} if c else {}))
            g = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor
            mk = lambda c: HistGradientBoostingRegressor(random_state=seed, min_samples_leaf=5,
                                                         **({"monotonic_cst": c} if c else {}))
            g = {"max_depth": [None, 5], "learning_rate": [0.05, 0.1]}
        m["MonoReg-GBM"] = (mk(cons), g)
        m["XGBoost-R"] = (mk(None), g)
        m["RF-R"] = (RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
                     {"max_depth": [None, 8], "min_samples_leaf": [1, 3]})
        m["SVR"] = (sc(SVR(kernel="rbf")), {"regressor__svr__C": [1, 10, 100], "regressor__svr__gamma": ["scale", 0.1]})
        m["Ridge"] = (sc(Ridge()), {"regressor__ridge__alpha": [0.1, 1, 10]})
        m["MLP-R"] = (sc(MLPRegressor(hidden_layer_sizes=(8,), max_iter=3000, early_stopping=True, random_state=seed)),
                      {"regressor__mlpregressor__alpha": [1e-4, 1e-2]})
        return m

    rows, oof_rows = [], []
    for tname, (col, unit, recovered_only) in REG_TARGETS.items():
        ok = np.isfinite(tgt[col].to_numpy())
        if recovered_only:
            ok &= runs.label_code.to_numpy() < 2
        R_ids = runs.run_id[ok].reset_index(drop=True)
        yv = tgt.loc[R_ids, col].to_numpy()
        for seed in SEEDS_REG:
            folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).set_index("run_id").fold.loc[R_ids].to_numpy()
            for name, (base, grid) in reg_models(seed).items():
                ys, ps, ids, v, n = [], [], [], 0, 0
                for k in range(N_FOLD):
                    tr, te = folds != k, folds == k
                    dtr, _ = get(R_ids[tr], TRAIN_NOISE, 0); dte, _ = get(R_ids[te], TEST_NOISE, 1)
                    gs = GridSearchCV(clone(base), grid, scoring="neg_root_mean_squared_error", n_jobs=-1,
                                      cv=KFold(3, shuffle=True, random_state=seed)).fit(dtr[F_ALL].to_numpy(), yv[tr])
                    p0 = gs.predict(dte[F_ALL].to_numpy())
                    ys.append(yv[te]); ps.append(p0); ids += list(R_ids[te])
                    sd = dtr[P_cols].std().to_numpy() + 1e-9
                    for j, c in enumerate(P_cols):
                        d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
                        v += int((gs.predict(d2[F_ALL].to_numpy()) < p0 - 1e-9).sum()); n += len(d2)
                y, p = np.concatenate(ys), np.concatenate(ps)
                rows.append({"target": tname, "unit": unit, "n_runs": len(y), "model": name, "seed": seed,
                             "R2": r2_score(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
                             "MAE": mean_absolute_error(y, p), "PCR_%": 100 * v / n})
                if seed == SEEDS_REG[0]:
                    oof_rows.append(pd.DataFrame({"target": tname, "model": name, "run_id": ids, "y_true": y, "y_pred": p}))
        log(f"  {tname} done")
    RG = pd.DataFrame(rows)
    RG.to_csv(os.path.join(T_DIR, "Table32b_regression_per_seed.csv"), index=False)
    pd.concat(oof_rows).to_csv(os.path.join(P_DIR, "oof_regression_seed0.csv"), index=False)
    agg = RG.groupby(["target", "unit", "n_runs", "model"])[["R2", "RMSE", "MAE", "PCR_%"]].agg(["mean", "std"])
    out = agg.copy(); out.columns = ["_".join(c) for c in out.columns]; out = out.reset_index()
    save_table(out, "Table32_severity_regression",
               f"Severity regression from the {TAU_MAIN} ms window: R2, RMSE, MAE (mean ± SD, {len(SEEDS_REG)} seeds x {N_FOLD} folds) and physics-violation rate")
    # figures
    tn = list(REG_TARGETS)
    fig, axes = plt.subplots(1, len(tn), figsize=(4 * len(tn), 3.6))
    oo = pd.concat(oof_rows)
    for a, t_ in zip(axes, tn):
        for mname, mk_ in [("XGBoost-R", "o"), ("MonoReg-GBM", "s")]:
            q = oo[(oo.target == t_) & (oo.model == mname)]
            a.scatter(q.y_true, q.y_pred, s=10, alpha=0.6, marker=mk_, label=mname)
        lo, hi = oo[oo.target == t_][["y_true", "y_pred"]].min().min(), oo[oo.target == t_][["y_true", "y_pred"]].max().max()
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        r2m = out[(out.target == t_) & (out.model == "MonoReg-GBM")].R2_mean.iloc[0]
        a.set_title(f"{t_}  (MonoReg R² = {r2m:.3f})", fontsize=8.5)
        a.set_xlabel(f"Simulated ({REG_TARGETS[t_][1]})"); a.set_ylabel(f"Predicted ({REG_TARGETS[t_][1]})")
    axes[0].legend(fontsize=7)
    save_fig(fig, "Fig18_severity_regression_parity", "Predicted vs simulated severity quantities (seed 0, out-of-fold)")
    piv = out.pivot(index="model", columns="target", values="R2_mean").loc[REG_MODELS, tn]
    err = out.pivot(index="model", columns="target", values="R2_std").loc[REG_MODELS, tn]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    piv.T.plot.bar(ax=ax, yerr=err.T, capsize=2); ax.set_ylabel("R² (mean ± SD)"); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=7, ncol=3)
    save_fig(fig, "Fig19_severity_regression_R2", "Coefficient of determination per severity target and model")

    with open(os.path.join(OUT, "README.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n\nSTEP 9 (step9_hparams_regression.py)\n")
        for k_, (title, _) in TABLES.items():
            fh.write(f"  {k_}: {title}\n")
        for k_, t_ in FIGS.items():
            fh.write(f"  {k_}: {t_}\n")
    open(os.path.join(OUT, "log_step9.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("\n================ SUMMARY ================")
    print(T.to_string(index=False))
    print()
    print(out[["target", "n_runs", "model", "R2_mean", "R2_std", "RMSE_mean", "MAE_mean", "PCR_%_mean"]].round(4).to_string(index=False))
    log(f"\nRuntime: {(time.time() - t_start) / 60:.1f} min\nDONE.")


if __name__ == "__main__":
    main()
