"""
MonoOrd-GBM, PhysOrd-Net and baselines under run-wise nested cross-validation:
main comparison (Table 14), physical consistency rate, statistical tests, window length,
generalisation, measurement noise, sensor loss and ablation (Table 17).
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


# ================================================================== MAIN
def main():
    global TABLES, FIGS
    TABLES, FIGS = {}, {}
    for d in (OUT, T_DIR, F_DIR, P_DIR):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    log("=" * 64 + "\n STEP 7: JOURNAL RESULTS (all tables, figures, predictions)\n" + "=" * 64)

    # ---------------------------------------------------------------- data
    keys = ["run_id", "ref", "tau_ms", "noise", "rep"]
    D = pd.read_csv(os.path.join(S4, "features.csv")).merge(
        pd.read_csv(os.path.join(S4, "physics_features.csv")), on=keys, how="inner")
    D = D[D.ref == REF].copy()
    ind = pd.read_csv(os.path.join(S4, "physics_indicators.txt"), sep="\t", header=None,
                      names=["name", "branch", "dir", "desc"])
    flist = pd.read_csv(os.path.join(S4, "feature_list.txt"), sep="\t", header=None, names=["name", "desc"])
    base_all = flist.name.tolist()
    B_cols = [c for c in base_all if c not in BASE_DROP] + ["osc_valid"]
    P_cols, dirs = ind.name.tolist(), ind.dir.tolist()
    F_ALL = B_cols + P_cols
    cons = [0] * len(B_cols) + [int(d) for d in dirs]
    runs = D.drop_duplicates("run_id")[["run_id", "label_code", "label", "ctrl", "grid", "SCR", "scenario",
                                         "scenario_name", "fault_dur_s"]].reset_index(drop=True)
    n_par = PhysOrdNet(len(P_cols), len(B_cols)).n_params()
    BM = baseline_models(n_par)
    BM["SVM"] = (make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True)),
                 {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.1]})
    xgb_name = "XGBoost" if "XGBoost" in BM else "HGB"
    alias = {"XGBoost": xgb_name}
    log(f"{len(runs)} run | base {len(B_cols)} + physical {len(P_cols)} feature | REF={REF} | "
        f"tau main={TAU_MAIN} ms | seeds={SEEDS}")

    def get(ids, tau, noise, rep):
        s = D[(D.tau_ms == tau) & (D.noise == noise) & (D.rep == rep)].set_index("run_id").loc[list(ids)]
        return s.reset_index(), s.label_code.to_numpy().astype(int)

    def fit_model(name, dtr, ytr, seed, feats=None):
        F = feats or F_ALL
        if name == "MonoOrd-GBM":
            m = MonoOrdGBM(cons, seed).fit(dtr[F_ALL].to_numpy(), ytr)
            return lambda d: m.predict_proba(d[F_ALL].to_numpy()), m
        if name == "PhysOrd-Net":
            m, pr, _ = fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed)
            return lambda d: m.proba(*pr(d)), m
        bname = alias.get(name, name)
        base, grid = BM[bname]
        if bname == "MLP_pm":
            base, _ = make_mlp_pm(len(F), n_par)
        est = fit_baseline(bname, base, grid, dtr[F].to_numpy(), ytr, seed)
        return lambda d: est.predict_proba(d[F].to_numpy()), est

    # ================================================================ T1-T3 dataset & setup
    log("\n[1] Dataset audit")
    excl = pd.read_csv(os.path.join(S4, "excluded_runs.csv"))
    lab_all = pd.read_csv(os.path.join(ROOT, "_STEP3", "labels_metrics.csv"))
    t1 = pd.crosstab([runs.scenario_name, runs.ctrl], runs.label).reindex(
        columns=CLASSES, fill_value=0).reset_index()
    t1.loc[len(t1)] = ["TOTAL", "", *runs.label.value_counts().reindex(CLASSES).fillna(0).astype(int)]
    save_table(t1, "Table01_dataset_by_scenario_controller",
               f"Learning dataset: {len(runs)} runs ({len(lab_all)} simulated, {len(excl)} excluded as "
               f"not steady before the event)")
    t1b = pd.crosstab([runs.grid, runs.SCR.round(2)], [runs.ctrl, runs.label]).reset_index()
    t1b.columns = [" ".join(map(str, c)).strip() for c in t1b.columns]
    save_table(t1b, "Table02_dataset_by_SCR", "Class counts per grid level (SCR) and controller")
    save_table(excl, "Table03_excluded_runs", "Runs excluded: not in steady state before the event")
    feat_tab = pd.concat([
        pd.DataFrame({"feature": B_cols, "group": "base (unconstrained)", "risk_direction": 0,
                      "description": [dict(zip(flist.name, flist.desc)).get(c, "1 if a mode estimate exists")
                                      for c in B_cols]}),
        pd.DataFrame({"feature": P_cols, "group": ["physical: " + b for b in ind.branch],
                      "risk_direction": dirs, "description": ind.desc})])
    save_table(feat_tab, "Table04_features_and_physical_indicators",
               "Model inputs: base features and physical indicators with monotone risk direction")
    hp = [{"model": "MonoOrd-GBM (proposed)", "search": "per cumulative model: max_depth {3,5}, "
           "learning_rate {0.05,0.1}, n_estimators 300; monotone constraints on 14 indicators", "inner_cv": "3-fold, ROC-AUC"},
          {"model": "PhysOrd-Net (proposed)", "search": f"L2 {L2_GRID}, J={J_UNITS}, H={H_HIDDEN}, L-BFGS-B "
           f"max {MAXITER} it, params {n_par}", "inner_cv": "3-fold, macro F1"}]
    for k, (b, g) in BM.items():
        hp.append({"model": k, "search": "; ".join(f"{a.split('__')[-1]} {v}" for a, v in g.items()),
                   "inner_cv": "3-fold, macro F1"})
    save_table(pd.DataFrame(hp), "Table05_hyperparameter_search",
               "Hyperparameter search space (nested: tuned inside each outer training fold)")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    pd.crosstab(runs.scenario_name, runs.label).reindex(columns=CLASSES).plot.bar(
        stacked=True, ax=ax[0], color=["#2e9d4a", "#f2c037", "#d1453b"])
    ax[0].set_ylabel("Runs"); ax[0].set_xlabel(""); ax[0].tick_params(axis="x", rotation=25)
    pd.crosstab([runs.SCR.round(2)], runs.label).reindex(columns=CLASSES).plot.bar(
        stacked=True, ax=ax[1], color=["#2e9d4a", "#f2c037", "#d1453b"], legend=False)
    ax[1].set_xlabel("SCR"); ax[1].set_ylabel("Runs")
    save_fig(fig, "Fig01_dataset_class_distribution", "Class distribution by disturbance type and SCR")
    sm = os.path.join(ROOT, "_STEP3", "fig_stability_map.png")
    if os.path.exists(sm):
        import shutil
        shutil.copy(sm, os.path.join(F_DIR, "Fig02_stability_map.png"))
        FIGS["Fig02_stability_map"] = "Simulated stability class per case x SCR (GFL vs VSM), from step 3"

    # ================================================================ T6 main comparison
    log("\n[2] Main comparison (repeated stratified 5-fold CV, nested tuning)")
    rows, oof, pcr_rows, pcr_ind, timing = [], {}, [], [], {}
    for seed in SEEDS:
        skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=seed)
        folds = np.zeros(len(runs), int)
        for k, (_, b) in enumerate(skf.split(runs.run_id, runs.label_code)):
            folds[b] = k
        pd.DataFrame({"run_id": runs.run_id, "fold": folds}).to_csv(
            os.path.join(P_DIR, f"folds_seed{seed}.csv"), index=False)
        for name in MAIN_MODELS:
            Ps, ys, ids, viol, tfit = [], [], [], np.zeros((2, len(P_cols))), []
            for k in range(N_FOLD):
                tr, te = runs.run_id[folds != k], runs.run_id[folds == k]
                dtr, ytr = get(tr, TAU_MAIN, TRAIN_NOISE, 0)
                dte, yte = get(te, TAU_MAIN, TEST_NOISE, 1)
                t0 = time.perf_counter()
                f, _ = fit_model(name, dtr, ytr, seed)
                tfit.append(time.perf_counter() - t0)
                P0 = f(dte)
                Ps.append(P0); ys.append(yte); ids += list(te)
                sd = dtr[P_cols].std().to_numpy() + 1e-9
                for j, c in enumerate(P_cols):
                    d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
                    viol[0, j] += int((risk(f(d2)) < risk(P0) - 1e-9).sum()); viol[1, j] += len(d2)
                if seed == SEEDS[0] and k == 0:
                    x1 = dte.iloc[:1]; t0 = time.perf_counter()
                    for _ in range(100):
                        f(x1)
                    timing[name] = {"train_incl_tuning_s": None, "inference_ms_per_sample": (time.perf_counter() - t0) * 10}
            P, y = np.vstack(Ps), np.concatenate(ys)
            rows.append({"model": name, "seed": seed, **full_metrics(y, P),
                         "PCR_violation_%": 100 * viol[0].sum() / viol[1].sum()})
            for j, c in enumerate(P_cols):
                pcr_ind.append({"model": name, "seed": seed, "indicator": c,
                                "violation_%": 100 * viol[0, j] / viol[1, j]})
            if seed == SEEDS[0]:
                timing[name]["train_incl_tuning_s"] = float(np.mean(tfit))
            out = pd.DataFrame({"run_id": ids, "y_true": y, "P_stable": P[:, 0], "P_alert": P[:, 1],
                                "P_unstable": P[:, 2], "y_pred": P.argmax(1)})
            out.to_csv(os.path.join(P_DIR, f"oof_{name}_seed{seed}.csv"), index=False)
            if seed == SEEDS[0]:
                oof[name] = out
        log(f"  seed {seed} done")
    R = pd.DataFrame(rows)
    agg = R.drop(columns="seed").groupby("model").agg(["mean", "std"])
    t6 = pd.DataFrame({"model": agg.index})
    for c in R.columns.drop(["model", "seed"]):
        t6[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(agg[(c, "mean")], agg[(c, "std")])]
    t6["params_or_size"] = [str(n_par) if m == "PhysOrd-Net" else "" for m in t6.model]
    t6["train_s (incl. tuning)"] = [f"{timing[m]['train_incl_tuning_s']:.2f}" for m in t6.model]
    t6["inference_ms/sample"] = [f"{timing[m]['inference_ms_per_sample']:.3f}" for m in t6.model]
    t6 = t6.set_index("model").loc[MAIN_MODELS].reset_index()
    save_table(t6, "Table06_main_comparison",
               f"Main comparison at {TAU_MAIN} ms after event / fault clearing, mean ± SD over {len(SEEDS)} seeds "
               f"x {N_FOLD} folds (train noise 1x rep0, test noise 1x rep1)")
    R.to_csv(os.path.join(T_DIR, "Table06b_main_comparison_per_seed.csv"), index=False)
    pi = pd.DataFrame(pcr_ind).groupby(["indicator", "model"])["violation_%"].mean().unstack()
    save_table(pi.reset_index(), "Table07_PCR_by_indicator",
               "Physical consistency: % of tests where risk fell when the indicator moved in its risk direction")

    # ---- figures: comparison, PCR, confusion, ROC/PR, calibration
    m_ = R.groupby("model")[["macro_F1", "recall_unstable", "PCR_violation_%"]].agg(["mean", "std"]).loc[MAIN_MODELS]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, (col, lab) in zip(ax, [("macro_F1", "Macro F1"), ("recall_unstable", "Unstable recall"),
                                  ("PCR_violation_%", "Physics violations (%)")]):
        a.bar(MAIN_MODELS, m_[(col, "mean")], yerr=m_[(col, "std")], capsize=3,
              color=["#1f5fa8", "#4a90d9"] + ["#9aa5b1"] * (len(MAIN_MODELS) - 2))
        a.set_title(lab); a.tick_params(axis="x", rotation=40)
    ax[0].set_ylim(max(0, m_[("macro_F1", "mean")].min() - 0.1), 1)
    save_fig(fig, "Fig03_model_comparison", "Macro F1, unstable recall and physics-violation rate (mean ± SD)")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, name in zip(axes, ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost"]):
        o = oof[name]
        cm = confusion_matrix(o.y_true, o.y_pred, labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                a.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8)
        a.set_xticks(range(3)); a.set_xticklabels(CLASSES); a.set_yticks(range(3)); a.set_yticklabels(CLASSES)
        a.set_title(name); a.set_xlabel("Predicted"); a.grid(False)
    axes[0].set_ylabel("True")
    save_fig(fig, "Fig04_confusion_matrices", f"Confusion matrices (seed {SEEDS[0]}, all folds)")

    from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for name in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"]:
        o = oof[name]; yb = (o.y_true == 2).astype(int)
        fpr, tpr, _ = roc_curve(yb, o.P_unstable); ax[0].plot(fpr, tpr, label=f"{name} AUC={auc(fpr, tpr):.3f}")
        pr_, rc_, _ = precision_recall_curve(yb, o.P_unstable)
        ax[1].plot(rc_, pr_, label=f"{name} AP={average_precision_score(yb, o.P_unstable):.3f}")
        bins = np.linspace(0, 1, 11); mid, frac = [], []
        for a_, b_ in zip(bins[:-1], bins[1:]):
            m = (o.P_unstable > a_) & (o.P_unstable <= b_)
            if m.sum() >= 3:
                mid.append(o.P_unstable[m].mean()); frac.append(yb[m].mean())
        ax[2].plot(mid, frac, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.6); ax[2].plot([0, 1], [0, 1], "k--", lw=0.6)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (unstable vs rest)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (unstable)")
    ax[2].set(xlabel="Predicted P(unstable)", ylabel="Observed frequency", title="Reliability diagram")
    for a in ax:
        a.legend(fontsize=7)
    save_fig(fig, "Fig05_ROC_PR_calibration", "ROC, precision-recall and calibration for the unstable class")

    # ================================================================ T8 statistics
    log("\n[3] Statistical tests")
    from scipy.stats import binomtest, wilcoxon
    rng = np.random.default_rng(0)
    st = []
    for prop in ["MonoOrd-GBM", "PhysOrd-Net"]:
        a = oof[prop]
        for other in [m for m in MAIN_MODELS if m != prop]:
            b = oof[other]
            assert (a.run_id.values == b.run_id.values).all()
            y, pa, pb = a.y_true.values, a.y_pred.values, b.y_pred.values
            d0 = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
            bs = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(y), len(y))
                bs.append(f1_score(y[i], pa[i], average="macro") - f1_score(y[i], pb[i], average="macro"))
            bs = np.array(bs)
            n10, n01 = int(((pa == y) & (pb != y)).sum()), int(((pa != y) & (pb == y)).sum())
            ds = (R[R.model == prop].macro_F1.values - R[R.model == other].macro_F1.values)
            try:
                pw = wilcoxon(ds).pvalue if np.any(ds != 0) else 1.0
            except ValueError:
                pw = np.nan
            pcr_d = R[R.model == other]["PCR_violation_%"].mean() - R[R.model == prop]["PCR_violation_%"].mean()
            st.append({"proposed": prop, "vs": other, "dF1 (seed0 OOF)": round(d0, 4),
                       "bootstrap 95% CI": f"[{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]",
                       "bootstrap p (2-sided)": round(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 4),
                       "McNemar prop_right/other_wrong": n10, "McNemar prop_wrong/other_right": n01,
                       "McNemar p": round(binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0, 4),
                       "dF1 over seeds (mean ± SD)": f"{ds.mean():.4f} ± {ds.std(ddof=1):.4f}",
                       "Wilcoxon p (seeds)": pw, "PCR reduction (pp)": round(pcr_d, 3)})
    save_table(pd.DataFrame(st), "Table08_statistical_tests",
               f"Paired comparison: bootstrap ({N_BOOT}) CI of macro-F1 difference, McNemar, Wilcoxon over seeds")

    # ================================================================ T9 time-resolved
    log("\n[4] Time-resolved (observation window)")
    tr_rows = []
    for tau in TAUS_SWEEP:
        for seed in SEEDS:
            folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).fold.values
            for name in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost"]:
                Ps, ys = [], []
                for k in range(N_FOLD):
                    dtr, ytr = get(runs.run_id[folds != k], tau, TRAIN_NOISE, 0)
                    dte, yte = get(runs.run_id[folds == k], tau, TEST_NOISE, 1)
                    f, _ = fit_model(name, dtr, ytr, seed)
                    Ps.append(f(dte)); ys.append(yte)
                mt = full_metrics(np.concatenate(ys), np.vstack(Ps))
                tr_rows.append({"tau_ms": tau, "seed": seed, "model": name, "macro_F1": mt["macro_F1"],
                                "recall_unstable": mt["recall_unstable"]})
        log(f"  tau {tau} ms done")
    TR = pd.DataFrame(tr_rows).groupby(["tau_ms", "model"])[["macro_F1", "recall_unstable"]].agg(["mean", "std"])
    TR.columns = ["_".join(c) for c in TR.columns]
    save_table(TR.reset_index(), "Table09_time_resolved",
               "Performance vs observation window after event / fault clearing")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for name in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost"]:
        s = TR.reset_index().query("model == @name")
        ax[0].errorbar(s.tau_ms, s.macro_F1_mean, yerr=s.macro_F1_std, marker="o", capsize=3, label=name)
        ax[1].errorbar(s.tau_ms, s.recall_unstable_mean, yerr=s.recall_unstable_std, marker="o", capsize=3, label=name)
    for a, l in zip(ax, ["Macro F1", "Unstable recall"]):
        a.set_xscale("log"); a.set_xlabel("Observation window (ms)"); a.set_ylabel(l); a.legend(fontsize=7)
    save_fig(fig, "Fig06_time_resolved", "Macro F1 and unstable recall vs observation window")

    # ================================================================ T10-T14 generalization & robustness
    log("\n[5] Generalization and robustness")

    def evaluate(tr_ids, te_ids, seed, test_noise=TEST_NOISE, mutate=None, models=GEN_MODELS):
        dtr, ytr = get(tr_ids, TAU_MAIN, TRAIN_NOISE, 0)
        dte, yte = get(te_ids, TAU_MAIN, test_noise, 1 if test_noise > 0 else 0)
        if mutate:
            dte = mutate(dtr, dte)
        out = {}
        for name in models:
            f, _ = fit_model(name, dtr, ytr, seed)
            out[name] = (yte, f(dte))
        return out

    g_rows = []
    for g in sorted(runs.grid.unique()):
        for seed in SEEDS:
            res = evaluate(runs.run_id[runs.grid != g], runs.run_id[runs.grid == g], seed)
            for n, (y, P) in res.items():
                g_rows.append({"held_out_level": g, "SCR": runs[runs.grid == g].SCR.iloc[0], "seed": seed,
                               "model": n, "macro_F1": f1_score(y, P.argmax(1), average="macro"),
                               "accuracy": accuracy_score(y, P.argmax(1))})
    G = pd.DataFrame(g_rows)
    Gs = G.groupby(["held_out_level", "SCR", "model"])[["macro_F1", "accuracy"]].agg(["mean", "std"])
    Gs.columns = ["_".join(c) for c in Gs.columns]
    save_table(Gs.reset_index(), "Table10_unseen_SCR", "Leave-one-SCR-level-out (grid strength never seen in training)")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for n in GEN_MODELS:
        s = Gs.reset_index().query("model == @n")
        ax.errorbar(s.SCR, s.macro_F1_mean, yerr=s.macro_F1_std, marker="o", capsize=3, label=n)
    ax.set_xscale("log"); ax.set_xlabel("Held-out SCR"); ax.set_ylabel("Macro F1"); ax.legend(fontsize=7)
    save_fig(fig, "Fig07_unseen_SCR", "Macro F1 on a grid strength not seen in training")

    c_rows = []
    for a_, b_ in [("GFL", "VSM"), ("VSM", "GFL")]:
        for seed in SEEDS:
            for n, (y, P) in evaluate(runs.run_id[runs.ctrl == a_], runs.run_id[runs.ctrl == b_], seed).items():
                c_rows.append({"train": a_, "test": b_, "seed": seed, "model": n,
                               "macro_F1": f1_score(y, P.argmax(1), average="macro")})
    C = pd.DataFrame(c_rows).groupby(["train", "test", "model"]).macro_F1.agg(["mean", "std"]).reset_index()
    save_table(C, "Table11_cross_controller", "Train on one BESS controller, test on the other")

    d_rows = []
    for sc in sorted(s for s in runs.scenario.unique() if s != 0):
        res = evaluate(runs.run_id[runs.scenario != sc], runs.run_id[runs.scenario == sc], SEEDS[0])
        for n, (y, P) in res.items():
            d_rows.append({"held_out": runs[runs.scenario == sc].scenario_name.iloc[0], "n_test": len(y),
                           "model": n, "accuracy": accuracy_score(y, P.argmax(1)),
                           "recall_unstable": recall_score(y == 2, P.argmax(1) == 2, zero_division=np.nan)})
    save_table(pd.DataFrame(d_rows), "Table12_unseen_disturbance", "Leave-one-disturbance-type-out (stress test)")

    n_rows, m_rows = [], []
    for seed in SEEDS:
        folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).fold.values
        for k in range(N_FOLD):
            tr, te = runs.run_id[folds != k], runs.run_id[folds == k]
            for tn in sorted(D.noise.unique()):
                for n, (y, P) in evaluate(tr, te, seed, test_noise=tn).items():
                    n_rows.append({"test_noise_x": tn, "seed": seed, "fold": k, "model": n, "y": y, "p": P.argmax(1)})
            for ch, kk in CHANNELS.items():
                cols = channel_cols(F_ALL, kk)

                def mut(dtr, dte, cols=cols):
                    d2 = dte.copy()
                    for c in cols:
                        d2[c] = dtr[c].median()
                    return d2
                for n, (y, P) in evaluate(tr, te, seed, mutate=mut).items():
                    m_rows.append({"missing_channel": ch, "n_features_replaced": len(cols), "seed": seed,
                                   "fold": k, "model": n, "y": y, "p": P.argmax(1)})
        log(f"  robustness seed {seed} done")

    def pool(rows, key):
        df = pd.DataFrame(rows)
        out = []
        for (kv, seed, n), g in df.groupby([key, "seed", "model"]):
            y, p = np.concatenate(g.y.values), np.concatenate(g.p.values)
            out.append({key: kv, "seed": seed, "model": n, "macro_F1": f1_score(y, p, average="macro"),
                        "recall_unstable": recall_score(y == 2, p == 2)})
        o = pd.DataFrame(out).groupby([key, "model"])[["macro_F1", "recall_unstable"]].agg(["mean", "std"])
        o.columns = ["_".join(c) for c in o.columns]
        return o.reset_index()
    NZ, MS = pool(n_rows, "test_noise_x"), pool(m_rows, "missing_channel")
    save_table(NZ, "Table13_noise_robustness",
               "PMU-like noise (IEEE C37.118.1-based sigma) at test: 0x (clean), 1x, 2x, 5x; train at 1x")
    save_table(MS, "Table14_missing_channel",
               "Missing measurement channel at test (features replaced by training median)")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for n in GEN_MODELS:
        s = NZ.query("model == @n")
        ax[0].errorbar(s.test_noise_x, s.macro_F1_mean, yerr=s.macro_F1_std, marker="o", capsize=3, label=n)
    ax[0].set(xlabel="Test noise (x PMU base sigma)", ylabel="Macro F1"); ax[0].legend(fontsize=7)
    piv = MS.pivot(index="missing_channel", columns="model", values="macro_F1_mean")[GEN_MODELS]
    piv.plot.bar(ax=ax[1]); ax[1].set_ylabel("Macro F1"); ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=20)
    ax[1].legend(fontsize=7)
    save_fig(fig, "Fig08_noise_missing_channel", "Robustness to measurement noise and a missing channel")

    # ================================================================ T15 prediction vs detection, errors
    log("\n[6] Prediction vs detection, error concentration")
    clean = D[(D.tau_ms == TAU_MAIN) & (D.noise == 0)].set_index("run_id")
    pv = []
    for n in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"]:
        o = oof[n].copy(); o["evident"] = o.run_id.map(clean.dV_min_cum) < -0.9
        u = o[o.y_true == 2]
        for lab_, g in [("all unstable", u), ("not yet evident at decision", u[~u.evident]),
                        ("collapse already evident", u[u.evident])]:
            pv.append({"model": n, "subset": lab_, "n": len(g), "recall_unstable": (g.y_pred == 2).mean()})
    meta = runs.set_index("run_id")
    dec = runs.apply(lambda r: 1000 * r.fault_dur_s + TAU_MAIN if r.scenario == 4 else
                     (500 + TAU_MAIN if r.scenario == 5 else TAU_MAIN), axis=1)
    pv_t = pd.DataFrame(pv)
    save_table(pv_t, "Table15_prediction_vs_detection",
               "Unstable recall when collapse is / is not yet visible at decision time")
    save_table(pd.DataFrame({"scenario": runs.scenario_name, "decision_ms": dec}).groupby("scenario")
               .decision_ms.agg(["mean", "min", "max"]).reset_index(), "Table16_decision_time",
               "Decision time after the event (ms): window start (event or fault clearing) + tau")
    o = oof["MonoOrd-GBM"].copy()
    o["err"] = (o.y_true != o.y_pred).astype(int)
    o = o.join(meta[["scenario_name", "grid", "ctrl", "fault_dur_s"]], on="run_id")
    er = o.groupby(["scenario_name", "grid"]).err.mean().unstack() * 100
    save_table(er.reset_index(), "Table17_error_by_scenario_SCR", "MonoOrd-GBM error rate (%) by disturbance and grid level")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(er.values, cmap="Reds", vmin=0, vmax=max(20, np.nanmax(er.values)))
    ax.set_xticks(range(er.shape[1])); ax.set_xticklabels(er.columns)
    ax.set_yticks(range(er.shape[0])); ax.set_yticklabels(er.index)
    for i in range(er.shape[0]):
        for j in range(er.shape[1]):
            ax.text(j, i, f"{er.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Error rate (%)"); ax.grid(False)
    save_fig(fig, "Fig09_error_concentration", "Where MonoOrd-GBM errors concentrate (disturbance x SCR)")

    # ================================================================ T18 explainability
    log("\n[7] Explainability and consistency")
    folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{SEEDS[0]}.csv")).fold.values
    imp = {n: np.zeros(len(F_ALL)) for n in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"]}
    prng = np.random.default_rng(0)

    def perm_importance(f, d, y, n_rep=5):
        base = f1_score(y, f(d).argmax(1), average="macro")
        out = np.zeros(len(F_ALL))
        for j, c in enumerate(F_ALL):
            drops = []
            for _ in range(n_rep):
                d2 = d.copy(); d2[c] = prng.permutation(d2[c].to_numpy())
                drops.append(base - f1_score(y, f(d2).argmax(1), average="macro"))
            out[j] = np.mean(drops)
        return out
    for k in range(N_FOLD):
        dtr, ytr = get(runs.run_id[folds != k], TAU_MAIN, TRAIN_NOISE, 0)
        dte, yte = get(runs.run_id[folds == k], TAU_MAIN, TEST_NOISE, 1)
        for n in imp:
            f, _ = fit_model(n, dtr, ytr, SEEDS[0])
            imp[n] += perm_importance(f, dte, yte) / N_FOLD
    IMP = pd.DataFrame(imp, index=F_ALL)
    IMP["group"] = ["physical: " + dict(zip(ind.name, ind.branch))[c] if c in P_cols else "base" for c in F_ALL]
    save_table(IMP.reset_index().rename(columns={"index": "feature"}).sort_values("MonoOrd-GBM", ascending=False),
               "Table18_permutation_importance", "Out-of-fold permutation importance (drop in macro F1)")
    from scipy.stats import spearmanr
    ms = list(imp)
    cons_rows = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = IMP[ms[i]], IMP[ms[j]]
            top = lambda s, k: set(s.sort_values(ascending=False).index[:k])
            cons_rows.append({"model_A": ms[i], "model_B": ms[j], "spearman": spearmanr(a, b).correlation,
                              "top5_overlap": len(top(a, 5) & top(b, 5)) / 5,
                              "top10_overlap": len(top(a, 10) & top(b, 10)) / 10})
    save_table(pd.DataFrame(cons_rows), "Table19_explanation_consistency",
               "Agreement of importance rankings across models (Spearman, top-k overlap)")
    grp = IMP.groupby("group")[ms].apply(lambda d: d.clip(lower=0).sum())
    grp = 100 * grp / grp.sum()
    save_table(grp.reset_index(), "Table20_importance_by_group", "Share of importance per feature group (%)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    top15 = IMP.sort_values("MonoOrd-GBM", ascending=False).head(15)[::-1]
    ax[0].barh(top15.index, top15["MonoOrd-GBM"], color=["#1f5fa8" if g != "base" else "#9aa5b1" for g in top15.group])
    ax[0].set_xlabel("Permutation importance (macro F1 drop)"); ax[0].set_title("MonoOrd-GBM, top 15 (blue = physical)")
    cm_ = np.ones((len(ms), len(ms)))
    for r_ in cons_rows:
        i, j = ms.index(r_["model_A"]), ms.index(r_["model_B"])
        cm_[i, j] = cm_[j, i] = r_["spearman"]
    im = ax[1].imshow(cm_, cmap="viridis", vmin=0, vmax=1)
    ax[1].set_xticks(range(len(ms))); ax[1].set_xticklabels(ms, rotation=30); ax[1].set_yticks(range(len(ms)))
    ax[1].set_yticklabels(ms); ax[1].grid(False)
    for i in range(len(ms)):
        for j in range(len(ms)):
            ax[1].text(j, i, f"{cm_[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    ax[1].set_title("Spearman rank consistency"); fig.colorbar(im, ax=ax[1])
    save_fig(fig, "Fig10_importance_consistency", "Feature importance and cross-model explanation consistency")

    # PhysOrd shape functions
    dall, yall = get(runs.run_id, TAU_MAIN, TRAIN_NOISE, 0)
    m, pr, _ = fit_physord(dall, yall, P_cols, B_cols, dirs, SEEDS[0])
    ncol = 5; nrow = int(np.ceil(len(P_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.6 * nrow))
    for k, c in enumerate(P_cols):
        a = axes.flat[k]; raw = dall[c].to_numpy()
        gr = np.linspace(np.percentile(raw, 1), np.percentile(raw, 99), 200)
        a.plot(gr, m.shape_curve(k, ((gr - pr.mp[k]) / pr.sp[k]) * dirs[k]), lw=1.8)
        a.set_title(f"{c} ({'+' if dirs[k] > 0 else '-'})", fontsize=8); a.tick_params(labelsize=7)
    for a in axes.flat[len(P_cols):]:
        a.axis("off")
    fig.supylabel("Contribution to risk score")
    save_fig(fig, "Fig11_PhysOrd_shape_functions", "PhysOrd-Net monotone shape function per physical indicator")

    # counter-physical illustration
    dtr, ytr = get(runs.run_id[folds != 0], TAU_MAIN, TRAIN_NOISE, 0)
    dte, _ = get(runs.run_id[folds == 0], TAU_MAIN, TEST_NOISE, 1)
    fx = {n: fit_model(n, dtr, ytr, SEEDS[0])[0] for n in ["XGBoost", "MonoOrd-GBM", "PhysOrd-Net"]}
    sd = dtr[P_cols].std().to_numpy() + 1e-9
    best = (0, None, None)
    for j, c in enumerate(P_cols):
        d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
        drop = risk(fx["XGBoost"](dte)) - risk(fx["XGBoost"](d2))
        i = int(np.argmax(drop))
        if drop[i] > best[0]:
            best = (drop[i], j, i)
    if best[1] is not None:
        j, i = best[1], best[2]; c = P_cols[j]
        deltas = np.linspace(0, 3, 31)
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for n, f in fx.items():
            vals = []
            for dlt in deltas:
                d2 = dte.iloc[[i]].copy(); d2[c] = d2[c] + dirs[j] * dlt * sd[j]
                vals.append(risk(f(d2))[0])
            ax.plot(deltas, vals, marker=".", label=n)
        ax.set_xlabel(f"Shift of {c} in its risk direction (training SD)")
        ax.set_ylabel("Expected severity  P(alert)+2P(unstable)")
        ax.set_title(f"Run {dte.run_id.iloc[i]}", fontsize=8); ax.legend(fontsize=7)
        save_fig(fig, "Fig12_counter_physical_example",
                 "Example: unconstrained model lowers risk as a physical indicator worsens; monotone models do not")

    # ================================================================ ablation
    log("\n[8] Ablation")
    ab = []
    variants = {"MonoOrd-GBM (full)": ("mono", None), "GBM unconstrained, base+physical": ("xgb", F_ALL),
                "GBM unconstrained, base only": ("xgb", base_all),
                "PhysOrd-Net (full)": ("po", (P_cols, B_cols, True)),
                "PhysOrd-Net, no monotone": ("po", (P_cols, B_cols, False)),
                "PhysOrd-Net, physical only": ("po", (P_cols, [], True))}
    for br in ["osc", "ride", "rec"]:
        variants[f"PhysOrd-Net, -{br} branch"] = ("po", ([c for c in P_cols if dict(zip(ind.name, ind.branch))[c] != br],
                                                      B_cols, True))
    for seed in SEEDS:
        folds = pd.read_csv(os.path.join(P_DIR, f"folds_seed{seed}.csv")).fold.values
        for vn, (kind, arg) in variants.items():
            ys, ps = [], []
            for k in range(N_FOLD):
                dtr, ytr = get(runs.run_id[folds != k], TAU_MAIN, TRAIN_NOISE, 0)
                dte, yte = get(runs.run_id[folds == k], TAU_MAIN, TEST_NOISE, 1)
                if kind == "mono":
                    f, _ = fit_model("MonoOrd-GBM", dtr, ytr, seed)
                elif kind == "xgb":
                    f, _ = fit_model("XGBoost", dtr, ytr, seed, feats=arg)
                else:
                    pc, bc, mono = arg
                    mm, pp, _ = fit_physord(dtr, ytr, pc, bc, [dirs[P_cols.index(c)] for c in pc], seed, mono=mono)
                    f = (lambda d, mm=mm, pp=pp: mm.proba(*pp(d)))
                ys.append(yte); ps.append(f(dte).argmax(1))
            y, p = np.concatenate(ys), np.concatenate(ps)
            ab.append({"variant": vn, "seed": seed, "macro_F1": f1_score(y, p, average="macro"),
                       "recall_unstable": recall_score(y == 2, p == 2)})
    AB = pd.DataFrame(ab).groupby("variant")[["macro_F1", "recall_unstable"]].agg(["mean", "std"])
    AB.columns = ["_".join(c) for c in AB.columns]
    save_table(AB.reset_index(), "Table21_ablation", "Ablation of constraints, physical indicators and branches")

    # ================================================================ config, environment, README
    import sklearn, scipy
    env = {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
           "cpu_count": os.cpu_count(), "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    try:
        import xgboost
        env["xgboost"] = xgboost.__version__
    except ImportError:
        env["xgboost"] = "not installed (HistGradientBoosting used)"
    cfg = {"ROOT": ROOT, "REF": REF, "TAU_MAIN_ms": TAU_MAIN, "TAUS_SWEEP_ms": TAUS_SWEEP, "SEEDS": SEEDS,
           "N_FOLD": N_FOLD, "TRAIN_NOISE": TRAIN_NOISE, "TEST_NOISE": TEST_NOISE, "PCR_DELTA_sd": PCR_DELTA,
           "PhysOrd": {"J": J_UNITS, "H": H_HIDDEN, "L2_grid": L2_GRID, "maxiter": MAXITER, "params": n_par},
           "BASE_DROP": BASE_DROP, "N_BOOT": N_BOOT, "missing_channels": CHANNELS,
           "runtime_min": round((time.time() - t_start) / 60, 1)}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w"), indent=2)
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=2)
    with pd.ExcelWriter(os.path.join(OUT, "ALL_TABLES.xlsx"), engine="openpyxl") as w:
        for k, (title, df) in TABLES.items():
            df.to_excel(w, sheet_name=k[:31], index=False)
    with open(os.path.join(OUT, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write("JOURNAL RESULTS - generated by step7_journal_results.py\n")
        fh.write(f"Runtime: {cfg['runtime_min']} min | seeds {SEEDS} | REF {REF} | tau {TAU_MAIN} ms\n\n")
        fh.write("TABLES (tables/*.csv, all in ALL_TABLES.xlsx)\n")
        for k, (title, _) in TABLES.items():
            fh.write(f"  {k}: {title}\n")
        fh.write("\nFIGURES (figures/*.png 300 dpi + *.pdf)\n")
        for k, t in FIGS.items():
            fh.write(f"  {k}: {t}\n")
        fh.write("\nREPRODUCIBILITY\n  predictions/folds_seed*.csv : outer-fold assignment of every run\n"
                 "  predictions/oof_<model>_seed*.csv : out-of-fold class probabilities\n"
                 "  config.json, environment.json : settings and software versions\n")
    open(os.path.join(OUT, "log.txt"), "w", encoding="utf-8").write("\n".join(LOG))

    log("\n================ HEADLINE (main comparison) ================")
    print(t6[["model", "macro_F1", "recall_unstable", "MCC", "PCR_violation_%", "inference_ms/sample"]].to_string(index=False))
    log(f"\nSaved everything in: {OUT}\nRuntime: {cfg['runtime_min']} min\nDONE.")


if __name__ == "__main__":
    main()
