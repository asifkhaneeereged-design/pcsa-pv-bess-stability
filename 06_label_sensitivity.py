"""
Label-threshold sensitivity and error rate versus distance to the label threshold (Figure 10).
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


# ================================================================== STEP 8: SENSITIVITY
SEEDS_SENS = [0, 1, 2]
SENS_MODELS = ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost"]
NOMINAL = dict(V_BAND=(0.90, 1.10), F_BAND=(58.8, 61.2), OSC_V=0.05, OSC_F=0.50, TREC=1.0)
CONFIGS = {
    "nominal": {},
    "oscillation strict (0.03 pu / 0.3 Hz)": dict(OSC_V=0.03, OSC_F=0.30),
    "oscillation loose (0.08 pu / 0.8 Hz)": dict(OSC_V=0.08, OSC_F=0.80),
    "recovery limit 0.5 s": dict(TREC=0.5),
    "recovery limit 1.5 s": dict(TREC=1.5),
    "voltage band 0.91-1.05 pu (IEEE 2800 continuous)": dict(V_BAND=(0.91, 1.05)),
    "frequency band 59.5-60.5 Hz": dict(F_BAND=(59.5, 60.5)),
}
LAB_F_FILTER, LAB_V_MASK, LAB_FINAL = 0.10, 0.10, 0.50
LAB_TRIP_V, LAB_TRIP_I, LAB_TRIP_MIN = 0.10, 0.05, 0.05
MARGIN_CAP = 5.0


def _mavg(x, n):
    n = max(1, int(n))
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    y = np.empty_like(x, dtype=float)
    y[n - 1:] = (c[n:] - c[:-n]) / n
    y[:n - 1] = c[1:n] / np.arange(1, n)
    return y


def _longest(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def prep_run(df, t_e, scen, fault_dur):
    """Band-niropekkho jinish ekbar hisab (step 3 er hubohu)."""
    t = df["Time_s"].to_numpy(); dt = float(np.median(np.diff(t)))
    V = df["Vmag_POI_pu"].to_numpy()
    fb = _mavg(df["f_POI_Hz"].to_numpy(), LAB_F_FILTER / dt); fb[V < LAB_V_MASK] = np.nan
    post, final = t >= t_e, t >= t[-1] - LAB_FINAL
    rms = lambda cols: np.sqrt(sum(df[c].to_numpy() ** 2 for c in cols) / 3.0)
    ib = rms(["Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu"]); ip = rms(["Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"])
    dead = (V < LAB_TRIP_V) & (np.maximum(ib, ip) < LAB_TRIP_I) & post
    trip = _longest(dead) * dt >= LAB_TRIP_MIN
    t0 = t_e + (fault_dur if scen == 4 and np.isfinite(fault_dur) else 0.0)
    ff = fb[final]
    A_V = float(np.ptp(V[final]))
    A_f = float(np.nanmax(ff) - np.nanmin(ff)) if np.isfinite(ff).any() else np.nan
    return dict(t=t, dt=dt, V=V, fb=fb, post=post, final=final, trip=trip, after=t >= t0, t0=t0, A_V=A_V, A_f=A_f)


def label_with(r, cfg):
    c = {**NOMINAL, **cfg}
    out = (r["V"] < c["V_BAND"][0]) | (r["V"] > c["V_BAND"][1]) | (r["fb"] < c["F_BAND"][0]) | (r["fb"] > c["F_BAND"][1])
    end_out = out[r["final"]].any()
    if end_out or r["trip"]:
        T_rec = np.nan
    else:
        idx = np.where(out & r["after"])[0]
        T_rec = 0.0 if len(idx) == 0 else float(r["t"][idx[-1]] + r["dt"] - r["t0"])
    osc = r["A_V"] > c["OSC_V"] or (np.isfinite(r["A_f"]) and r["A_f"] > c["OSC_F"])
    if r["trip"] or end_out or osc:
        return 2
    if out[r["post"]].any() or (np.isfinite(T_rec) and T_rec > c["TREC"]):
        return 1
    return 0


def margin(row):
    """Normalised distance of the label-determining metrics to their thresholds."""
    if bool(row.trip):
        return MARGIN_CAP
    m = []
    m.append(abs(row.A_V - NOMINAL["OSC_V"]) / NOMINAL["OSC_V"])
    if np.isfinite(row.A_f):
        m.append(abs(row.A_f - NOMINAL["OSC_F"]) / NOMINAL["OSC_F"])
    if np.isfinite(row.T_rec):
        m.append(abs(row.T_rec - NOMINAL["TREC"]) / NOMINAL["TREC"])
    vb, fbd = NOMINAL["V_BAND"], NOMINAL["F_BAND"]
    m += [abs(row.V_min - vb[0]) / (vb[1] - vb[0]), abs(row.V_max - vb[1]) / (vb[1] - vb[0])]
    for v, edge in [(row.f_nadir, fbd[0]), (row.f_peak, fbd[1])]:
        if np.isfinite(v):
            m.append(abs(v - edge) / (fbd[1] - fbd[0]))
    return float(min(min(m), MARGIN_CAP))


def main():
    global TABLES, FIGS
    TABLES, FIGS = {}, {}
    for d in (OUT, T_DIR, F_DIR, P_DIR):
        os.makedirs(d, exist_ok=True)
    t_start = time.time()
    log("=" * 64 + "\n STEP 8: SENSITIVITY (label thresholds, distance to label boundary)\n" + "=" * 64)

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
    lab = pd.read_csv(os.path.join(ROOT, "_STEP3", "labels_metrics.csv")).set_index("run_id")
    run_ids = D.run_id.unique()
    log_ = pd.read_csv(os.path.join(ROOT, "run_log.csv"))
    log_ = log_[log_.status.astype(str).str.strip() == "ok"].drop_duplicates("run_id", keep="last")
    t_event = dict(zip(log_.run_id, log_.t_event))
    n_par = PhysOrdNet(len(P_cols), len(B_cols)).n_params()
    BM = baseline_models(n_par)

    # ================================================================ 1. relabel
    log(f"\n[1] Relabelling {len(run_ids)} runs under {len(CONFIGS)} threshold settings (raw CSV, step 3 logic)")
    usecols = ["Time_s", "Vmag_POI_pu", "f_POI_Hz", "Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu",
               "Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"]
    L = {k: {} for k in CONFIGS}
    for i, rid in enumerate(run_ids):
        df = pd.read_csv(os.path.join(ROOT, "runs", f"{rid}.csv"), usecols=usecols)
        r = prep_run(df, float(t_event.get(rid, 4.0)), int(lab.at[rid, "scenario"]), lab.at[rid, "fault_dur_s"])
        for k, cfg in CONFIGS.items():
            L[k][rid] = label_with(r, cfg)
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(run_ids)} run")
    L = pd.DataFrame(L)
    agree_nom = float((L["nominal"] == lab.loc[L.index, "label_code"]).mean())
    log(f"  check: nominal relabel vs step 3 labels agreement = {100 * agree_nom:.1f}%")
    lab_rows = []
    for k in CONFIGS:
        cnt = L[k].value_counts().reindex([0, 1, 2]).fillna(0).astype(int)
        lab_rows.append({"setting": k, "stable": cnt[0], "alert": cnt[1], "unstable": cnt[2],
                         "runs_relabelled_vs_nominal_%": round(100 * float((L[k] != L["nominal"]).mean()), 2)})

    # ================================================================ 2. CV per setting
    log(f"\n[2] Repeated CV per setting (seeds {SEEDS_SENS}, {N_FOLD} folds) for {SENS_MODELS}")
    Dn = {nz: D[(D.noise == nz) & (D.rep == rp)].set_index("run_id") for nz, rp in [(TRAIN_NOISE, 0)]}
    Dt = D[(D.noise == TEST_NOISE) & (D.rep == 1)].set_index("run_id")

    def fit(name, dtr, ytr, seed):
        if name == "MonoOrd-GBM":
            m = MonoOrdGBM(cons, seed).fit(dtr[F_ALL].to_numpy(), ytr)
            return lambda d: m.predict_proba(d[F_ALL].to_numpy())
        if name == "PhysOrd-Net":
            m, pr, _ = fit_physord(dtr, ytr, P_cols, B_cols, dirs, seed)
            return lambda d: m.proba(*pr(d))
        bn = "XGBoost" if "XGBoost" in BM else "HGB"
        est = fit_baseline(bn, *BM[bn], dtr[F_ALL].to_numpy(), ytr, seed)
        return lambda d: est.predict_proba(d[F_ALL].to_numpy())

    rows = []
    ids = np.array(L.index)
    for k in CONFIGS:
        y_all = L.loc[ids, k].to_numpy().astype(int)
        for seed in SEEDS_SENS:
            skf = StratifiedKFold(N_FOLD, shuffle=True, random_state=seed)
            for name in SENS_MODELS:
                ys, ps, v, n = [], [], 0, 0
                for a, b in skf.split(ids, y_all):
                    dtr = Dn[TRAIN_NOISE].loc[ids[a]].reset_index(); ytr = y_all[a]
                    dte = Dt.loc[ids[b]].reset_index(); yte = y_all[b]
                    f = fit(name, dtr, ytr, seed)
                    P = f(dte); ys.append(yte); ps.append(P.argmax(1))
                    sd = dtr[P_cols].std().to_numpy() + 1e-9
                    for j, c in enumerate(P_cols):
                        d2 = dte.copy(); d2[c] = d2[c] + dirs[j] * PCR_DELTA * sd[j]
                        v += int((risk(f(d2)) < risk(P) - 1e-9).sum()); n += len(d2)
                y, p = np.concatenate(ys), np.concatenate(ps)
                rows.append({"setting": k, "seed": seed, "model": name, "macro_F1": f1_score(y, p, average="macro"),
                             "recall_unstable": recall_score(y == 2, p == 2), "PCR_violation_%": 100 * v / n})
        log(f"  {k} done")
    S = pd.DataFrame(rows).groupby(["setting", "model"])[["macro_F1", "recall_unstable", "PCR_violation_%"]].agg(["mean", "std"])
    S.columns = ["_".join(c) for c in S.columns]
    S = S.reset_index()
    order = list(CONFIGS)
    S["o"] = S.setting.map(order.index); S = S.sort_values(["o", "model"]).drop(columns="o")
    save_table(pd.DataFrame(lab_rows), "Table27_label_sensitivity_classes",
               f"Class counts under alternative label thresholds (nominal relabel agrees with step 3 in {100 * agree_nom:.1f}% of runs)")
    save_table(S, "Table28_label_sensitivity_performance",
               f"Macro F1, unstable recall and physics-violation rate under alternative label thresholds ({len(SEEDS_SENS)} seeds x {N_FOLD} folds)")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    piv = S.pivot(index="setting", columns="model", values="macro_F1_mean").loc[order, SENS_MODELS]
    err = S.pivot(index="setting", columns="model", values="macro_F1_std").loc[order, SENS_MODELS]
    piv.plot.barh(ax=ax[0], xerr=err, capsize=2); ax[0].set_xlabel("Macro F1"); ax[0].set_ylabel("")
    ax[0].set_xlim(max(0, piv.values.min() - 0.08), 1); ax[0].invert_yaxis(); ax[0].legend(fontsize=7)
    pv = S.pivot(index="setting", columns="model", values="PCR_violation_%_mean").loc[order, SENS_MODELS]
    pv.plot.barh(ax=ax[1], legend=False); ax[1].set_xlabel("Physics violations (%)"); ax[1].set_yticklabels([])
    ax[1].set_ylabel(""); ax[1].invert_yaxis()
    save_fig(fig, "Fig16_label_sensitivity", "Conclusions under alternative label thresholds")

    # ================================================================ 3. error vs distance to label boundary
    log("\n[3] Error vs distance to the label boundary (stored out-of-fold predictions, all seeds)")
    lb = lab.loc[ids].copy()
    lb["margin"] = lb.apply(margin, axis=1)
    err_cols = {}
    for name in ["MonoOrd-GBM", "PhysOrd-Net", "XGBoost", "RF"]:
        e = []
        for seed in SEEDS:
            pth = os.path.join(P_DIR, f"oof_{name}_seed{seed}.csv")
            if os.path.exists(pth):
                o = pd.read_csv(pth).set_index("run_id")
                e.append((o.y_true != o.y_pred).astype(float).reindex(ids))
        if e:
            err_cols[name] = pd.concat(e, axis=1).mean(axis=1)
    E = pd.DataFrame(err_cols); E["margin"] = lb["margin"]
    E["decile"] = pd.qcut(E.margin.rank(method="first"), 10, labels=[f"D{i}" for i in range(1, 11)])
    dt_ = E.groupby("decile", observed=True).agg(
        margin_min=("margin", "min"), margin_max=("margin", "max"), n=("margin", "size"),
        **{f"err_%_{m}": (m, lambda s: 100 * s.mean()) for m in err_cols}).reset_index()
    from scipy.stats import spearmanr, pearsonr
    cor = []
    for m in err_cols:
        q10 = E.margin <= E.margin.quantile(0.10); top = E.margin > E.margin.median()
        cor.append({"model": m, "spearman(margin, error)": spearmanr(E.margin, E[m]).correlation,
                    "pearson(margin, error)": pearsonr(E.margin, E[m])[0],
                    "error_%_lowest_10%_margin": 100 * E.loc[q10, m].mean(),
                    "error_%_upper_50%_margin": 100 * E.loc[top, m].mean(), "error_%_overall": 100 * E[m].mean()})
    save_table(dt_, "Table29_error_vs_label_margin",
               "Error rate (%) by decile of normalized distance to the nearest label threshold (D1 = closest)")
    save_table(pd.DataFrame(cor), "Table30_error_margin_correlation",
               "Correlation between distance to label threshold and error; error near vs far from the boundary")
    E[["margin"] + list(err_cols)].to_csv(os.path.join(P_DIR, "error_vs_margin_per_run.csv"))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = np.arange(len(dt_)); w = 0.8 / len(err_cols)
    for i, m in enumerate(err_cols):
        ax.bar(x + i * w - 0.4 + w / 2, dt_[f"err_%_{m}"], w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(dt_.decile); ax.set_xlabel("Decile of distance to label threshold (D1 = closest)")
    ax.set_ylabel("Error rate (%)"); ax.legend(fontsize=7)
    save_fig(fig, "Fig17_error_vs_label_margin", "Errors concentrate near the label thresholds")

    with open(os.path.join(OUT, "README.txt"), "a", encoding="utf-8") as fh:
        fh.write("\n\nSENSITIVITY (step8_sensitivity.py)\n")
        for k, (title, _) in TABLES.items():
            fh.write(f"  {k}: {title}\n")
        for k, t_ in FIGS.items():
            fh.write(f"  {k}: {t_}\n")
    open(os.path.join(OUT, "log_step8.txt"), "w", encoding="utf-8").write("\n".join(LOG))
    log("\n================ SUMMARY ================")
    print(pd.DataFrame(lab_rows).to_string(index=False))
    print()
    print(S[["setting", "model", "macro_F1_mean", "macro_F1_std", "PCR_violation_%_mean"]].round(4).to_string(index=False))
    print()
    print(dt_.round(2).to_string(index=False))
    print()
    print(pd.DataFrame(cor).round(3).to_string(index=False))
    log(f"\nRuntime: {(time.time() - t_start) / 60:.1f} min\nDONE.")


if __name__ == "__main__":
    main()
