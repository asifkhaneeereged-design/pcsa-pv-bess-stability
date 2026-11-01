"""
Fourteen physical stability indicators with a priori risk directions
(oscillation, ride-through and recovery branches). Writes _STEP4/physics_features.csv.
"""
import os
import json
import time
import zlib
import numpy as np
import pandas as pd


ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
OUT = os.path.join(ROOT, "_STEP4")
LABELS = os.path.join(ROOT, "_STEP3", "labels_metrics.csv")
TAUS = [0.02, 0.05, 0.10, 0.20, 0.50]
REF_MODES = ["event", "clear"]
PF_DECISION = 0.50
PRE_WIN, END_WIN, V_MASK = 0.10, 0.01, 0.10
PRE_V_PP, PRE_F_PP = 0.02, 0.2
NOISE_LEVELS, N_REP = [0, 1, 2, 5], 2
SIG_V, SIG_F, SIG_I_REL, SIG_PQ_REL = 0.005, 0.005, 0.01, 0.01

# Model er IEEE 2800 table (BatteryStoragePVPlantGFMParameters.m)
LV_LEVELS = [(0.10, 0.32), (0.25, 0.32), (0.50, 1.2), (0.70, 3.0), (0.90, 6.0)]  # V <, trip s
HV_LEVELS = [(1.10, 1.0), (1.20, 0.015), (1.40, 0.003)]                          # V >, trip s
F_LEVELS = [("lt", 48.0, 0.1), ("lt", 57.0, 299.0), ("gt", 61.26, 299.0), ("gt", 61.86, 0.1)]
I_LIMIT = 1.2            # pu, BESS current limit (maxCurrentFactor)
OSC_FBAND = (2.0, 40.0)  # Hz, dominant mode khojar band
# ==================================================================================

IB = ["Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu"]
IP = ["Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"]
VP = ["Va_POI_pu", "Vb_POI_pu", "Vc_POI_pu"]

INDICATORS = [  # name, branch, direction, description
    ("zeta_V",     "osc",  -1, "damping ratio of dominant V_POI mode in window (matrix pencil)"),
    ("zeta_P",     "osc",  -1, "damping ratio of dominant P_BESS mode in window"),
    ("growth_V",   "osc",  +1, "log RMS ratio, 2nd vs 1st half of detrended V_POI in window"),
    ("growth_P",   "osc",  +1, "log RMS ratio, 2nd vs 1st half of detrended P_BESS in window"),
    ("U_LV",       "ride", +1, "max over LVRT levels of (time below level)/(allowed time)"),
    ("U_HV",       "ride", +1, "max over HVRT levels of (time above level)/(allowed time)"),
    ("U_F",        "ride", +1, "max over FRT levels of (time outside level)/(allowed time)"),
    ("I_ratio",    "ride", +1, "max BESS RMS current / 1.2 pu limit, event to decision"),
    ("I_over_s",   "ride", +1, "time BESS RMS current above 1.2 pu, event to decision (s)"),
    ("sag_depth",  "rec",  +1, "pre-event V minus min V, event to decision (pu)"),
    ("rec_index",  "rec",  -1, "(V_end - V_min)/(V_pre - V_min): fraction of sag recovered"),
    ("V_resid",    "rec",  +1, "|V_end - V_pre| (pu)"),
    ("f_resid",    "rec",  +1, "|f_end - f_pre|, 50 ms filtered (Hz)"),
    ("P_resid",    "rec",  +1, "|P_total_end - P_pre| / max(|P_pre|, 1)"),
]
NAMES = [n for n, *_ in INDICATORS]
EXTRA = ["osc_valid"]


def rms3(a, b, c):
    return np.sqrt((a**2 + b**2 + c**2) / 3.0)


def add_noise(sig, pre, level, rng):          # step 4 er hubohu
    if level == 0:
        return sig
    out = {}
    for k, x in sig.items():
        if k == "Time_s":
            out[k] = x
            continue
        if k == "Vmag_POI_pu" or k in VP:
            s = SIG_V
        elif k == "f_POI_Hz":
            s = SIG_F
        elif k in IB or k in IP:
            grp = IB if k in IB else IP
            s = SIG_I_REL * max(float(np.mean(rms3(*(sig[g][pre] for g in grp)))), 1e-3)
        else:
            s = SIG_PQ_REL * max(abs(float(np.mean(x[pre]))), 1.0)
        out[k] = x + rng.normal(0.0, level * s, size=x.shape)
    return out


def ref_time(t_e, scen, fault_dur, mode):     # step 4 er hubohu
    if mode == "clear":
        if scen == 4 and np.isfinite(fault_dur):
            return t_e + float(fault_dur)
        if scen == 5:
            return t_e + PF_DECISION
    return t_e


def longest_run(mask, dt):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * dt


def detrend(x, t):
    if len(x) < 3:
        return x - x.mean()
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def matrix_pencil_zeta(x, dt, order=4):
    """Damping ratio of the dominant oscillatory mode; None if no valid mode."""
    step = max(1, int(round(1e-3 / dt)))           # ~1 kHz e niye ashi
    y, d = x[::step], dt * step
    N = len(y)
    if N < 30 or np.std(y) < 1e-6:
        return None
    L = N // 2
    Y = np.array([y[i:i + L + 1] for i in range(N - L)])
    try:
        _, _, Vh = np.linalg.svd(Y, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    M = min(order, Vh.shape[0])
    V = Vh[:M].conj().T
    try:
        z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    except np.linalg.LinAlgError:
        return None
    z = z[np.abs(z) > 1e-8]
    s = np.log(z.astype(complex)) / d
    f = np.abs(s.imag) / (2 * np.pi)
    keep = (f >= OSC_FBAND[0]) & (f <= OSC_FBAND[1])
    if not keep.any():
        return None
    s = s[keep]
    n = np.arange(N)
    Z = np.exp(np.outer(n * d, s))
    amp, *_ = np.linalg.lstsq(Z, y.astype(complex), rcond=None)
    k = int(np.argmax(np.abs(amp)))
    sig, om = s[k].real, abs(s[k].imag)
    if N * d < 1.5 / max(om / (2 * np.pi), 1e-6):
        return None
    return float(np.clip(-sig / np.sqrt(sig**2 + om**2), -1.0, 1.0))


def growth(x, t):
    if len(x) < 20:
        return 0.0
    r = detrend(x, t)
    h = len(r) // 2
    a, b = np.sqrt(np.mean(r[:h] ** 2)), np.sqrt(np.mean(r[h:] ** 2))
    return float(np.log((b + 1e-6) / (a + 1e-6)))


def movavg(x, n):
    n = max(1, int(n))
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = np.empty_like(x, dtype=float)
    y[n - 1:] = (c[n:] - c[:-n]) / n
    y[:n - 1] = c[1:n] / np.arange(1, n)
    return y


def indicators(sig, t_e, t_ref, tau):
    t = sig["Time_s"]
    dt = float(np.median(np.diff(t)))
    t_dec = t_ref + tau
    pre = (t >= t_e - PRE_WIN) & (t < t_e)
    win = (t >= t_ref) & (t <= t_dec)
    cum = (t >= t_e) & (t <= t_dec)
    end = (t > t_dec - END_WIN) & win
    V, P, PB = sig["Vmag_POI_pu"], sig["P_total_MW"], sig["P_BESS_MW"]
    fF = movavg(sig["f_POI_Hz"], 0.05 / dt)
    v0, f0, p0 = V[pre].mean(), fF[pre].mean(), P[pre].mean()
    F = {}

    # [A] oscillation
    tw = t[win]
    zv = matrix_pencil_zeta(detrend(V[win], tw), dt)
    zp = matrix_pencil_zeta(detrend(PB[win], tw), dt)
    F["osc_valid"] = float(zv is not None or zp is not None)
    F["zeta_V"] = zv if zv is not None else 0.3        # invalid -> bhalo damped (neutral)
    F["zeta_P"] = zp if zp is not None else 0.3
    F["growth_V"] = growth(V[win], tw) if tau >= 0.04 else 0.0
    F["growth_P"] = growth(PB[win], tw) if tau >= 0.04 else 0.0


    Vc = V[cum]
    F["U_LV"] = max(longest_run(Vc < lv, dt) / ts for lv, ts in LV_LEVELS)
    F["U_HV"] = max(longest_run(Vc > hv, dt) / ts for hv, ts in HV_LEVELS)
    fc = np.where(Vc >= V_MASK, fF[cum], f0)
    uf = []
    for op, lv, ts in F_LEVELS:
        m = fc < lv if op == "lt" else fc > lv
        uf.append(longest_run(m, dt) / ts)
    F["U_F"] = max(uf)
    ib = rms3(*(sig[k] for k in IB))[cum]
    F["I_ratio"] = float(ib.max() / I_LIMIT)
    F["I_over_s"] = float((ib > I_LIMIT).sum() * dt)

    # [C] recovery
    vmin = Vc.min()
    v_end = V[end].mean()
    F["sag_depth"] = float(v0 - vmin)
    F["rec_index"] = float(np.clip((v_end - vmin) / max(v0 - vmin, 0.02), -1.0, 2.0))
    F["V_resid"] = float(abs(v_end - v0))
    fe = fF[end][V[end] >= V_MASK]
    F["f_resid"] = float(abs(fe.mean() - f0)) if len(fe) else 0.0
    F["P_resid"] = float(abs(P[end].mean() - p0) / max(abs(p0), 1.0))
    return {k: float(F[k]) for k in NAMES + EXTRA}


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    lab = lab[~((lab.pre_V_pp > PRE_V_PP) | (lab.pre_f_pp > PRE_F_PP))].reset_index(drop=True)
    log = pd.read_csv(os.path.join(ROOT, "run_log.csv"))
    log = log[log.status.astype(str).str.strip() == "ok"].drop_duplicates("run_id", keep="last")
    t_event = dict(zip(log.run_id, log.t_event))
    print(f"{len(lab)} run (step 4 er moto 10 ta bad)")

    rows, tt = [], []
    for i, r in lab.iterrows():
        df = pd.read_csv(os.path.join(ROOT, "runs", f"{r.run_id}.csv"))
        sig = {c: df[c].to_numpy(dtype=float) for c in df.columns if c != "run_id"}
        t_e = float(t_event.get(r.run_id, 4.0))
        pre = (sig["Time_s"] >= t_e - PRE_WIN) & (sig["Time_s"] < t_e)
        for lvl in NOISE_LEVELS:
            for rep in ([0] if lvl == 0 else range(N_REP)):
                seed = zlib.crc32(f"{r.run_id}|{lvl}|{rep}".encode())
                s_n = add_noise(sig, pre, lvl, np.random.default_rng(seed))
                for mode in REF_MODES:
                    t_ref = ref_time(t_e, int(r.scenario), r.fault_dur_s, mode)
                    for tau in TAUS:
                        t0 = time.perf_counter()
                        F = indicators(s_n, t_e, t_ref, tau)
                        tt.append(time.perf_counter() - t0)
                        rows.append({"run_id": r.run_id, "ref": mode,
                                     "tau_ms": int(round(tau * 1000)), "noise": lvl,
                                     "rep": rep, **F})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(lab)} run ...")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "physics_features.csv"), index=False)
    with open(os.path.join(OUT, "physics_indicators.txt"), "w", encoding="utf-8") as fh:
        for n, b, d, desc in INDICATORS:
            fh.write(f"{n}\t{b}\t{d}\t{desc}\n")
    tm = np.array(tt) * 1e3
    json.dump({"physics_ms_mean": float(tm.mean()), "physics_ms_p95": float(np.percentile(tm, 95))},
              open(os.path.join(OUT, "timing_physics.json"), "w"), indent=2)
    print(f"\nSaved physics_features.csv: {len(out)} row, {len(NAMES)} indicator")
    print(f"Physics indicator time: {tm.mean():.2f} ms/sample (mean)")
    c = out[(out.ref == "clear") & (out.noise == 0) & (out.tau_ms == 100)]
    print("\nClean, 100 ms, 'clear' window er summary:")
    print(c[NAMES + EXTRA].describe().T[["mean", "min", "max"]].round(3).to_string())
    print("\nDONE.")


if __name__ == "__main__":
    main()
