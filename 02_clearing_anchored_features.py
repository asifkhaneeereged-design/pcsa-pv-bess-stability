"""
Base features from clearing-anchored observation windows with PMU-grade measurement noise.
Writes _STEP4/features.csv for all window lengths, noise levels and realizations.
"""
import os
import json
import time
import zlib
import numpy as np
import pandas as pd

# ========================= SETTINGS =========================
ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
OUT = os.path.join(ROOT, "_STEP4")
LABELS = os.path.join(ROOT, "_STEP3", "labels_metrics.csv")

TAUS = [0.02, 0.05, 0.10, 0.20, 0.50]   # s, reference time er por observation window


#             permanent fault e t_e + PF_DECISION (sobcheye lomba simulated clearing time);

REF_MODES = ["event", "clear"]
PF_DECISION = 0.50                      # s
PRE_WIN = 0.10
END_WIN = 0.01
V_MASK = 0.10

PRE_V_PP, PRE_F_PP = 0.02, 0.2

NOISE_LEVELS = [0, 1, 2, 5]             # 0 = clean; base sigma er gun
N_REP = 2                               # noise level >0 e realization: 0 = train, 1 = test
SIG_V = 0.005                           # pu  (IEEE C37.118.1 1% TVE er ordhek)
SIG_F = 0.005                           # Hz  (C37.118.1 steady-state frequency error)
SIG_I_REL = 0.01                        # current: pre-event RMS er 1%
SIG_PQ_REL = 0.01                       # P, Q: pre-event |mean| er 1% (min 1 MW/MVAr)
# ============================================================

PQ = ["P_total_MW", "Q_total_MVAr", "P_BESS_MW", "Q_BESS_MVAr", "P_PV_MW", "Q_PV_MVAr"]
IB = ["Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu"]
IP = ["Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"]
VP = ["Va_POI_pu", "Vb_POI_pu", "Vc_POI_pu"]
META = ["scenario", "scenario_name", "ctrl", "grid", "SCR", "pv_factor", "load_P2_MW",
        "load_P1_MW", "fault_dur_s", "fault_x"]
TARGETS = ["label", "label_code", "V_min", "V_max", "f_nadir", "f_peak", "ROCOF_max",
           "T_rec", "A_V", "A_f", "I_BESS_peak", "trip"]

FEATURE_DOC = {
    "dV_mean": "mean V_POI in window minus pre-event mean (pu)",
    "dV_min": "min V_POI in window minus pre-event mean (pu)",
    "dV_max": "max V_POI in window minus pre-event mean (pu)",
    "dV_end": "mean V_POI over last 10 ms of window minus pre-event mean (pu)",
    "V_std": "std of V_POI in window (pu)",
    "V_slope": "linear-fit slope of V_POI in window (pu/s)",
    "V_collapse_frac": "fraction of window with V_POI < 0.1 pu",
    "df_absmax": "max |f - f_pre| in window, V >= 0.1 pu only (Hz)",
    "df_end": "mean f over last 10 ms minus f_pre (Hz)",
    "f_std": "std of f in window (Hz)",
    "ROCOF_fit": "linear-fit slope of f in window (Hz/s)",
    "VI_unbalance": "(max - min)/mean of phase-voltage RMS in window",
    "I_BESS_rms_max": "max BESS 3-phase RMS current in window (pu)",
    "dI_BESS_rms": "mean BESS RMS current in window minus pre-event (pu)",
    "I_BESS_abs_peak": "max |phase current| of BESS in window (pu)",
    "I_POI_rms_max": "max POI 3-phase RMS current in window (pu)",
    "dV_min_cum": "min V_POI from event to window end minus pre-event mean (pu)",
    "df_absmax_cum": "max |f - f_pre| from event to window end, V >= 0.1 pu (Hz)",
    "I_BESS_abs_peak_cum": "max |BESS phase current| from event to window end (pu)",
    "dI_POI_rms": "mean POI RMS current in window minus pre-event (pu)",
}
for c in PQ:
    s = c.split("_")[0] + "_" + c.split("_")[1]
    FEATURE_DOC[f"d{s}_mean"] = f"mean {c} in window minus pre-event mean"
    FEATURE_DOC[f"d{s}_end"] = f"mean {c} over last 10 ms minus pre-event mean"
for c in ["P_total_MW", "P_BESS_MW"]:
    s = c.split("_")[0] + "_" + c.split("_")[1]
    FEATURE_DOC[f"{s}_std"] = f"std of {c} in window"
FEATURES = list(FEATURE_DOC.keys())


def rms3(a, b, c):
    return np.sqrt((a**2 + b**2 + c**2) / 3.0)


def slope(t, x):
    if len(x) < 3 or not np.isfinite(x).all():
        return 0.0
    tc = t - t.mean()
    d = (tc**2).sum()
    return float((tc * (x - x.mean())).sum() / d) if d > 0 else 0.0


def add_noise(sig, pre, level, rng):
    """Raw signal e PMU-sadrisho Gaussian noise."""
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


def features(sig, t_e, t_ref, tau):
    t = sig["Time_s"]
    pre = (t >= t_e - PRE_WIN) & (t < t_e)
    win = (t >= t_ref) & (t <= t_ref + tau)
    end = (t > t_ref + tau - END_WIN) & win
    cum = (t >= t_e) & (t <= t_ref + tau)
    tw = t[win]
    F = {}

    V = sig["Vmag_POI_pu"]
    v0 = V[pre].mean()
    Vw = V[win]
    F["dV_mean"] = Vw.mean() - v0
    F["dV_min"] = Vw.min() - v0
    F["dV_max"] = Vw.max() - v0
    F["dV_end"] = V[end].mean() - v0
    F["V_std"] = Vw.std()
    F["V_slope"] = slope(tw, Vw)
    F["V_collapse_frac"] = float((Vw < V_MASK).mean())

    f = sig["f_POI_Hz"]
    f0 = f[pre].mean()
    ok = Vw >= V_MASK
    fw = f[win][ok]
    if ok.sum() >= 5:
        F["df_absmax"] = np.abs(fw - f0).max()
        fe = f[end][V[end] >= V_MASK]
        F["df_end"] = (fe.mean() - f0) if len(fe) else 0.0
        F["f_std"] = fw.std()
        F["ROCOF_fit"] = slope(tw[ok], fw)
    else:
        F["df_absmax"] = F["df_end"] = F["f_std"] = F["ROCOF_fit"] = 0.0

    ph = [np.sqrt(np.mean(sig[k][win] ** 2)) for k in VP]
    F["VI_unbalance"] = (max(ph) - min(ph)) / max(np.mean(ph), 1e-6)

    ib = rms3(*(sig[k] for k in IB))
    F["I_BESS_rms_max"] = ib[win].max()
    F["dI_BESS_rms"] = ib[win].mean() - ib[pre].mean()
    F["I_BESS_abs_peak"] = max(np.abs(sig[k][win]).max() for k in IB)
    ip = rms3(*(sig[k] for k in IP))
    F["I_POI_rms_max"] = ip[win].max()
    F["dV_min_cum"] = V[cum].min() - v0
    okc = V[cum] >= V_MASK
    F["df_absmax_cum"] = np.abs(f[cum][okc] - f0).max() if okc.sum() >= 5 else 0.0
    F["I_BESS_abs_peak_cum"] = max(np.abs(sig[k][cum]).max() for k in IB)
    F["dI_POI_rms"] = ip[win].mean() - ip[pre].mean()

    for c in PQ:
        s = c.split("_")[0] + "_" + c.split("_")[1]
        x0 = sig[c][pre].mean()
        F[f"d{s}_mean"] = sig[c][win].mean() - x0
        F[f"d{s}_end"] = sig[c][end].mean() - x0
    for c in ["P_total_MW", "P_BESS_MW"]:
        s = c.split("_")[0] + "_" + c.split("_")[1]
        F[f"{s}_std"] = sig[c][win].std()
    return {k: float(F[k]) for k in FEATURES}


def ref_time(t_e, scen, fault_dur, mode):
    if mode == "clear":
        if scen == 4 and np.isfinite(fault_dur):
            return t_e + float(fault_dur)
        if scen == 5:
            return t_e + PF_DECISION
    return t_e


def main():
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(LABELS)
    bad = (lab.pre_V_pp > PRE_V_PP) | (lab.pre_f_pp > PRE_F_PP)
    lab[bad][["run_id", "label", "pre_V_pp", "pre_f_pp"]].to_csv(
        os.path.join(OUT, "excluded_runs.csv"), index=False)
    lab = lab[~bad].reset_index(drop=True)
    log = pd.read_csv(os.path.join(ROOT, "run_log.csv"))
    log = log[log.status.astype(str).str.strip() == "ok"].drop_duplicates("run_id", keep="last")
    t_event = dict(zip(log.run_id, log.t_event))
    print(f"Bad: {bad.sum()} run (event er age stir na). Baki: {len(lab)} run")

    rows, t_feat = [], []
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
                        F = features(s_n, t_e, t_ref, tau)
                        t_feat.append(time.perf_counter() - t0)
                        rows.append({"run_id": r.run_id, "ref": mode,
                                     "tau_ms": int(round(tau * 1000)),
                                     "noise": lvl, "rep": rep, **F,
                                     **{k: r[k] for k in META + TARGETS if k in r}})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(lab)} run ...")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "features.csv"), index=False)
    with open(os.path.join(OUT, "feature_list.txt"), "w", encoding="utf-8") as fh:
        for k, v in FEATURE_DOC.items():
            fh.write(f"{k}\t{v}\n")
    tf = np.array(t_feat) * 1e3
    timing = {"feature_extraction_ms_mean": float(tf.mean()),
              "feature_extraction_ms_p95": float(np.percentile(tf, 95)),
              "n_samples_timed": int(len(tf))}
    with open(os.path.join(OUT, "timing.json"), "w") as fh:
        json.dump(timing, fh, indent=2)

    print(f"\nSaved features.csv: {len(out)} row x {out.shape[1]} col "
          f"({out.run_id.nunique()} run x {len(REF_MODES)} ref x {len(TAUS)} tau x noise/rep)")
    print(f"Feature: {len(FEATURES)} ta")
    print(f"Feature extraction: {timing['feature_extraction_ms_mean']:.2f} ms/sample (mean)")
    print("\nClass count (unique run):")
    print(out.drop_duplicates("run_id").label.value_counts().to_string())
    print("\nDONE.")


if __name__ == "__main__":
    main()
