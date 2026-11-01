"""
Grid-code-based ordinal stability labelling (IEEE Std 2800-2022).
Reads runs/*.csv and run_log.csv; writes _STEP3/labels_metrics.csv, the stability map (Figure 7)
and the list of runs that fail the pre-event steadiness check.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ========================= SETTINGS =========================
ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
OUT = os.path.join(ROOT, "_STEP3")

V_BAND = (0.90, 1.10)       # pu
F_BAND = (58.8, 61.2)       # Hz
F_FILTER_S = 0.10           # filtered frequency window (s)
V_MASK = 0.10
FINAL_WIN_S = 0.50          # sesh 0.5 s
OSC_V = 0.05                # sustained oscillation: peak-to-peak V (pu)
OSC_F = 0.50                # sustained oscillation: peak-to-peak f (Hz)
TREC_LIMIT_S = 1.0          # recovery er limit (s)
TRIP_V, TRIP_I, TRIP_MIN_S = 0.10, 0.05, 0.05   # trip: V<0.1 pu & I_rms<0.05 pu >= 50 ms
ROCOF_WIN_S = 1.0
# ============================================================

SCEN_NAME = {0: "Steady", 1: "PV drop", 2: "Load rejection", 3: "Islanding",
             4: "Temporary fault", 5: "Permanent fault"}
LEVEL_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
CLASS_CODE = {"stable": 0, "alert": 1, "unstable": 2}
USECOLS = ["Time_s", "Vmag_POI_pu", "f_POI_Hz",
           "Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu",
           "Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"]


def parse_overrides(s):
    """'Scenario.Fault_duration=0.05; Scenario.Fault_distance=0.1; ' -> dict"""
    out = {}
    for k, v in re.findall(r"([\w\.]+)=([^;]+);", str(s)):
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def magnitude_cols(scen, ov):
    """Disturbance er magnitude alada column e."""
    return {
        "pv_factor": ov.get("PV_after_factor", np.nan) if scen == 1 else np.nan,
        "load_P2_MW": ov.get("Load.P2", np.nan) / 1e6 if scen == 2 else np.nan,
        "load_P1_MW": ov.get("Load.P1", np.nan) / 1e6 if scen == 3 else np.nan,
        "fault_dur_s": (ov.get("Scenario.Fault_duration", 0.1) if scen == 4 else np.nan),
        "fault_x": ov.get("Scenario.Fault_distance", np.nan) if scen in (4, 5) else np.nan,
    }


def moving_avg_causal(x, n):
    n = max(1, int(n))
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0.0))
    y = np.empty_like(x, dtype=float)
    y[n - 1:] = (c[n:] - c[:-n]) / n
    y[:n - 1] = c[1:n] / np.arange(1, n)
    return y


def three_phase_rms(df, cols):
    a, b, c = (df[k].values for k in cols)
    return np.sqrt((a**2 + b**2 + c**2) / 3.0)


def longest_true_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def metrics(df, t_e, scen, fault_dur):
    t = df["Time_s"].values
    dt = float(np.median(np.diff(t)))
    V = df["Vmag_POI_pu"].values
    fb = moving_avg_causal(df["f_POI_Hz"].values, F_FILTER_S / dt)
    fb[V < V_MASK] = np.nan
    rocof = np.gradient(fb, dt)

    post = t >= t_e
    final = t >= t[-1] - FINAL_WIN_S
    pre = (t >= t_e - 0.5) & (t < t_e)

    I_bess = three_phase_rms(df, ["Ia_BESS_pu", "Ib_BESS_pu", "Ic_BESS_pu"])
    I_poi = three_phase_rms(df, ["Ia_POI_pu", "Ib_POI_pu", "Ic_POI_pu"])
    dead = (V < TRIP_V) & (np.maximum(I_bess, I_poi) < TRIP_I) & post
    trip = longest_true_run(dead) * dt >= TRIP_MIN_S

    V_out = (V < V_BAND[0]) | (V > V_BAND[1])
    f_out = (fb < F_BAND[0]) | (fb > F_BAND[1])          # NaN -> False
    out_band = V_out | f_out


    t0 = t_e + (fault_dur if scen == 4 and np.isfinite(fault_dur) else 0.0)
    after = t >= t0
    if out_band[final].any() or trip:
        T_rec = np.nan
    else:
        idx = np.where(out_band & after)[0]
        T_rec = 0.0 if len(idx) == 0 else float(t[idx[-1]] + dt - t0)

    fin_f = fb[final]
    A_V = float(np.ptp(V[final]))
    A_f = float(np.nanmax(fin_f) - np.nanmin(fin_f)) if np.isfinite(fin_f).any() else np.nan
    rmask = post & (t <= t_e + ROCOF_WIN_S) & np.isfinite(rocof)

    m = {
        "V_min": float(V[post].min()), "V_max": float(V[post].max()),
        "f_nadir": float(np.nanmin(fb[post])) if np.isfinite(fb[post]).any() else np.nan,
        "f_peak": float(np.nanmax(fb[post])) if np.isfinite(fb[post]).any() else np.nan,
        "ROCOF_max": float(np.nanmax(np.abs(rocof[rmask]))) if rmask.any() else np.nan,
        "T_rec": T_rec, "A_V": A_V, "A_f": A_f,
        "I_BESS_peak": float(np.abs(df.loc[post, ["Ia_BESS_pu", "Ib_BESS_pu",
                                                   "Ic_BESS_pu"]].values).max()),
        "trip": bool(trip),
        "left_band": bool(out_band[post].any()),
        "end_out_band": bool(out_band[final].any()),
        "pre_V_pp": float(np.ptp(V[pre])),
        "pre_f_pp": float(np.nanmax(fb[pre]) - np.nanmin(fb[pre])),
    }

    # ---------- class ----------
    reasons = []
    if trip:
        reasons.append("trip/collapse")
    if m["end_out_band"] and not trip:
        reasons.append("final window outside band")
    if A_V > OSC_V or (np.isfinite(A_f) and A_f > OSC_F):
        reasons.append("sustained oscillation")
    if reasons:
        cls = "unstable"
    elif m["left_band"] or (np.isfinite(T_rec) and T_rec > TREC_LIMIT_S):
        cls = "alert"
        reasons.append("left band" if m["left_band"] else f"T_rec>{TREC_LIMIT_S}s")
    else:
        cls = "stable"
    m["label"] = cls
    m["label_code"] = CLASS_CODE[cls]
    m["label_reason"] = "; ".join(reasons)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    log = pd.read_csv(os.path.join(ROOT, "run_log.csv"))
    log = log[log["status"].astype(str).str.strip() == "ok"]
    log = log.drop_duplicates("run_id", keep="last").reset_index(drop=True)
    print(f"run_log: {len(log)} ta 'ok' run")

    rows, missing = [], []
    for i, r in log.iterrows():
        path = os.path.join(ROOT, "runs", f"{r.run_id}.csv")
        if not os.path.exists(path):
            missing.append(r.run_id)
            continue
        df = pd.read_csv(path, usecols=USECOLS)
        ov = parse_overrides(r.get("overrides", ""))
        mag = magnitude_cols(int(r.scenario), ov)
        m = metrics(df, float(r.t_event), int(r.scenario), mag["fault_dur_s"])
        rows.append({"run_id": r.run_id, "scenario": int(r.scenario),
                     "scenario_name": SCEN_NAME[int(r.scenario)], "ctrl": r.ctrl,
                     "grid": str(r.grid).upper(), "SCR": round(float(r.SCR), 3),
                     **mag, **m})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(log)} porlam ...")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "labels_metrics.csv"), index=False)
    print(f"\nSaved labels_metrics.csv: {len(res)} run")
    if missing:
        print(f"[WARN] {len(missing)} CSV files missing: {missing[:5]} ...")

    # ---------- summary ----------
    by_ctrl_scr = pd.crosstab([res.ctrl, res.grid], res.label).reindex(
        columns=["stable", "alert", "unstable"], fill_value=0)
    by_scen = pd.crosstab([res.scenario_name, res.ctrl], res.label).reindex(
        columns=["stable", "alert", "unstable"], fill_value=0)
    print("\n===== Class count: controller x grid level =====")
    print(by_ctrl_scr.to_string())
    print("\n===== Class count: scenario x controller =====")
    print(by_scen.to_string())
    print("\n===== Total =====")
    print(res.label.value_counts().to_string())

    steady_bad = res[(res.scenario == 0) & (res.label != "stable")]
    print(f"\nCheck: steady-state run je gulo stable NA: {len(steady_bad)}")
    if len(steady_bad):
        print(steady_bad[["run_id", "label", "label_reason", "A_V", "A_f"]].to_string(index=False))
    unsettled = res[(res.pre_V_pp > 0.02) | (res.pre_f_pp > 0.2)]
    print(f"Check: event er age stir na (V pp>0.02 ba f pp>0.2): {len(unsettled)} run")

    with pd.ExcelWriter(os.path.join(OUT, "label_summary.xlsx"), engine="openpyxl") as w:
        by_ctrl_scr.to_excel(w, sheet_name="ctrl_x_grid")
        by_scen.to_excel(w, sheet_name="scenario_x_ctrl")
        res.to_excel(w, sheet_name="all_runs", index=False)

    # ---------- stability map ----------
    res["case"] = res.run_id.str.split("_").str[3:].str.join("_")
    res["case_key"] = res.scenario.astype(str) + "_" + res.case
    cases = sorted(res.case_key.unique(), key=lambda s: (int(s.split("_")[0]), s))
    cmap = ListedColormap(["#2e9d4a", "#f2c037", "#d1453b"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 12), sharey=True)
    for ax, ctrl in zip(axes, ["GFL", "VSM"]):
        sub = res[res.ctrl == ctrl].pivot_table(index="case_key", columns="grid",
                                                values="label_code", aggfunc="first")
        sub = sub.reindex(index=cases, columns=LEVEL_ORDER)
        ax.imshow(sub.values, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        scr = res[res.ctrl == ctrl].groupby("grid").SCR.first().reindex(LEVEL_ORDER)
        ax.set_xticks(range(len(LEVEL_ORDER)))
        ax.set_xticklabels([f"{g}\n{s:.2f}" for g, s in zip(LEVEL_ORDER, scr)], fontsize=8)
        ax.set_title(f"{ctrl} BESS", fontsize=11)
        ax.set_xlabel("Grid level / SCR")
    axes[0].set_yticks(range(len(cases)))
    axes[0].set_yticklabels([c.split("_", 1)[1] for c in cases], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(3)]
    fig.legend(handles, ["stable", "alert", "unstable"], loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "fig_stability_map.png"), dpi=160)
    plt.close(fig)
    print("\nPlot: fig_stability_map.png")


    for cls in ["stable", "alert", "unstable"]:
        pick = res[res.label == cls].sample(min(2, (res.label == cls).sum()), random_state=0)
        for _, r in pick.iterrows():
            df = pd.read_csv(os.path.join(ROOT, "runs", f"{r.run_id}.csv"),
                             usecols=["Time_s", "Vmag_POI_pu", "f_POI_Hz"])
            fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
            ax[0].plot(df.Time_s, df.Vmag_POI_pu, lw=0.7)
            ax[0].axhspan(*V_BAND, color="g", alpha=0.08)
            ax[1].plot(df.Time_s, df.f_POI_Hz, lw=0.6)
            ax[1].axhspan(*F_BAND, color="g", alpha=0.08)
            ax[0].set_ylabel("V (pu)"); ax[1].set_ylabel("f (Hz)"); ax[1].set_xlabel("t (s)")
            ax[0].set_title(f"{r.run_id}  ->  {cls}  ({r.label_reason})", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"check_{cls}_{r.run_id}.png"), dpi=110)
            plt.close(fig)
    print("Check plot: check_*.png (prottek class er 2 ta)")
    print("\nDONE.")


if __name__ == "__main__":
    main()
