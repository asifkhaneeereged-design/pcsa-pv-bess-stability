"""
Per-scenario EMT statistics reported in the EMT results subsection.
"""
import os, re, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================== SETTINGS
ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
REF_GRID = "G4"            # reference grid level for the overview figures (SCR 0.89)
SWEEP_CASE_SCEN = 2        # disturbance shown across all SCR levels (2 = load rejection)
SWEEP_CASE_PATTERN = "40"  # magnitude in the case name (40 MW load rejection)
FAULT_X_PATTERN = "x50"    # fault location shown in the overview and ride-through figures
T_WIN = (3.9, 6.0)         # time window plotted (s); event at 4 s
F_FILTER_S = 0.10          # same 100 ms filter as the labels
DPI = 300
# ======================================================================

OUT = os.path.join(ROOT, "_RESULTS_JOURNAL", "scenario_results")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3})
SCEN = {1: "PV irradiance drop", 2: "Load rejection", 3: "Unintentional islanding",
        4: "Temporary three-phase fault", 5: "Permanent three-phase fault"}
LEVELS = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
COLS = ["Time_s", "Vmag_POI_pu", "f_POI_Hz", "P_BESS_MW"]

lab = pd.read_csv(os.path.join(ROOT, "_STEP3", "labels_metrics.csv"))
lab["case"] = lab.run_id.str.split("_", n=3).str[3]          # e.g. lr40, pv30, tf_d150_x50
excl_path = os.path.join(ROOT, "_STEP4", "excluded_runs.csv")
excluded = set(pd.read_csv(excl_path).iloc[:, 0]) if os.path.exists(excl_path) else set()
obs_path = os.path.join(ROOT, "_RESULTS_JOURNAL", "predictions", "target_observability_per_run.csv")
obs = pd.read_csv(obs_path).set_index("run_id") if os.path.exists(obs_path) else None
CLS_COLOR = {"stable": "#2e9d4a", "alert": "#e0a21b", "unstable": "#d1453b"}
selected = []


def load(run_id):
    df = pd.read_csv(os.path.join(ROOT, "runs", f"{run_id}.csv"), usecols=lambda c: c in COLS)
    t = df.Time_s.to_numpy(); dt = float(np.median(np.diff(t)))
    n = max(1, int(round(F_FILTER_S / dt)))
    f = df.f_POI_Hz.to_numpy(); c = np.cumsum(np.insert(f, 0, 0.0))
    fb = np.empty_like(f); fb[n - 1:] = (c[n:] - c[:-n]) / n; fb[:n - 1] = c[1:n] / np.arange(1, n)
    fb[df.Vmag_POI_pu.to_numpy() < 0.1] = np.nan
    m = (t >= T_WIN[0]) & (t <= T_WIN[1])
    out = {"t": t[m], "V": df.Vmag_POI_pu.to_numpy()[m], "f": fb[m]}
    out["P"] = df.P_BESS_MW.to_numpy()[m] if "P_BESS_MW" in df else None
    return out


def num(s):
    d = re.findall(r"\d+", str(s)); return int(d[0]) if d else 0


def runs_of(scen, ctrl, grid, pattern=None):
    q = lab[(lab.scenario == scen) & (lab.ctrl == ctrl) & (lab.grid == grid)].copy()
    if pattern:
        q = q[q.case.astype(str).str.contains(pattern)]
    return q.sort_values("case", key=lambda s: s.map(num))


def style_time(ax):
    ax.axvline(4.0, color="#888", ls="--", lw=0.7); ax.set_xlim(*T_WIN)


# ---------------------------------------------------------------- FigS1-S3 overview
for var, ylab, fname, band in [("V", "POI voltage (pu)", "FigS1_voltage_all_scenarios", (0.9, 1.1)),
                               ("f", "Filtered POI frequency (Hz)", "FigS2_frequency_all_scenarios", (58.8, 61.2)),
                               ("P", "BESS active power (MW)", "FigS3_bess_power_all_scenarios", None)]:
    fig, axes = plt.subplots(5, 2, figsize=(10, 12), sharex=True)
    for i, scen in enumerate(SCEN):
        for j, ctrl in enumerate(["GFL", "VSM"]):
            ax = axes[i, j]
            q = runs_of(scen, ctrl, REF_GRID, FAULT_X_PATTERN if scen in (4, 5) else None)
            cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(q), 1)))
            for k, (_, r) in enumerate(q.iterrows()):
                d = load(r.run_id)
                if d[var] is None:
                    continue
                ls = ":" if r.run_id in excluded else "-"
                ax.plot(d["t"], d[var], color=cmap[k], lw=1.0, ls=ls, label=f"{r.case} ({r.label})")
                if var == "V":
                    selected.append({"figure": "S1-S3", "run_id": r.run_id, "scenario": SCEN[scen], "ctrl": ctrl,
                                     "grid": REF_GRID, "case": r.case, "label": r.label})
            if band:
                ax.axhspan(*band, color="#2e9d4a", alpha=0.07)
            style_time(ax)
            if i == 0:
                ax.set_title(f"{ctrl} BESS, SCR {q.SCR.iloc[0]:.2f}" if len(q) else ctrl, fontsize=9.5)
            if j == 0:
                ax.set_ylabel(f"{SCEN[scen]}\n{ylab}", fontsize=8)
            ax.legend(fontsize=5.8, ncol=2, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{fname}.{e}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)

# ---------------------------------------------------------------- FigS4 grid strength effect
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
for j, ctrl in enumerate(["GFL", "VSM"]):
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(LEVELS)))
    for k, g in enumerate(LEVELS):
        q = runs_of(SWEEP_CASE_SCEN, ctrl, g, SWEEP_CASE_PATTERN)
        if q.empty:
            continue
        r = q.iloc[0]; d = load(r.run_id)
        lbl = f"SCR {r.SCR:.2f} ({r.label})"
        axes[0, j].plot(d["t"], d["V"], color=cmap[k], lw=1.0, label=lbl)
        axes[1, j].plot(d["t"], d["f"], color=cmap[k], lw=1.0, label=lbl)
        selected.append({"figure": "S4", "run_id": r.run_id, "scenario": SCEN[SWEEP_CASE_SCEN], "ctrl": ctrl,
                         "grid": g, "case": r.case, "label": r.label})
    axes[0, j].set_title(f"{ctrl} BESS, {SCEN[SWEEP_CASE_SCEN].lower()} ({SWEEP_CASE_PATTERN} MW)", fontsize=9.5)
    axes[0, j].axhspan(0.9, 1.1, color="#2e9d4a", alpha=0.07); axes[1, j].axhspan(58.8, 61.2, color="#2e9d4a", alpha=0.07)
    for a in axes[:, j]:
        style_time(a)
    axes[0, j].legend(fontsize=6.2, ncol=2); axes[1, j].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("POI voltage (pu)"); axes[1, 0].set_ylabel("Filtered POI frequency (Hz)")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"FigS4_grid_strength_effect.{e}"), dpi=DPI, bbox_inches="tight")
plt.close(fig); print("saved FigS4_grid_strength_effect")

# ---------------------------------------------------------------- FigS5 fault ride-through with LVRT envelope
lv_t = np.array([0, 0.32, 0.32, 1.2, 1.2, 3.0, 3.0, 6.0])
lv_v = np.array([0.25, 0.25, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for j, ctrl in enumerate(["GFL", "VSM"]):
    q = runs_of(4, ctrl, REF_GRID, FAULT_X_PATTERN)
    q = pd.concat([q, runs_of(5, ctrl, REF_GRID, FAULT_X_PATTERN)])
    cmap = plt.cm.viridis(np.linspace(0, 0.9, max(len(q), 1)))
    for k, (_, r) in enumerate(q.iterrows()):
        d = load(r.run_id)
        axes[j].plot(d["t"], d["V"], color=cmap[k], lw=1.0, label=f"{r.case} ({r.label})")
    axes[j].axvline(4.32, color="#c0392b", ls=":", lw=1.2, label="0.32 s after inception")
    axes[j].set_xlim(3.95, 5.2); axes[j].set_title(f"{ctrl} BESS, SCR {q.SCR.iloc[0]:.2f}" if len(q) else ctrl, fontsize=9.5)
    axes[j].set_xlabel("Time (s)"); axes[j].legend(fontsize=6, ncol=2)
axes[0].set_ylabel("POI voltage (pu)")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"FigS5_fault_ride_through.{e}"), dpi=DPI, bbox_inches="tight")
plt.close(fig); print("saved FigS5_fault_ride_through")

# ---------------------------------------------------------------- summary table (all 476 runs)
L = lab.copy()
L["excluded"] = L.run_id.isin(excluded)
L["fdev"] = np.maximum((60 - L.f_nadir).abs(), (L.f_peak - 60).abs())
if obs is not None:
    L["ROCOF_500ms"] = L.run_id.map(obs.ROCOF_500ms)
rows = []
for (scen, ctrl), g in L[L.scenario.isin(SCEN)].groupby(["scenario", "ctrl"]):
    gl = g[~g.excluded]
    row = {"Disturbance": SCEN[scen], "BESS": ctrl, "Runs": len(g), "Excluded": int(g.excluded.sum()),
           "Stable": int((gl.label == "stable").sum()), "Alert": int((gl.label == "alert").sum()),
           "Unstable": int((gl.label == "unstable").sum()),
           "V_min median [range] (pu)": f"{gl.V_min.median():.3f} [{gl.V_min.min():.3f}–{gl.V_min.max():.3f}]",
           "Max |Δf| median [max] (Hz)": f"{gl.fdev.median():.2f} [{gl.fdev.max():.2f}]",
           "T_rec median (s), recovered runs": f"{gl.T_rec.dropna().median():.3f}" if gl.T_rec.notna().any() else "–"}
    if "ROCOF_500ms" in gl:
        row["ROCOF 0.5 s median [max] (Hz/s)"] = f"{gl.ROCOF_500ms.median():.2f} [{gl.ROCOF_500ms.max():.2f}]"
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, "TableS_scenario_summary.csv"), index=False)
try:
    T.to_excel(os.path.join(OUT, "TableS_scenario_summary.xlsx"), index=False)
except Exception:
    pass
pd.DataFrame(selected).to_csv(os.path.join(OUT, "selected_runs.csv"), index=False)
print("\n================ SCENARIO SUMMARY ================")
print(T.to_string(index=False))
print(f"\nReference grid level {REF_GRID}; outputs in {OUT}\nDONE.")
