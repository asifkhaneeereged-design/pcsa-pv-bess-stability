"""
POI voltage responses for every disturbance and controller (Figure 6, panels a-j).
"""
import os, re, math, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================================================== SETTINGS
ROOT = __import__("os").environ.get("PCSA_DATA", "data")   # data folder (see README)
REF_GRID = "G4"            # SCR 0.89
FAULT_X_PATTERN = "x50"    # fault location for temporary / permanent faults
T_WIN = (3.9, 6.0)         # s; event at 4 s
FONT_SIZE = 14             # same for title, labels, ticks and legend
LINE_W = 2.2
FIG_SIZE = (8.0, 5.2)      # inches, identical for all figures
DPI = 600
SHOW_BAND = True           # shade the 0.9-1.1 pu band
FIG_NO = 6                 # paper figure number -> Fig6a ... Fig6j
# ======================================================================

OUT = os.path.join(ROOT, "_RESULTS_JOURNAL", "scenario_figures_separate")
os.makedirs(OUT, exist_ok=True)

# ---- font: Times New Roman if installed (Windows), otherwise a metric-compatible serif
avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((f for f in ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"] if f in avail), "serif")
plt.rcParams.update({
    "font.family": FONT, "font.size": FONT_SIZE, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "axes.linewidth": 1.4, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
    "xtick.direction": "in", "ytick.direction": "in", "axes.grid": True,
    "grid.color": "#c8c8c8", "grid.linewidth": 0.6, "grid.alpha": 0.7, "mathtext.default": "bf",
})
print("Font used:", FONT)

SCEN = {1: ("PV irradiance drop", "pv_drop"), 2: ("Load rejection", "load_rejection"),
        3: ("Unintentional islanding", "islanding"), 4: ("Temporary three-phase fault", "temporary_fault"),
        5: ("Permanent three-phase fault", "permanent_fault")}
COLORS = ["#1f4e9c", "#d1453b", "#2e9d4a", "#e08a00", "#7b3fa0", "#0f9aa8"]

lab = pd.read_csv(os.path.join(ROOT, "_STEP3", "labels_metrics.csv"))
lab["case"] = lab.run_id.str.split("_", n=3).str[3]


def num(s):
    d = re.findall(r"\d+", str(s)); return int(d[0]) if d else 0


def runs_of(scen, ctrl):
    q = lab[(lab.scenario == scen) & (lab.ctrl == ctrl) & (lab.grid == REF_GRID)].copy()
    if scen in (4, 5):
        q = q[q.case.astype(str).str.contains(FAULT_X_PATTERN)]
    return q.sort_values("case", key=lambda s: s.map(num))


def legend_text(scen, case, label):
    n = num(case)
    txt = {1: f"{n} % irradiance drop", 2: f"P2 = {n} MW", 3: f"P1 = {n} MW",
           4: f"tf = {n} ms", 5: "Permanent fault"}[scen]
    return f"{txt} ({label})"


def load_v(run_id):
    d = pd.read_csv(os.path.join(ROOT, "runs", f"{run_id}.csv"), usecols=["Time_s", "Vmag_POI_pu"])
    d = d[(d.Time_s >= T_WIN[0]) & (d.Time_s <= T_WIN[1])]
    return d.Time_s.to_numpy(), d.Vmag_POI_pu.to_numpy()


made = []
letters = iter("abcdefghij")
for scen, (title, tag) in SCEN.items():
    for ctrl in ["GFL", "VSM"]:
        q = runs_of(scen, ctrl)
        if q.empty:
            print("no runs for", title, ctrl); continue
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        lo, hi = np.inf, -np.inf
        for k, (_, r) in enumerate(q.iterrows()):
            t, v = load_v(r.run_id)
            ax.plot(t, v, color=COLORS[k % len(COLORS)], lw=LINE_W, label=legend_text(scen, r.case, r.label), zorder=3)
            lo, hi = min(lo, np.nanmin(v)), max(hi, np.nanmax(v))
        if SHOW_BAND:
            ax.axhspan(0.9, 1.1, color="#2e9d4a", alpha=0.08, zorder=1)
            if lo > 0.85:                                    # band visible only when the data are near it
                lo, hi = min(lo, 0.9), max(hi, 1.1)
        ax.axvline(4.0, color="#555555", ls="--", lw=1.2, zorder=2)
        # reserve a band above the data for the legend so it never covers a line
        n = len(q); ncol = 1 if n == 1 else 2
        rows = math.ceil(n / ncol)
        span = max(hi - lo, 0.05)
        ax.set_ylim(lo - 0.04 * span, hi + span * (0.06 + 0.13 * rows))
        ax.set_xlim(*T_WIN)
        leg = ax.legend(loc="upper center", ncol=ncol, frameon=True, framealpha=1.0, edgecolor="#444444",
                        handlelength=1.8, columnspacing=1.0, borderaxespad=0.3, prop={"weight": "bold", "size": FONT_SIZE})
        leg.set_zorder(5)
        letter = next(letters)
        ax.set_title(f"({letter}) {title}, {ctrl} BESS (SCR {q.SCR.iloc[0]:.2f})", pad=8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("POI voltage (pu)")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        fig.tight_layout()
        fn = f"Fig{FIG_NO}{letter}_{tag}_{ctrl}.png"
        fig.savefig(os.path.join(OUT, fn), dpi=DPI, bbox_inches="tight")
        plt.close(fig); made.append(fn); print("saved", fn)

print(f"\n{len(made)} figures saved in {OUT}\nDONE.")
