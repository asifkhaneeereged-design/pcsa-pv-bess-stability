# Machine-learning dataset

**DOI:** https://doi.org/10.5281/zenodo.XXXXXXX

Labels, features and physical stability indicators derived from 476 EMT simulations of a 100 MWp PV and 60 MWh / 35 MW BESS plant with grid-following (GFL) or grid-forming (VSM) BESS control, across seven grid-strength levels (SCR 0.44–3.52) and 34 disturbance cases (PV drop, load rejection, islanding, temporary and permanent three-phase faults).

## Files

| File | Content |
|---|---|
| `run_log.csv` | Configuration and short-circuit ratio of every run |
| `labels_metrics.csv` | Ordinal label (0 stable, 1 alert, 2 unstable) and severity metrics of every run |
| `excluded_runs.csv` | Ten runs excluded for pre-event oscillation |
| `features.csv` | Base features for every window length, noise level and noise realization |
| `feature_list.txt` | Names and descriptions of the base features |
| `physics_features.csv` | Fourteen physical stability indicators |
| `physics_indicators.txt` | Indicator names, branches and risk directions |
| `folds_seed0.csv` … `folds_seed4.csv` | Outer cross-validation fold of every run for each seed |

Each row is identified by `run_id`, the window anchor, the window length, the noise level and the noise realization. The learning set contains 466 runs: 132 stable, 213 alert and 121 unstable.

The raw EMT waveforms are available from the corresponding author on reasonable request.

## License

CC BY 4.0.
