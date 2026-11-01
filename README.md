# Physically Consistent Early Stability Assessment of PV–Battery Plants

Code accompanying the paper:

> **Physically Consistent Early Stability Assessment of Grid-Following and Grid-Forming PV–Battery Plants via Monotone Ordinal Learning.**
> [Author names]. *Scientific Reports* (under review), 2026.

The repository contains the MATLAB scripts that generate the electromagnetic-transient (EMT) dataset and the Python scripts that label the runs, extract features and physical stability indicators, train the proposed monotone learners (MonoOrd-GBM, PhysOrd-Net, MonoReg-GBM) and the baselines, and reproduce the results of the paper.

## Dataset

The EMT dataset (476 runs, about 4.6 GB) is archived on Zenodo: **https://doi.org/10.5281/zenodo.XXXXXXX**. See [DATA.md](DATA.md).

## Requirements

- MATLAB R2025b with Simscape Electrical (only to regenerate the EMT data)
- Python 3.10 or later: `pip install -r requirements.txt`

## Plant model

The EMT model is developed from the MathWorks Simscape Electrical example *100 MWp PV Plant & 60 MWh Battery Energy Storage System with Grid Forming Control*. The model is not redistributed here; open it in MATLAB with

```matlab
openExample('simscapeelectrical/BatteryStoragePVPlantGFMExample')
```

and place `matlab/run_scenario.m` and `matlab/run_sweep.m` in the model folder. All study-specific settings are applied by these scripts at run time.

## Data folder

All scripts read and write one data folder, set by the environment variable `PCSA_DATA` (default `./data`):

```bash
# Windows (PowerShell)
$env:PCSA_DATA = "D:\pcsa_data"
# Linux / macOS
export PCSA_DATA=/path/to/pcsa_data
```

To reproduce the results without re-simulating, extract the Zenodo archive into this folder and start from step 1 of the Python pipeline.

## Pipeline

| # | Script | Paper section | Output | Time* |
|---|---|---|---|---|
| 0 | `matlab/run_sweep.m` | Simulation protocol and data acquisition | `runs/*.csv`, `run_log.csv` | 8.3 h |
| 1 | `python/01_ordinal_labels.py` | Grid-code-based ordinal stability labelling | labels, Figure 7 | 10 min |
| 2 | `python/02_clearing_anchored_features.py` | Clearing-anchored observation window | base features | 20 min |
| 3 | `python/03_physical_indicators.py` | Physical stability indicators | 14 indicators | 10 min |
| 4 | `python/04_monotone_learning_and_evaluation.py` | Proposed learners; accuracy and physical consistency | Tables 14, 17; Figure 8 | 35 min |
| 5 | `python/05_operating_points_and_noise_augmentation.py` | Operating points, channel importance, noise augmentation | Figures 9, 10 | 5 min |
| 6 | `python/06_label_sensitivity.py` | Label sensitivity and error analysis | Figure 10 | 8 min |
| 7 | `python/07_selected_hyperparameters.py` | Selected hyperparameters | Table 11 | 8 min |
| 8 | `python/08_extended_grids_and_rocof.py` | Extended grids, 500 ms ROCOF | Tables 11, 16 | 5 min |
| 9 | `python/09_severity_regression.py` | Severity regression | Table 15 | 6 min |
| 10 | `python/10_scenario_summary.py` | EMT simulation results | per-scenario statistics | 2 min |
| 11 | `python/11_fig_poi_voltage.py` | EMT simulation results | Figure 6 (a–j) | 2 min |

\*AMD Ryzen 7 7700 (8 cores), 16 GB RAM, Windows 11 Pro. Run the steps in order; each step reads the outputs of the previous ones.

Results are written to `$PCSA_DATA/_RESULTS_JOURNAL/` (`tables/`, `figures/`, `predictions/`). The `predictions/` folder contains the fold assignment of every run for every seed and the out-of-fold predictions of every model.

## Reproducibility

The EMT model contains no random elements, so the waveforms are exactly reproducible. All randomness in learning is controlled by fixed seeds (cross-validation seeds 0–4; measurement noise seeded by a hash of the run identifier, noise level and realization).

## Citation

Please cite the paper when using this code or dataset (see [CITATION.cff](CITATION.cff)).

## License

Code: MIT License. Dataset: CC BY 4.0.
