# Physically Consistent Early Stability Assessment of PV–Battery Plants

Code for the paper *"Physically Consistent Early Stability Assessment of Grid-Following and Grid-Forming PV–Battery Plants via Monotone Ordinal Learning"* (Scientific Reports, under review).

## Method

The framework assesses the stability of a PV–battery plant from the first 20–100 ms of measurements after a disturbance or fault clearing and classifies each event as stable, alert or unstable. Fourteen physical stability indicators, covering oscillation, ride-through and voltage recovery, are computed from the measurements, and each is given a risk direction fixed from grid physics. Two monotone ordinal learners, **MonoOrd-GBM** (monotone gradient boosting) and **PhysOrd-Net** (physics-structured monotone network), are constrained so that the predicted severity can never decrease when a physical indicator worsens; **MonoReg-GBM** applies the same constraint to severity regression. The **physical consistency rate (PCR)** measures how often any model violates this requirement.

## Simulation model

The EMT simulations use the MathWorks Simscape Electrical example **"Evaluate Performance of Grid-Forming Battery Energy Storage Systems in Solar PV Plants"**, a 100 MWp PV plant with a 60 MWh / 35 MW battery energy storage system under grid-following or grid-forming control:
https://www.mathworks.com/help/simscape-electrical/renewable-energy.html

The model is not redistributed here. `run_scenario.m` and `run_sweep.m` apply the grid strength, BESS controller and disturbance of each run and generate the 476 simulations (MATLAB R2025b, Simscape Electrical).

## Dataset

https://doi.org/10.5281/zenodo.XXXXXXX (see DATA.md)

## Scripts

| Script | Purpose |
|---|---|
| `run_sweep.m`, `run_scenario.m` | EMT dataset generation |
| `01_ordinal_labels.py` | Grid-code-based ordinal labels |
| `02_clearing_anchored_features.py` | Clearing-anchored features |
| `03_physical_indicators.py` | Physical stability indicators |
| `04_monotone_learning_and_evaluation.py` | Monotone ordinal learners and main evaluation |
| `05_operating_points_and_noise_augmentation.py` | Operating points and noise robustness |
| `06_label_sensitivity.py` | Label sensitivity |
| `07_selected_hyperparameters.py` | Selected hyperparameters |
| `08_extended_grids_and_rocof.py` | Extended grids and 500 ms ROCOF |
| `09_severity_regression.py` | Monotone severity regression |
| `10_scenario_summary.py` | Scenario statistics |
| `11_fig_poi_voltage.py` | POI voltage figure |

Run the scripts in order after `pip install -r requirements.txt`. Set the data folder with the environment variable `PCSA_DATA`.

## License

Code: MIT. Dataset: CC BY 4.0.
