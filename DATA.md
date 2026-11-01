# Machine-learning dataset

**DOI:** https://doi.org/10.5281/zenodo.23855800

| File | Content |
|---|---|
| `features.csv` | Base features and ordinal labels (0 stable, 1 alert, 2 unstable) for every window length, noise level and noise realization |
| `physics_features.csv` | Fourteen physical stability indicators |
| `feature_list.txt` | Names and descriptions of the base features |
| `physics_indicators.txt` | Indicator names, branches and risk directions |

Each row is identified by `run_id`, the window anchor, the window length, the noise level and the noise realization. The data come from 476 EMT simulations of a 100 MWp PV and 60 MWh / 35 MW BESS plant with grid-following or grid-forming BESS control, seven grid-strength levels (SCR 0.44–3.52) and 34 disturbance cases. After excluding ten runs with pre-event oscillation, the learning set contains 466 runs: 132 stable, 213 alert and 121 unstable.

The raw EMT waveforms are available from the corresponding author on reasonable request.

License: CC BY 4.0.
