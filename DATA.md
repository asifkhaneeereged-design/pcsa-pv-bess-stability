# Dataset description

**DOI:** https://doi.org/10.5281/zenodo.XXXXXXX

EMT simulations of a 100 MWp photovoltaic and 60 MWh / 35 MW battery energy storage plant connected to a 230 kV grid, with the BESS under grid-following (GFL) or grid-forming virtual synchronous machine (VSM) control.

## Design

476 runs = 34 disturbance cases × 2 BESS controllers × 7 grid-strength levels (SCR 0.44, 0.54, 0.69, 0.89, 1.11, 1.47, 3.52). Each disturbance is applied at 4 s; each run lasts 6 s.

| Disturbance | Swept values |
|---|---|
| Steady state | – |
| PV irradiance drop | 10, 30, 50, 70 % |
| Load rejection | 10, 20, 30, 40, 60 MW |
| Unintentional islanding | local load 20, 40, 60 MW |
| Temporary three-phase fault | clearing time 50–500 ms × location 10, 50, 99 % |
| Permanent three-phase fault | location 10, 50, 99 % |

## Contents

```
runs/                          476 waveform files (<run_id>.csv)
run_log.csv                    configuration, SCR and status of every run
_STEP3/labels_metrics.csv      ordinal label and severity metrics
_STEP4/features.csv            base features
_STEP4/physics_features.csv    physical stability indicators
_STEP4/excluded_runs.csv       ten runs excluded for pre-event oscillation
_RESULTS_JOURNAL/predictions/  fold assignments and out-of-fold predictions
```

Run identifier: `S<disturbance>_<controller>_<grid level>_<case>`, e.g. `S4_VSM_G4_tf_d150_x50` = 150 ms temporary fault at 50 % of the line, VSM BESS, SCR 0.89.

## Waveform columns (30 001 samples, 0.2 ms)

`Time_s`; `P_total_MW`, `Q_total_MVAr` (POI); `Vmag_POI_pu`; `f_POI_Hz` (PLL); `Va/Vb/Vc_POI_pu`; `Ia/Ib/Ic_POI_pu`; `P_BESS_MW`, `Q_BESS_MVAr`; `P_PV_MW`, `Q_PV_MVAr`; `Ia/Ib/Ic_BESS_pu`. Power is positive when injected into the POI (positive BESS power = discharging).

## Labels

Stable (0), alert (1) or unstable (2) from IEEE Std 2800-2022 ride-through bands (0.9–1.1 pu, 58.8–61.2 Hz), sustained-oscillation limits (0.05 pu, 0.5 Hz), a trip criterion (V < 0.1 pu and I < 0.05 pu for ≥ 50 ms) and a 1 s recovery limit. Learning set after exclusions: 466 runs (132 stable, 213 alert, 121 unstable).

## License

CC BY 4.0.
