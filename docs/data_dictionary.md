# Published data dictionary

The source tables are stored in `multiseed/results/published/`. Original files are retained; generated summaries do not overwrite them.

## Common fields

| Field | Meaning |
|---|---|
| `state_type` | State ensemble, such as structured `pure`/`mixed`, `haar_pure`, or `bures_mixed` |
| `architecture` | SCNN, SCGAN, or SVAE family |
| `model` | More specific implementation/model label |
| `hardware` / `hw` | GPU or analytical target-hardware model |
| `N` | Number of qubits |
| `method` | M1 expectation-value or M2 outcome-probability representation |
| `M` | Number of measurement operators/settings, as defined by the generating experiment |
| `T` / `T_spiking` | Number of spiking simulation time steps |
| `n_seeds` | Number of independent training runs aggregated in the row |
| `n_params` | Trainable parameter count |

## Fidelity fields

| Field | Meaning |
|---|---|
| `F_mean` | Mean reconstruction fidelity across reported seeds |
| `F_std` | Standard deviation of reconstruction fidelity across reported seeds |
| `F_sem` | Standard error of the reported fidelity mean |
| `F_k1_mean` | Mean single stochastic reconstruction fidelity |
| `F_k10_mean` | Mean fidelity after ten stochastic reconstructions per input |
| `F_k10_std` | Reported standard deviation for the (k=10) evaluation |
| `F_train_mean` | Mean training/evaluation fidelity field emitted by the sweep pipeline |

Fidelity standard deviations must not be interpreted as uncertainty in the analytical energy model.

## Cost and timing fields

| Field | Unit | Meaning |
|---|---:|---|
| `E_inf_uJ` / `E_inference_uJ` / `E_inf_uJ_mean` | microjoules | Analytical inference-energy estimate |
| `E_inf_uJ_std` | microjoules | Variation present in the aggregated source rows, not a universal hardware uncertainty bound |
| `E_train_mJ` / `E_training_mJ` | millijoules | Training-energy estimate where available |
| `train_time_sec_mean` | seconds | FPGA-related training-time proxy mean |
| `train_time_sec_std` | seconds | Standard deviation of the timing proxy |
| `time_sec` / `Time_sec` | seconds | Runtime field emitted by the generating script |

## Source tables

| File | Content |
|---|---|
| `benchmarks.csv` | Aggregated structured-state benchmarks |
| `qst_generalization.csv` | Aggregated Haar/Bures evaluation; two seeds in the supplied table |
| `qst_m_sweep.csv` | (N=3) measurement-budget sweep with (k=1) and (k=10) fields |
| `sweeps_pure.csv` | Aggregated pure-state hyperparameter sweeps |
| `sweeps_mixed.csv` | Aggregated mixed-state hyperparameter sweeps |
| `pdneuram_regimes.csv` | Timing-regime sensitivity of analytical PdNeuRAM-inspired estimates |
| `fpga.csv` | FPGA-related training-time proxy |
| `all_results.csv` | Pre-aggregation result export |
| `all_mixed_states_results.csv` | Pre-aggregation mixed-state sweep export |
| `multiseed_results.xlsx` | Workbook snapshot containing collected results |

