# Published result summary

> These tables report trained-model fidelity alongside analytical hardware-energy estimates. Energy values are not direct, like-for-like hardware measurements.

## Dataset inventory

| File | Rows | Purpose |
|---|---:|---|
| `benchmarks.csv` | 288 | Structured pure/mixed multi-seed benchmarks |
| `qst_generalization.csv` | 30 | Haar-pure and Bures-mixed generalization evaluation |
| `qst_m_sweep.csv` | 80 | Measurement-budget sweep at N=3 |
| `fpga.csv` | 18 | FPGA-related training-time proxy |
| `pdneuram_regimes.csv` | 72 | Energy-model timing-regime sensitivity |

## Structured benchmarks at N=8, M1

- **Mixed:** numerically highest mean fidelity is 0.8677 +/- 0.0007 for SVAE-Crossbar-8b, with estimated inference energy 0.0862 uJ.
- **Pure:** numerically highest mean fidelity is 0.9998 +/- 0.0001 for SCGAN-GPU, with estimated inference energy 8.289e+04 uJ.

## Random-state evaluation

The generalization table uses only two seeds. Values below are descriptive and do not establish statistically resolved architecture rankings.

- **Bures Mixed:** numerical maximum F=0.7455 +/- 0.0014 (SVAE-GPU; 2.945 uJ estimated).
- **Haar Pure:** numerical maximum F=0.6709 +/- 0.0023 (SCNN-GPU; 80.09 uJ estimated).

## Measurement-budget sweep

- At M=63, **bures mixed** reaches a numerical maximum k=10 fidelity of 0.8848 +/- 0.0086 for SCGAN-GPU.
- At M=63, **haar pure** reaches a numerical maximum k=10 fidelity of 0.9909 +/- 0.0003 for SCGAN-GPU.

## Interpretation limits

- Structured benchmarks generally contain three seeds; the random-state table contains two.
- Reported standard deviations describe run-to-run fidelity variation, not uncertainty in hardware-energy models.
- GPU, Loihi-style, crossbar, PdNeuRAM-inspired, and FPGA-related values use heterogeneous analytical assumptions.
- `fpga.csv` is a training-time proxy and must not be described as measured FPGA inference energy.
- `F_k10` averages multiple stochastic reconstructions per test input; it is not a ten-seed statistic.

## Regeneration

```bash
python analysis/summarize_results.py
```

The command validates required columns and regenerates the figures in `figures/`.
