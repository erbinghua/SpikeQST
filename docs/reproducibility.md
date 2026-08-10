# Reproducibility guide

## Repository status

The repository preserves the supplied scripts without refactoring their numerical implementations. `seed_utils.py` was added at the repository root because every training script imports `set_global_seed`, while the helper was absent from the supplied snapshot.

## Structured-state protocol encoded in the scripts

The structured mixed state is implemented as

\[
\rho = p\lvert\mathrm{GHZ}_N\rangle\langle\mathrm{GHZ}_N\rvert
      +(1-p)I/2^N,
\]

with (p=0.5) in the benchmark calls.

The scripts use two measurement representations:

- **M1:** expectation values of non-identity (N)-qubit Pauli strings. For the main structured benchmark, the budget is capped as `min(4**N - 1, 256)`. The supplied selection function prioritizes operators with non-zero expectation on the target state and uses NumPy RNG seed 1234.
- **M2:** concatenated outcome probabilities from product-Pauli bases. The default four bases are (Z^{\otimes N}), (X^{\otimes N}), (Y^{\otimes N}), and one additional seeded random product basis.

Because M1 operator selection inspects the target state, the published paper should disclose this design. It represents a controlled benchmark selection rule, not a deployable unknown-state measurement-selection algorithm.

## Random-state and measurement-budget wrappers

`qst_gen_run_benchmarks.py` defaults to:

- architectures: SCNN, SVAE, and SCGAN;
- state types: Haar-random pure and Bures-random mixed;
- (N=3) unless overridden;
- 5,000/500/1,000 train/validation/test states;
- batch size 64 and 300 epochs;
- dataset-state seed `42 + N` and operator seed 42;
- learning rate (10^{-3}) for SCNN/SVAE and (2\times10^{-4}) for SCGAN.

Dataset sizes are capped at 2,000/200/500 for (N\ge7) and 1,000/100/200 for (N\ge8).

`qst_gen_m_sweep.py` fixes (N=3), evaluates (M\in\{12,24,48,63\}), defaults to 100 epochs, and evaluates SCGAN/SVAE on GPU, Loihi, Crossbar-8b, and Crossbar-4b. Its `k_multi=10` setting averages ten stochastic reconstructions for a test input; it is not a ten-seed experiment.

These wrappers require the absent `QST_Generalization/` package documented in the top-level README.

## Script inventory

| Script | Role |
|---|---|
| `scnn.py` | Structured SCNN/CNN benchmark |
| `scgan.py` | Structured SCGAN/CGAN benchmark |
| `svae.py` | Structured SVAE/VAE benchmark |
| `p_sweep_sc{nn,gan,vae}.py` | Mixed-state parameter-(p) sweeps |
| `m_sweep.py` | Structured measurement-count sweep |
| `capacity_sweep.py` | Model-capacity sweep |
| `t_sweep.py` | Spiking time-step sweep |
| `tau_sweep.py` | LIF time-constant sweep |
| `vth_sweep.py` | Firing-threshold sweep |
| `surrogate_sweep.py` | Surrogate-gradient sweep |
| `qst_gen_run_benchmarks.py` | Random-state benchmark wrapper |
| `qst_gen_m_sweep.py` | Random-state measurement-budget wrapper |
| `fpga_train_export.py` | FPGA training/export orchestration |

The supplied `capacity_sweep 1.py` and `t_sweep 1.py` are distinct historical variants and are retained verbatim for provenance. Use the filenames without ` 1` as the canonical entry points unless a result record explicitly identifies the historical variant.

## Multi-seed execution

Run each seed into a separate directory to avoid output collisions:

```bash
python multiseed/scripts/scnn.py --seed 0 --out_dir multiseed/results/seed0/scnn
python multiseed/scripts/scnn.py --seed 1 --out_dir multiseed/results/seed1/scnn
python multiseed/scripts/scnn.py --seed 2 --out_dir multiseed/results/seed2/scnn
```

Use `--quick` only for smoke testing. It reduces steps/dataset sizes and must not be used to reproduce the published tables.

## Environment and logging

The original package lock and hardware manifest were absent. For every new run, record:

```bash
python --version
python -m pip freeze
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Archive the command line, seed, Git commit, environment freeze, output CSV, and model checkpoint together. Do not combine new runs with the published snapshot without adding provenance columns.

## Validation performed on this snapshot

- All 17 supplied Python files compile syntactically as UTF-8.
- No common API-token, password, or hard-coded absolute-path patterns were found.
- Five core published CSVs have the columns required by the figure-generation script.
- Generated PNG figures were visually inspected for clipping and legibility.
- Numerical training was not rerun because PyTorch/Norse and the omitted project modules were unavailable in the supplied environment.

