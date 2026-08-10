#!/usr/bin/env python3
"""Validate the published result tables and regenerate summary figures/docs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REQUIRED_COLUMNS = {
    "benchmarks.csv": {
        "state_type", "architecture", "hardware", "N", "method", "n_seeds",
        "F_mean", "F_std", "E_inf_uJ_mean",
    },
    "qst_generalization.csv": {
        "state_type", "architecture", "hardware", "n_seeds", "F_mean",
        "F_std", "E_inf_uJ",
    },
    "qst_m_sweep.csv": {
        "state_type", "architecture", "hardware", "M", "n_seeds",
        "F_k1_mean", "F_k10_mean", "F_k10_std", "E_inf_uJ",
    },
    "fpga.csv": {
        "architecture", "N", "n_seeds", "train_time_sec_mean",
        "train_time_sec_std",
    },
    "pdneuram_regimes.csv": {
        "state_type", "architecture", "N", "method", "F",
        "E_inf_uJ_crossbar8b", "E_inf_uJ_ns_regime",
        "E_inf_uJ_paper_10us",
    },
}

ARCH_COLORS = {"SCNN": "#2563EB", "SCGAN": "#DC2626", "SVAE": "#059669"}
HW_MARKERS = {
    "GPU": "o", "Loihi": "s", "Crossbar-8b": "^", "Crossbar-4b": "v",
    "PdNeuRAM": "D", "PdNeuRAM-inspired": "D",
}


def load_and_validate(data_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for filename, required in REQUIRED_COLUMNS.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required result file is missing: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
        if df.empty:
            raise ValueError(f"{filename} contains no rows")
        fidelity_cols = [c for c in df.columns if c == "F" or c.startswith("F_")]
        for column in fidelity_cols:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if column.endswith(("_mean", "_best")) or column == "F":
                if not values.between(0.0, 1.0).all():
                    raise ValueError(f"{filename}:{column} contains values outside [0, 1]")
            elif (values < 0).any():
                raise ValueError(f"{filename}:{column} contains negative uncertainty values")
        energy_cols = [c for c in df.columns if c.startswith("E_inf")]
        for column in energy_cols:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if (values < 0).any():
                raise ValueError(f"{filename}:{column} contains negative energy values")
        if "n_seeds" in df.columns and (pd.to_numeric(df["n_seeds"], errors="coerce") <= 0).any():
            raise ValueError(f"{filename}:n_seeds must be positive")
        frames[filename] = df
    return frames


def _arch(row: pd.Series) -> str:
    return str(row["architecture"]).upper()


def plot_energy_fidelity(df: pd.DataFrame, output: Path, title_prefix: str) -> None:
    states = list(df["state_type"].drop_duplicates())
    fig, axes = plt.subplots(1, len(states), figsize=(6.4 * len(states), 4.8), squeeze=False)
    for ax, state in zip(axes[0], states):
        part = df[df["state_type"] == state]
        energy_col = "E_inf_uJ_mean" if "E_inf_uJ_mean" in part.columns else "E_inf_uJ"
        for _, row in part.iterrows():
            arch = _arch(row)
            hw = str(row["hardware"])
            ax.errorbar(
                float(row[energy_col]), float(row["F_mean"]),
                yerr=float(row.get("F_std", 0.0)),
                marker=HW_MARKERS.get(hw, "o"), color=ARCH_COLORS.get(arch, "#475569"),
                markersize=6, capsize=2, linestyle="none", alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"Estimated inference energy ($\mu$J, log scale)")
        ax.set_ylabel("Reconstruction fidelity")
        ax.set_title(state.replace("_", " ").title())
        ax.grid(True, which="both", alpha=0.2)
    arch_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor=color, label=arch, markersize=7)
        for arch, color in ARCH_COLORS.items()
    ]
    hardware = list(dict.fromkeys(str(v) for v in df["hardware"]))
    hw_handles = [
        Line2D([0], [0], marker=HW_MARKERS.get(hw, "o"), color="#475569",
               linestyle="none", label=hw, markersize=7)
        for hw in hardware
    ]
    fig.suptitle(title_prefix)
    fig.legend(arch_handles + hw_handles, [h.get_label() for h in arch_handles + hw_handles],
               loc="lower center", ncol=min(7, len(arch_handles) + len(hw_handles)),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_measurement_sweep(df: pd.DataFrame, output: Path) -> None:
    states = list(df["state_type"].drop_duplicates())
    fig, axes = plt.subplots(1, len(states), figsize=(6.4 * len(states), 4.8), squeeze=False)
    for ax, state in zip(axes[0], states):
        part = df[df["state_type"] == state]
        for (arch, hw), group in part.groupby(["architecture", "hardware"]):
            group = group.sort_values("M")
            ax.errorbar(
                group["M"], group["F_k10_mean"], yerr=group["F_k10_std"],
                label=f"{arch}-{hw}", color=ARCH_COLORS.get(str(arch).upper(), "#475569"),
                marker=HW_MARKERS.get(str(hw), "o"), linewidth=1.2, capsize=2,
            )
        ax.set_xlabel("Measurement budget M")
        ax.set_ylabel("Fidelity (k=10 evaluation)")
        ax.set_title(state.replace("_", " ").title())
        ax.grid(True, alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("Measurement-budget sensitivity")
    fig.tight_layout(rect=(0, 0.14, 1, 0.94))
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_hardware_sensitivity(fpga: pd.DataFrame, pdn: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for arch, group in fpga.groupby("architecture"):
        group = group.sort_values("N")
        axes[0].errorbar(
            group["N"], group["train_time_sec_mean"],
            yerr=group["train_time_sec_std"], marker="o", capsize=2,
            label=arch, color=ARCH_COLORS.get(str(arch).split("_")[0].upper()),
        )
    axes[0].set_title("FPGA-related training-time proxy")
    axes[0].set_xlabel("Qubits N")
    axes[0].set_ylabel("Training time (s)")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(frameon=False)

    subset = pdn[(pdn["N"] == pdn["N"].max()) & (pdn["method"] == "M1")].copy()
    labels = [f"{r.state_type}\n{r.architecture}" for r in subset.itertuples()]
    x = np.arange(len(subset))
    width = 0.26
    axes[1].bar(x - width, subset["E_inf_uJ_crossbar8b"], width, label="Crossbar-8b")
    axes[1].bar(x, subset["E_inf_uJ_ns_regime"], width, label="PdNeuRAM ns regime")
    axes[1].bar(x + width, subset["E_inf_uJ_paper_10us"], width, label="10-us regime")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylabel(r"Estimated inference energy ($\mu$J, log scale)")
    axes[1].set_title(f"Energy-model sensitivity at N={int(pdn['N'].max())}, M1")
    axes[1].grid(True, axis="y", which="both", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_summary(frames: dict[str, pd.DataFrame], output: Path) -> None:
    bench = frames["benchmarks.csv"]
    core = bench[(bench["N"] == 8) & (bench["method"] == "M1")]
    gen = frames["qst_generalization.csv"]
    ms = frames["qst_m_sweep.csv"]

    lines = [
        "# Published result summary",
        "",
        "> These tables report trained-model fidelity alongside analytical hardware-energy estimates. "
        "Energy values are not direct, like-for-like hardware measurements.",
        "",
        "## Dataset inventory",
        "",
        "| File | Rows | Purpose |",
        "|---|---:|---|",
    ]
    purposes = {
        "benchmarks.csv": "Structured pure/mixed multi-seed benchmarks",
        "qst_generalization.csv": "Haar-pure and Bures-mixed generalization evaluation",
        "qst_m_sweep.csv": "Measurement-budget sweep at N=3",
        "fpga.csv": "FPGA-related training-time proxy",
        "pdneuram_regimes.csv": "Energy-model timing-regime sensitivity",
    }
    for name, df in frames.items():
        lines.append(f"| `{name}` | {len(df):,} | {purposes[name]} |")

    lines += ["", "## Structured benchmarks at N=8, M1", ""]
    for state, group in core.groupby("state_type"):
        row = group.loc[group["F_mean"].idxmax()]
        lines.append(
            f"- **{state.title()}:** numerically highest mean fidelity is "
            f"{row['F_mean']:.4f} +/- {row['F_std']:.4f} for "
            f"{row['architecture']}-{row['hardware']}, with estimated inference energy "
            f"{row['E_inf_uJ_mean']:.4g} uJ."
        )

    lines += [
        "",
        "## Random-state evaluation",
        "",
        "The generalization table uses only two seeds. Values below are descriptive and do not "
        "establish statistically resolved architecture rankings.",
        "",
    ]
    for state, group in gen.groupby("state_type"):
        row = group.loc[group["F_mean"].idxmax()]
        lines.append(
            f"- **{state.replace('_', ' ').title()}:** numerical maximum "
            f"F={row['F_mean']:.4f} +/- {row['F_std']:.4f} "
            f"({row['architecture']}-{row['hardware']}; {row['E_inf_uJ']:.4g} uJ estimated)."
        )

    lines += ["", "## Measurement-budget sweep", ""]
    max_m = int(ms["M"].max())
    at_max = ms[ms["M"] == max_m]
    for state, group in at_max.groupby("state_type"):
        row = group.loc[group["F_k10_mean"].idxmax()]
        lines.append(
            f"- At M={max_m}, **{state.replace('_', ' ')}** reaches a numerical maximum "
            f"k=10 fidelity of {row['F_k10_mean']:.4f} +/- {row['F_k10_std']:.4f} "
            f"for {row['architecture']}-{row['hardware']}."
        )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Structured benchmarks generally contain three seeds; the random-state table contains two.",
        "- Reported standard deviations describe run-to-run fidelity variation, not uncertainty in hardware-energy models.",
        "- GPU, Loihi-style, crossbar, PdNeuRAM-inspired, and FPGA-related values use heterogeneous analytical assumptions.",
        "- `fpga.csv` is a training-time proxy and must not be described as measured FPGA inference energy.",
        "- `F_k10` averages multiple stochastic reconstructions per test input; it is not a ten-seed statistic.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "python analysis/summarize_results.py",
        "```",
        "",
        "The command validates required columns and regenerates the figures in `figures/`.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir or root / "multiseed" / "results" / "published"
    figures_dir = args.figures_dir or root / "figures"
    summary = args.summary or root / "docs" / "results.md"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    frames = load_and_validate(data_dir)
    core = frames["benchmarks.csv"]
    core = core[(core["N"] == 8) & (core["method"] == "M1")]
    plot_energy_fidelity(core, figures_dir / "structured_energy_fidelity", "Structured benchmarks at N=8, M1")
    plot_energy_fidelity(
        frames["qst_generalization.csv"], figures_dir / "random_state_energy_fidelity",
        "Random-state evaluation (n=2 seeds)",
    )
    plot_measurement_sweep(frames["qst_m_sweep.csv"], figures_dir / "measurement_budget")
    plot_hardware_sensitivity(
        frames["fpga.csv"], frames["pdneuram_regimes.csv"],
        figures_dir / "hardware_model_sensitivity",
    )
    write_summary(frames, summary)
    print(f"Validated {len(frames)} core result tables")
    print(f"Wrote {summary}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
