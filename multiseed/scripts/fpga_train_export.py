#!/usr/bin/env python3
"""Multi-seed wrapper for the FPGA train_and_export_*.py scripts.

For each (architecture, N qubits), invokes the appropriate FPGA export
script with --seed routed through. Records the per-run fidelity in
multiseed/results/seed{S}/fpga_train_export/results.csv.

The FPGA RTL itself is deterministic given trained weights — we only
need to re-train the PyTorch reference models across seeds. RTL
re-simulation per seed is out of scope (the RTL doesn't change).

Usage:
    python multiseed/scripts/fpga_train_export.py --seed S --out_dir OUT
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
TESI_ROOT = HERE.parent.parent
FPGA_SCRIPTS = TESI_ROOT / "FPGA Hardware" / "scripts"


# (arch_label, script_filename, N range)
ARCH_CONFIGS = [
    ("SCNN_pure",   "train_and_export.py",        [3, 4, 5, 6, 7, 8]),
    ("SCGAN_pure",  "train_and_export_scgan.py",  [3, 4, 5, 6, 7, 8]),
    ("SVAE_pure",   "train_and_export_svae.py",   [3, 4, 5, 6, 7, 8]),
]


def parse_fidelity_from_stdout(out: str) -> float:
    """The FPGA scripts print 'Training complete. Best fidelity: F.FFFFFF'."""
    for line in reversed(out.splitlines()):
        if "Best fidelity:" in line:
            try:
                return float(line.split("Best fidelity:")[1].strip())
            except Exception:
                pass
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--archs", nargs="+", default=None,
                   help="Restrict to specific arch labels (default: all)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    print(f"[multiseed] seed={args.seed} out_dir={out_dir}", flush=True)

    # Resume: load existing rows if any
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        rows = df.to_dict("records")
        done = {(int(r["seed"]), r["arch"], int(r["N"])) for r in rows
                if r.get("seed") == args.seed}
    else:
        rows = []
        done = set()

    device = args.device or ("cuda" if "torch" in sys.modules and __import__("torch").cuda.is_available() else "cpu")
    steps = 3 if args.quick else 400
    n_range = [3] if args.quick else None

    archs = ARCH_CONFIGS
    if args.archs:
        archs = [a for a in ARCH_CONFIGS if a[0] in args.archs]
    if args.quick:
        archs = archs[:1]  # just SCNN for smoke

    for arch_label, script, N_list in archs:
        if n_range is not None:
            N_list = n_range
        script_path = FPGA_SCRIPTS / script
        if not script_path.exists():
            print(f"  [skip] missing script: {script_path}", flush=True)
            continue

        for N in N_list:
            key = (args.seed, arch_label, N)
            if key in done:
                print(f"  [skip] seed={args.seed} {arch_label} N={N} already done",
                      flush=True)
                continue
            per_run_dir = out_dir / f"{arch_label}_N{N}_seed{args.seed}"
            per_run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable, str(script_path),
                "--n_qubits", str(N),
                "--steps", str(steps),
                "--output_dir", str(per_run_dir),
                "--device", device,
            ]
            env = os.environ.copy()
            # Propagate seed via PYTHONHASHSEED + a deterministic preamble we
            # inject; the FPGA scripts themselves don't accept --seed today.
            env["PYTHONHASHSEED"] = str(args.seed)
            env["TESI_SEED"] = str(args.seed)
            preamble = (
                f"import os, random; "
                f"random.seed({args.seed}); "
                f"import numpy as np; np.random.seed({args.seed}); "
                f"import torch; torch.manual_seed({args.seed}); "
                f"torch.cuda.manual_seed_all({args.seed}) if torch.cuda.is_available() else None"
            )
            # Inject via -c is invasive; simpler: prepend PYTHONSTARTUP override.
            startup_path = per_run_dir / "_seed_init.py"
            startup_path.write_text(preamble)
            env["PYTHONSTARTUP"] = str(startup_path)

            print(f"\n[seed={args.seed}] {arch_label} N={N} → {per_run_dir}",
                  flush=True)
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      cwd=TESI_ROOT)
                dt = time.time() - t0
                if proc.returncode != 0:
                    print(f"  FAILED rc={proc.returncode}: {proc.stderr[-500:]}",
                          flush=True)
                    fid = float("nan")
                else:
                    fid = parse_fidelity_from_stdout(proc.stdout)
                    print(f"  F={fid:.4f}  ({dt:.1f}s)", flush=True)
            except Exception as e:
                dt = time.time() - t0
                print(f"  EXCEPTION: {e}", flush=True)
                fid = float("nan")

            rows.append({
                "seed": args.seed,
                "arch": arch_label,
                "N": N,
                "F_best": fid,
                "train_time_sec": dt,
                "output_dir": str(per_run_dir),
            })
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\n[seed={args.seed}] done. {len(rows)} total rows in {csv_path}",
          flush=True)


if __name__ == "__main__":
    main()
