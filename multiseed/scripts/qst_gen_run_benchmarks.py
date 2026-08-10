#!/usr/bin/env python3
"""Multi-seed wrapper around QST_Generalization/run_benchmarks.py.

Reproduces the original benchmark grid (3 archs × N=3-8 × {haar_pure,
bures_mixed} × 4 hardware platforms) for one specific seed, writing the
results CSV under multiseed/results/seed{S}/qst_generalization/.

The wrapper:
  - sets the global seed (seed_utils.set_global_seed)
  - re-points the QST_Generalization checkpoint and results dirs to a
    seed-scoped subdir so 5 concurrent runs cannot collide
  - adds `seed` as a column on every output row
  - skips combos already present in the output CSV (resume-friendly)

The existing QST_Generalization modules (dataset, train, models, etc.)
are imported as-is; only the orchestration is rewritten here.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
TESI_ROOT = HERE.parent.parent
QST_DIR = TESI_ROOT / "QST_Generalization"
sys.path.insert(0, str(QST_DIR))
sys.path.insert(0, str(HERE.parent))  # multiseed/ for seed_utils

from seed_utils import set_global_seed  # noqa: E402

from dataset import build_datasets, get_dataloaders  # noqa: E402
from models import get_arch_layers, HW_CONFIG, get_model  # noqa: E402
from train import train_cnn, train_vae, train_cgan  # noqa: E402
from energy_model import estimate_all_platforms  # noqa: E402
from qst_utils import get_default_M  # noqa: E402


# Mirror the existing T defaults from QST_Generalization/run_benchmarks.py
T_VALUES = {
    "SCNN":  {3: 8,  4: 8,  5: 8,  6: 16, 7: 32, 8: 32},
    "SVAE":  {3: 8,  4: 8,  5: 8,  6: 16, 7: 32, 8: 48},
    "SCGAN": {3: 20, 4: 20, 5: 20, 6: 32, 7: 32, 8: 44},
}

PLATFORMS = ["GPU", "Loihi", "Crossbar-8b", "Crossbar-4b"]


def existing_keys(df: "pd.DataFrame | None"):
    if df is None or len(df) == 0:
        return set()
    keys = set()
    for r in df.itertuples():
        keys.add((int(r.N), r.architecture, r.hardware, r.state_type, int(r.seed)))
    return keys


def append_row(df, row, csv_path):
    new = (
        pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        if df is not None
        else pd.DataFrame([row])
    )
    new.to_csv(csv_path, index=False)
    return new


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True,
                   help="e.g. multiseed/results/seed0/qst_generalization")
    p.add_argument("--n_qubits", type=int, nargs="+", default=[3])
    p.add_argument("--archs", nargs="+", default=["SCNN", "SVAE", "SCGAN"])
    p.add_argument("--state_types", nargs="+",
                   default=["haar_pure", "bures_mixed"])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.n_train, args.n_val, args.n_test = 200, 50, 50
        args.epochs = 3
        print("*** QUICK SMOKE TEST ***", flush=True)

    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "qst_generalization_results.csv"

    df = pd.read_csv(csv_path) if csv_path.exists() else None
    done = existing_keys(df)
    print(f"[seed={args.seed}] existing rows: {len(done)}", flush=True)

    t_start = time.time()
    for state_type in args.state_types:
        for N in args.n_qubits:
            M = get_default_M(N, state_type)
            # cap dataset size for large N (mirrors run_benchmarks.py)
            n_train, n_val, n_test = args.n_train, args.n_val, args.n_test
            if N >= 7:
                n_train, n_val, n_test = min(n_train, 2000), min(n_val, 200), min(n_test, 500)
            if N >= 8:
                n_train, n_val, n_test = min(n_train, 1000), min(n_val, 100), min(n_test, 200)

            for arch in args.archs:
                T = T_VALUES[arch][N]

                # Build datasets once per (N, state_type) for all hw variants
                train_ds, val_ds, test_ds, ops, _ = build_datasets(
                    N, state_type=state_type, n_train=n_train, n_val=n_val,
                    n_test=n_test, M=M, seed_states=42 + N, seed_ops=42,
                )
                train_dl, val_dl, test_dl = get_dataloaders(
                    train_ds, val_ds, test_ds, batch_size=args.batch_size,
                )
                layers, n_params = get_arch_layers(arch, M, N)
                energies = estimate_all_platforms(layers, n_params, T_spiking=T)

                for hw in PLATFORMS:
                    key = (N, arch, hw, state_type, args.seed)
                    if key in done:
                        print(f"  [skip] {key}", flush=True)
                        continue

                    spiking, weight_bits = HW_CONFIG[hw]
                    tag = f"{arch}_{hw}_{state_type}_N{N}_seed{args.seed}"
                    ckpt_path = ckpt_dir / f"{tag}.pt"

                    print(f"\n=== seed={args.seed} N={N} {arch} {hw} {state_type} ===",
                          flush=True)
                    t0 = time.time()
                    try:
                        if arch == "SCNN":
                            model = get_model("SCNN", M, N, hardware=hw, T=T)
                            metrics = train_cnn(model, train_dl, val_dl, test_dl,
                                                ops, epochs=args.epochs, lr=1e-3,
                                                device=args.device,
                                                save_path=ckpt_path,
                                                is_spiking=spiking)
                        elif arch == "SVAE":
                            model = get_model("SVAE", M, N, hardware=hw, T=T)
                            metrics = train_vae(model, train_dl, val_dl, test_dl,
                                                ops, epochs=args.epochs, lr=1e-3,
                                                device=args.device,
                                                save_path=ckpt_path,
                                                is_spiking=spiking)
                        elif arch == "SCGAN":
                            gen, disc = get_model("SCGAN", M, N, hardware=hw, T=T)
                            metrics = train_cgan(gen, disc, train_dl, val_dl,
                                                 test_dl, ops, epochs=args.epochs,
                                                 lr=2e-4, device=args.device,
                                                 save_path=ckpt_path,
                                                 is_spiking=spiking)
                        else:
                            raise ValueError(arch)
                    except Exception as e:
                        print(f"  ERROR ({arch} {hw} {state_type} N={N}): {e}",
                              flush=True)
                        import traceback; traceback.print_exc()
                        continue

                    dt = time.time() - t0
                    row = {
                        "seed": args.seed,
                        "N": N, "architecture": arch, "hardware": hw,
                        "variant": "spiking" if spiking else "classical",
                        "weight_bits": weight_bits,
                        "state_type": state_type, "M": M, "T_spiking": T,
                        "d": 2 ** N, "n_params": n_params,
                        "n_train": len(train_ds),
                        "test_fidelity": metrics["test_fidelity"],
                        "val_fidelity": metrics["best_val_fidelity"],
                        "train_time_sec": dt,
                        "E_inference_uJ": energies[hw],
                        "epochs": args.epochs,
                    }
                    df = append_row(df, row, csv_path)

                    # PdNeuRAM mirror of Crossbar-8b (same fidelity, different energy)
                    if hw == "Crossbar-8b":
                        pd_row = dict(row)
                        pd_row["hardware"] = "PdNeuRAM"
                        pd_row["E_inference_uJ"] = energies["PdNeuRAM"]
                        pd_row["train_time_sec"] = 0.0
                        df = append_row(df, pd_row, csv_path)

                    print(f"  -> F={row['test_fidelity']:.4f}  ({dt:.1f}s)",
                          flush=True)

    print(f"\n[seed={args.seed}] done in {(time.time()-t_start)/3600:.2f}h",
          flush=True)
    print(f"CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
