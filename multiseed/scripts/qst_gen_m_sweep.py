#!/usr/bin/env python3
"""Multi-seed wrapper around QST_Generalization/m_sweep.py.

Replicates the M-sweep at N=3 for one seed, writing results CSV under
multiseed/results/seed{S}/qst_m_sweep/results.csv.

Imports the existing modules from QST_Generalization (dataset, models,
train, energy_model, eval_multisample) so we get exact behavior parity.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
TESI_ROOT = HERE.parent.parent
QST_DIR = TESI_ROOT / "QST_Generalization"
sys.path.insert(0, str(QST_DIR))
sys.path.insert(0, str(HERE.parent))  # multiseed/ for seed_utils
from seed_utils import set_global_seed  # noqa: E402

from dataset import build_datasets, get_dataloaders  # noqa: E402
from models import get_model, get_arch_layers, HW_CONFIG  # noqa: E402
from train import train_cnn, train_vae, train_cgan  # noqa: E402
from energy_model import estimate_all_platforms  # noqa: E402
from eval_multisample import eval_fidelities, is_vae  # noqa: E402

N = 3
T_VALUES = {"SCNN": 8, "SVAE": 8, "SCGAN": 20}
M_VALUES = [12, 24, 48, 63]
STATE_TYPES = ["haar_pure", "bures_mixed"]
ARCHS = ["SCGAN", "SVAE"]
HARDWARES_TO_TRAIN = ["GPU", "Loihi", "Crossbar-8b", "Crossbar-4b"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=500)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--k_multi", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    set_global_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "results.csv"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        args.n_train, args.n_val, args.n_test = 200, 50, 50
        args.epochs = 3
        m_values = [24]
        state_types = ["haar_pure"]
        archs = ["SVAE"]
        print("*** QUICK SMOKE TEST ***", flush=True)
    else:
        m_values = M_VALUES
        state_types = STATE_TYPES
        archs = ARCHS

    df = pd.read_csv(csv_path) if csv_path.exists() else None
    done = set()
    if df is not None and len(df) > 0:
        for r in df.itertuples():
            done.add((int(r.seed), int(r.M), r.state_type, r.architecture, r.hardware))

    print(f"[seed={args.seed}] existing rows: {len(done)}", flush=True)

    ds_cache = {}

    def get_datasets(M, state_type):
        key = (M, state_type)
        if key in ds_cache:
            return ds_cache[key]
        train_ds, val_ds, test_ds, ops, _ = build_datasets(
            N, state_type=state_type, n_train=args.n_train, n_val=args.n_val,
            n_test=args.n_test, M=M, seed_states=42 + N, seed_ops=42,
        )
        train_dl, val_dl, test_dl = get_dataloaders(
            train_ds, val_ds, test_ds, batch_size=args.batch_size,
        )
        ds_cache[key] = (train_dl, val_dl, test_dl, ops)
        return ds_cache[key]

    rows = df.to_dict("records") if df is not None else []
    t_start = time.time()

    for M in m_values:
        for state_type in state_types:
            for arch in archs:
                for hw in HARDWARES_TO_TRAIN:
                    key = (args.seed, M, state_type, arch, hw)
                    if key in done:
                        print(f"  [skip] {key}", flush=True)
                        continue
                    spiking, weight_bits = HW_CONFIG[hw]
                    T = T_VALUES[arch]
                    tag = f"seed{args.seed}_{arch}_{hw}_{state_type}_M{M}"
                    ckpt_path = ckpt_dir / f"{tag}.pt"

                    print(f"\n=== seed={args.seed} M={M} {state_type} {arch} {hw} ===",
                          flush=True)
                    t0 = time.time()
                    train_dl, val_dl, test_dl, ops = get_datasets(M, state_type)

                    try:
                        if arch == "SCNN":
                            model = get_model("SCNN", M, N, hardware=hw, T=T)
                            metrics = train_cnn(model, train_dl, val_dl, test_dl, ops,
                                                epochs=args.epochs, lr=1e-3,
                                                device=device, save_path=ckpt_path,
                                                is_spiking=spiking)
                        elif arch == "SVAE":
                            model = get_model("SVAE", M, N, hardware=hw, T=T)
                            metrics = train_vae(model, train_dl, val_dl, test_dl, ops,
                                                epochs=args.epochs, lr=1e-3,
                                                device=device, save_path=ckpt_path,
                                                is_spiking=spiking)
                        elif arch == "SCGAN":
                            gen, disc = get_model("SCGAN", M, N, hardware=hw, T=T)
                            metrics = train_cgan(gen, disc, train_dl, val_dl, test_dl, ops,
                                                 epochs=args.epochs, lr=2e-4,
                                                 device=device, save_path=ckpt_path,
                                                 is_spiking=spiking)
                            model = gen
                        else:
                            raise ValueError(arch)
                    except Exception as e:
                        print(f"  ERROR: {e}", flush=True)
                        import traceback; traceback.print_exc()
                        continue

                    if spiking:
                        f1, fk, fk_std = eval_fidelities(
                            model.to(device), test_dl, k=args.k_multi,
                            device=device, vae=is_vae(arch),
                        )
                    else:
                        f1 = metrics["test_fidelity"]
                        fk = metrics["test_fidelity"]
                        fk_std = 0.0

                    layers, n_params = get_arch_layers(arch, M, N)
                    energies = estimate_all_platforms(layers, n_params, T_spiking=T)

                    dt = time.time() - t0
                    row = {
                        "seed": args.seed,
                        "N": N, "architecture": arch, "hardware": hw,
                        "variant": "spiking" if spiking else "classical",
                        "weight_bits": weight_bits, "state_type": state_type,
                        "M": M, "T_spiking": T, "d": 2 ** N,
                        "n_params": n_params, "n_train": args.n_train,
                        "test_fidelity": metrics["test_fidelity"],
                        "val_fidelity": metrics["best_val_fidelity"],
                        "test_fidelity_k1": f1,
                        f"test_fidelity_k{args.k_multi}": fk,
                        f"test_fidelity_k{args.k_multi}_std": fk_std,
                        "train_time_sec": dt,
                        "E_inference_uJ": energies[hw],
                        "epochs": args.epochs,
                    }
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(csv_path, index=False)

                    if hw == "Crossbar-8b":
                        pd_row = dict(row)
                        pd_row["hardware"] = "PdNeuRAM"
                        pd_row["E_inference_uJ"] = energies["PdNeuRAM"]
                        pd_row["train_time_sec"] = 0.0
                        rows.append(pd_row)
                        pd.DataFrame(rows).to_csv(csv_path, index=False)

                    print(f"  -> F={row['test_fidelity']:.4f}  ({dt:.1f}s)", flush=True)

    print(f"\n[seed={args.seed}] done in {(time.time()-t_start)/3600:.2f}h. "
          f"{len(rows)} total rows -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
