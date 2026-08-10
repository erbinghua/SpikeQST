#!/usr/bin/env python3
"""AUTO-GENERATED from SCGAN_Mixed_V1.ipynb.

Manual patches applied: see CONVERSION_CHECKLIST.md.
"""
import os, sys, argparse
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")  # headless: no interactive plotting on VM

# Make seed_utils importable regardless of where the script is launched from
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent.parent))  # multiseed/
from seed_utils import set_global_seed  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed for this run (training stochasticity).")
    p.add_argument("--out_dir", required=True,
                   help="Output dir; CSVs and checkpoints land here.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: smallest config, few steps.")
    p.add_argument("--device", default=None,
                   help="Override torch device (e.g. cpu/cuda).")
    return p.parse_args()


_ARGS = _parse_args()
SEED = _ARGS.seed
OUT_DIR = Path(_ARGS.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
set_global_seed(SEED)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "results.csv"
# === MULTISEED BENCHMARK PATCHED ===


# ─── MULTISEED HELPERS ────────────────────────────────────────────────────
def _seed_done(method_label):
    """Has this seed × method combo already produced rows in CSV_PATH?"""
    if not CSV_PATH.exists():
        return False
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return False
    if df.empty or 'seed' not in df.columns or 'method' not in df.columns:
        return False
    return ((df['seed'] == SEED) & (df['method'] == method_label)).any()


def _save_with_seed(results_dict, method_label, flatten_fn):
    """Flatten the nested results dict to a DataFrame, add seed/method, append to CSV.

    Tries to pass method_label to flatten_fn if the function accepts it
    (e.g. create_results_table(results, method_name='M1')).
    """
    import inspect
    sig = inspect.signature(flatten_fn)
    try:
        if 'method_name' in sig.parameters:
            df_new = flatten_fn(results_dict, method_name=method_label)
        else:
            df_new = flatten_fn(results_dict)
    except Exception as e:
        print(f"[multiseed] flatten_fn failed: {e}; falling back to manual flatten",
              flush=True)
        df_new = pd.DataFrame()
    if df_new is None or len(df_new) == 0:
        # Final fallback: manually flatten any nested dict-of-lists-of-dicts
        rows = []
        def _walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for r in o:
                    if isinstance(r, dict):
                        rows.append(r)
                    else:
                        _walk(r)
        _walk(results_dict)
        df_new = pd.DataFrame(rows) if rows else pd.DataFrame()
    df_new['seed'] = SEED
    df_new['method'] = method_label
    if CSV_PATH.exists():
        df_old = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(CSV_PATH, index=False)
    print(f"[multiseed] saved {len(df_new)} rows for seed={SEED} method={method_label} → {CSV_PATH}",
          flush=True)
# ──────────────────────────────────────────────────────────────────────────
print(f"[multiseed] seed={SEED}  out_dir={OUT_DIR}", flush=True)


# === cell #0 ===
# =============================================================================
# IMPORTS & SETUP
# =============================================================================

import math, random, os, time, itertools, gc
# Anti-fragmentation: must be set BEFORE any CUDA call
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Norse imports
import norse.torch as norse
from norse.torch import LIFParameters

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FDTYPE = torch.float32
CDTYPE = torch.complex64


print("="*80)
print("SCGAN ENERGY BENCHMARK: Mixed GHZ States (Werner p=0.5)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")


# === cell #1 ===
# =============================================================================
# HARDWARE ENERGY PARAMETERS
# =============================================================================

GPU_PARAMS = {
    'name': 'NVIDIA A100 (7nm)',
    'E_MAC': 35e-12,
    'E_DRAM': 640e-12,
    'E_L2_cache': 5e-12,
    'cache_hit_rate': 0.85,
    'TDP_W': 400,
}

LOIHI_PARAMS = {
    'name': 'Intel Loihi (14nm)',
    'E_spike': 23.6e-12,
    'E_synapse': 81e-15,
    'E_neuron_leak': 0.5e-12,
    'E_router': 10e-12,
}

MEMRISTOR_PARAMS = {
    'name': 'Memristor Crossbar (65nm)',
    'E_MAC_analog': 2e-15,
    'E_ADC_8bit': 20e-12,
    'E_ADC_4bit': 5e-12,
    'E_DAC_8bit': 10e-12,
    'E_DAC_4bit': 2.5e-12,
    'E_spike_gen': 3e-12,
    'E_write': 100e-12,
    'E_leakage': 0.1e-12,
}

print("✓ Hardware parameters defined")


# === cell #2 ===
# =============================================================================
# ENERGY ESTIMATION FUNCTIONS (CORRECTED: per-layer MAC counting)
# =============================================================================
# Fixes applied:
#   - GPU: per-layer MACs instead of n_params*2
#   - Loihi/Crossbar: analytical spatial dims instead of hardcoded fm=16
#   - CrossbarLinear/CrossbarConv2d: explicit hasattr checks
#   - T=1 for non-spiking models (CNN, CGAN, VAE)
# Energy formulas identical to pure-states analytical_energy_recompute.py

def _is_linear_like(m):
    """Return True for nn.Linear OR any custom layer that exposes
    in_features/out_features/weight (e.g. QuantizedLinear).
    Needed because the crossbar models use QuantizedLinear which does not
    inherit from nn.Linear, so plain isinstance checks miss them."""
    import torch.nn as _nn
    return isinstance(m, _nn.Linear) or (
        hasattr(m, 'in_features') and hasattr(m, 'out_features') and hasattr(m, 'weight')
    )

def count_macs_per_inference(model, T=8, proj_hw=None):
    """Count actual MAC operations per inference with T spiking timesteps.

    Uses forward hooks to capture output spatial dimensions.
    Falls back to analytical dims via proj_hw if hooks miss layers.
    """
    import torch, torch.nn as nn
    macs = {}
    hooks, shapes = [], {}
    def make_hook(name):
        def hook_fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                shapes[name] = out.shape
        return hook_fn
    for n, m in model.named_modules():
        if (_is_linear_like(m) or isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))):
            hooks.append(m.register_forward_hook(make_hook(n)))
    # Determine dummy input
    first_lin = None
    for m in model.modules():
        if _is_linear_like(m):
            first_lin = m; break
    if first_lin is not None:
        dummy = torch.zeros(1, first_lin.in_features, device=next(model.parameters()).device)
    else:
        dummy = torch.zeros(1, 1, device=next(model.parameters()).device)
    with torch.no_grad():
        try:
            model.eval()
            model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()

    for name, m in model.named_modules():
        if _is_linear_like(m):
            macs[name] = m.in_features * m.out_features * T
        elif isinstance(m, nn.Conv2d):
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            if name in shapes and len(shapes[name]) == 4:
                H_out, W_out = shapes[name][2], shapes[name][3]
            elif proj_hw is not None:
                H_out, W_out = proj_hw  # fallback to analytical
            else:
                H_out, W_out = 1, 1
            macs[name] = m.out_channels * m.in_channels * k * k * H_out * W_out * T
        elif isinstance(m, nn.ConvTranspose2d):
            k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
            if name in shapes and len(shapes[name]) == 4:
                H_out, W_out = shapes[name][2], shapes[name][3]
            elif proj_hw is not None:
                H_out, W_out = proj_hw
            else:
                H_out, W_out = 1, 1
            macs[name] = m.out_channels * m.in_channels * k * k * H_out * W_out * T
    return macs


def count_neurons(model, T=8, proj_hw=None):
    """Count total spiking neurons using actual spatial dims."""
    import torch, torch.nn as nn
    hooks, shapes = [], {}
    def make_hook(name):
        def hook_fn(m, inp, out):
            if isinstance(out, torch.Tensor):
                shapes[name] = out.shape
        return hook_fn
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(make_hook(n)))
    first_lin = None
    for m in model.modules():
        if _is_linear_like(m):
            first_lin = m; break
    if first_lin is not None:
        dummy = torch.zeros(1, first_lin.in_features, device=next(model.parameters()).device)
    else:
        dummy = torch.zeros(1, 1, device=next(model.parameters()).device)
    with torch.no_grad():
        try:
            model.eval()
            model(dummy)
        except Exception:
            pass
    for h in hooks:
        h.remove()
    n_neurons = 0
    for name, m in model.named_modules():
        if _is_linear_like(m):
            n_neurons += m.out_features
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if name in shapes and len(shapes[name]) == 4:
                n_neurons += m.out_channels * shapes[name][2] * shapes[name][3]
            elif proj_hw is not None:
                n_neurons += m.out_channels * proj_hw[0] * proj_hw[1]
            else:
                n_neurons += m.out_channels
        # Also catch CrossbarLinear / CrossbarLIFLinear
        elif hasattr(m, 'crossbar') and hasattr(m.crossbar, 'weight'):
            n_neurons += m.crossbar.weight.size(0)
        elif hasattr(m, 'weight') and hasattr(m, 'in_features') and not _is_linear_like(m):
            n_neurons += m.weight.size(0)
    return n_neurons


def estimate_gpu_inference_energy(model, T=1, batch_size=1, proj_hw=None):
    """GPU inference energy. T=1 for non-spiking (CNN/CGAN/VAE)."""
    n_params = sum(p.numel() for p in model.parameters())
    macs_dict = count_macs_per_inference(model, T=T, proj_hw=proj_hw)
    n_macs = sum(macs_dict.values()) * batch_size
    E_compute = n_macs * GPU_PARAMS['E_MAC']
    n_mem = n_params * batch_size
    miss_rate = 1 - GPU_PARAMS['cache_hit_rate']
    E_memory = n_mem * miss_rate * GPU_PARAMS['E_DRAM'] + n_mem * GPU_PARAMS['cache_hit_rate'] * GPU_PARAMS['E_L2_cache']
    E_total = E_compute + E_memory
    return {'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
            'E_compute_J': E_compute, 'E_memory_J': E_memory,
            'n_macs': n_macs, 'n_params': n_params,
            'breakdown': {'compute_%': E_compute/E_total*100, 'memory_%': E_memory/E_total*100}}


def estimate_gpu_training_energy(model, steps, batch_size=1, measured_time_s=None):
    """GPU training energy."""
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 100_000: util = 0.05
    elif n_params < 1_000_000: util = 0.15
    else: util = 0.30
    if measured_time_s is not None and measured_time_s > 0:
        E_total = GPU_PARAMS['TDP_W'] * util * measured_time_s
    else:
        n_macs_total = n_params * 2 * 3 * steps * batch_size
        E_compute = n_macs_total * GPU_PARAMS['E_MAC']
        E_memory = n_params * steps * 3 * GPU_PARAMS['E_L2_cache']
        E_total = E_compute + E_memory
    return {'E_total_J': E_total, 'E_total_mJ': E_total * 1e3,
            'n_params': n_params, 'utilization': util,
            'steps': steps, 'measured_time_s': measured_time_s}


def estimate_loihi_inference_energy(model, T=8, sparsity=0.1, batch_size=1, proj_hw=None):
    """Loihi neuromorphic inference energy."""
    n_params = sum(p.numel() for p in model.parameters())
    n_neurons = count_neurons(model, T=T, proj_hw=proj_hw)
    macs_dict = count_macs_per_inference(model, T=T, proj_hw=proj_hw)
    total_macs = sum(macs_dict.values())
    n_syn_ops = total_macs * sparsity * batch_size
    n_spikes = n_neurons * T * sparsity * batch_size
    n_leak_ops = n_neurons * T * batch_size
    E_syn = n_syn_ops * LOIHI_PARAMS['E_synapse']
    E_spikes = n_spikes * LOIHI_PARAMS['E_spike']
    E_leak = n_leak_ops * LOIHI_PARAMS['E_neuron_leak']
    E_routing = n_spikes * LOIHI_PARAMS['E_router']
    E_total = E_syn + E_spikes + E_leak + E_routing
    return {'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
            'E_syn_J': E_syn, 'E_spikes_J': E_spikes,
            'E_leak_J': E_leak, 'E_routing_J': E_routing,
            'n_params': n_params, 'n_neurons': n_neurons, 'n_spikes': n_spikes,
            'T': T, 'sparsity': sparsity,
            'breakdown': {
                'syn_%': E_syn/E_total*100 if E_total > 0 else 0,
                'spikes_%': E_spikes/E_total*100 if E_total > 0 else 0,
                'leak_%': E_leak/E_total*100 if E_total > 0 else 0,
                'routing_%': E_routing/E_total*100 if E_total > 0 else 0}}


def estimate_crossbar_inference_energy(model, T=8, sparsity=0.1, bits=8, batch_size=1, proj_hw=None):
    """Memristor crossbar inference energy. ADC dominates (70-90%)."""
    E_ADC = MEMRISTOR_PARAMS['E_ADC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_ADC_4bit']
    E_DAC = MEMRISTOR_PARAMS['E_DAC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_DAC_4bit']
    n_params = sum(p.numel() for p in model.parameters())

    import torch as _torch
    hooks, shapes = [], {}
    def _make_hook(name):
        def hook_fn(m, inp, out):
            if isinstance(out, _torch.Tensor): shapes[name] = out.shape
        return hook_fn
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(_make_hook(n)))
    first_lin = None
    for m in model.modules():
        if _is_linear_like(m): first_lin = m; break
    if first_lin is not None:
        _dummy = _torch.zeros(1, first_lin.in_features, device=next(model.parameters()).device)
    else:
        _dummy = _torch.zeros(1, 1, device=next(model.parameters()).device)
    with _torch.no_grad():
        try: model.eval(); model(_dummy)
        except: pass
    for h in hooks: h.remove()

    total_in, total_out = 0, 0
    for name, m in model.named_modules():
        if _is_linear_like(m):
            total_in += m.in_features; total_out += m.out_features
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            if name in shapes and len(shapes[name]) == 4:
                fm = shapes[name][2] * shapes[name][3]
            elif proj_hw is not None:
                fm = proj_hw[0] * proj_hw[1]
            else:
                fm = 1
            total_in += m.in_channels * fm; total_out += m.out_channels * fm
        elif hasattr(m, 'crossbar') and hasattr(m.crossbar, 'weight'):
            total_in += m.crossbar.weight.size(1); total_out += m.crossbar.weight.size(0)
        elif hasattr(m, 'weight') and hasattr(m, 'in_features') and not _is_linear_like(m):
            total_in += m.in_features; total_out += m.out_features

    macs_dict = count_macs_per_inference(model, T=T, proj_hw=proj_hw)
    total_macs = sum(macs_dict.values())
    n_dac = total_in * T * sparsity * batch_size
    n_adc = total_out * T * batch_size  # NO sparsity
    n_mvm = total_macs * sparsity * batch_size
    n_spikes = total_out * T * sparsity * batch_size
    E_dac = n_dac * E_DAC
    E_adc = n_adc * E_ADC
    E_mvm = n_mvm * MEMRISTOR_PARAMS['E_MAC_analog']
    E_spike = n_spikes * MEMRISTOR_PARAMS['E_spike_gen']
    E_leak = total_out * T * batch_size * MEMRISTOR_PARAMS['E_leakage']
    E_total = E_dac + E_adc + E_mvm + E_spike + E_leak
    return {'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
            'E_dac_J': E_dac, 'E_adc_J': E_adc, 'E_mvm_J': E_mvm,
            'E_spike_J': E_spike, 'E_leak_J': E_leak,
            'n_params': n_params, 'bits': bits, 'T': T, 'sparsity': sparsity,
            'breakdown': {
                'DAC_%': E_dac/E_total*100 if E_total > 0 else 0,
                'ADC_%': E_adc/E_total*100 if E_total > 0 else 0,
                'MVM_%': E_mvm/E_total*100 if E_total > 0 else 0,
                'spike_%': E_spike/E_total*100 if E_total > 0 else 0,
                'leak_%': E_leak/E_total*100 if E_total > 0 else 0}}

print("✓ Energy estimation functions defined (corrected: per-layer MAC counting)")


# === cell #3 ===
# =============================================================================
# QUANTUM UTILITIES (FIXED FOR GPU)
# =============================================================================

SQRT2 = 1.0/math.sqrt(2.0)

def get_pauli(c, device):
    """Get Pauli matrix on specified device."""
    if c == 'I':
        return torch.eye(2, dtype=CDTYPE, device=device)
    elif c == 'X':
        return torch.tensor([[0.,1.],[1.,0.]], dtype=CDTYPE, device=device)
    elif c == 'Y':
        return torch.tensor([[0.,-1j],[1j,0.]], dtype=CDTYPE, device=device)
    elif c == 'Z':
        return torch.tensor([[1.,0.],[0.,-1.]], dtype=CDTYPE, device=device)

def get_rotation_unitary(c, device):
    """Get rotation unitary for basis c on specified device."""
    if c == 'X':
        return torch.tensor([[1., 1.],[1.,-1.]], dtype=CDTYPE, device=device) * SQRT2
    elif c == 'Y':
        H = torch.tensor([[1., 1.],[1.,-1.]], dtype=CDTYPE, device=device) * SQRT2
        S_dag = torch.tensor([[1.,0.],[0.,-1j]], dtype=CDTYPE, device=device)
        return H @ S_dag
    else:  # Z
        return torch.eye(2, dtype=CDTYPE, device=device)

def ghz_state(N, device=None):
    if device is None:
        device = DEVICE
    d = 2**N
    psi = torch.zeros(d, dtype=CDTYPE, device=device)
    psi[0] = 1.0 / math.sqrt(2)
    psi[-1] = 1.0 / math.sqrt(2)
    return psi


def mixed_ghz_state(N, p=0.5, device=None):
    """
    Generalized Werner state (Eq. A4 from Hua et al., arXiv:2507.23007).

    rho = p |GHZ><GHZ| + (1 - p) I_N / 2^N

    Args:
        N: Number of qubits
        p: Pure-state weight (1 = pure GHZ, 0 = maximally mixed)

    Returns:
        rho: (1, d, d) complex64 density matrix tensor
    """
    if device is None:
        device = DEVICE
    d = 2**N
    psi = ghz_state(N, device=device)
    rho_pure = torch.outer(psi, psi.conj())
    I_d = torch.eye(d, dtype=CDTYPE, device=device) / d
    rho = p * rho_pure + (1 - p) * I_d
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / rho.diagonal().sum().real.clamp(min=1e-12)
    return rho.unsqueeze(0)  # (1, d, d)

def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out

def build_rotation_unitary(basis_str, device=None):
    """Build rotation unitary on the correct device."""
    if device is None:
        device = DEVICE
    return kron_all([get_rotation_unitary(c, device) for c in basis_str])

def kron_pauli_string(s, device=None):
    """Build Pauli string on the correct device."""
    if device is None:
        device = DEVICE
    return kron_all([get_pauli(c, device) for c in s])

def fidelity(rho_hat, rho_true):
    rho_hat_np = rho_hat[0].detach().cpu().numpy() if rho_hat.dim() == 3 else rho_hat.detach().cpu().numpy()
    rho_true_np = rho_true[0].detach().cpu().numpy() if rho_true.dim() == 3 else rho_true.detach().cpu().numpy()
    rho_hat_np = (rho_hat_np + rho_hat_np.conj().T) / 2
    rho_true_np = (rho_true_np + rho_true_np.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(rho_true_np)
    eigvals = np.maximum(eigvals, 0)
    sqrt_rho_true = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T
    M = sqrt_rho_true @ rho_hat_np @ sqrt_rho_true
    M = (M + M.conj().T) / 2
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.maximum(eigvals_M, 0)
    F = (np.sum(np.sqrt(eigvals_M)) ** 2).real
    return float(min(F, 1.0))

print("✓ Quantum utilities defined (mixed_ghz_state + Uhlmann fidelity)")


# === cell #4 ===
# =============================================================================
# MEASUREMENT FUNCTIONS (M1 & M2) - FIXED FOR GPU
# =============================================================================

# M2: Probability distributions
def generate_bases_M2(N, M, seed=0):
    random.seed(seed)
    bases = []
    if M >= 1: bases.append('Z'*N)
    if M >= 2: bases.append('X'*N)
    if M >= 3: bases.append('Y'*N)
    letters = ['X','Y','Z']
    seen = set(bases)
    while len(bases) < M:
        b = ''.join(random.choice(letters) for _ in range(N))
        if b not in seen:
            seen.add(b)
            bases.append(b)
    return bases

@torch.no_grad()
def measure_M2(rho, bases, eps=1e-12):
    device = rho.device  # Use rho's device
    d = rho.shape[1]
    rows = []
    for b in bases:
        U = build_rotation_unitary(b, device=device)
        rhob = U @ rho[0] @ U.conj().T
        p = rhob.diagonal().real.clamp(min=0.0)
        s = p.sum()
        p = p/s if s > 0 else torch.full((d,), 1.0/d, dtype=FDTYPE, device=device)
        rows.append(p)
    P = torch.stack(rows, dim=0)
    P = (P + eps) / (P.sum(dim=1, keepdim=True) + eps*d)
    return P.reshape(1, -1).to(FDTYPE)

# M1: Expectation values
def all_pauli_strings(N, include_I=False):
    letters = ['I','X','Y','Z'] if include_I else ['X','Y','Z']
    return [''.join(p) for p in itertools.product(letters, repeat=N)]

@torch.no_grad()
def measure_M1(rho, ops_strings):
    device = rho.device  # Use rho's device
    vals = []
    for s in ops_strings:
        U = kron_pauli_string(s, device=device)
        vals.append((rho[0] @ U).diagonal().sum().real)
    return torch.stack(vals, dim=0).reshape(1, -1).to(FDTYPE)

@torch.no_grad()
def select_ops_nonzero_M1(rho, M, N, seed=1234, tries=8, tol=1e-6):
    """Seleziona M operatori Pauli con |<O>| > tol."""
    device = rho.device  # Use rho's device
    strings_all = all_pauli_strings(N, include_I=False)
    L = len(strings_all)
    rng = np.random.default_rng(seed)
    chosen = []
    seen = set()

    for _ in range(tries):
        pool_sz = min(5*M + 32, L)
        idx = rng.choice(L, size=pool_sz, replace=False)
        for i in idx:
            s = strings_all[i]
            if s in seen:
                continue
            U = kron_pauli_string(s, device=device)
            exp_val = abs((rho[0] @ U).diagonal().sum().real.item())
            if exp_val > tol:
                chosen.append(s)
                seen.add(s)
            if len(chosen) >= M:
                break
        if len(chosen) >= M:
            break

    # Phase 2: fill remaining with zero-expectation operators (no duplicates)
    if len(chosen) < M:
        for s in strings_all:
            if s not in seen:
                chosen.append(s)
                seen.add(s)
                if len(chosen) >= M:
                    break

    # Cap at min(M, total available) to avoid duplicates when M > 3^N
    L = len(strings_all)
    return chosen[:min(M, L)]

print("✓ Measurement functions defined")


# === cell #5 ===
# =============================================================================
# MEASUREMENT LAYERS
# =============================================================================

class DensityMap(nn.Module):
    """Fixed DensityMap for GPU with float32."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.d = d
        self.eps = eps
        # Buffer verrà creato al primo forward sul device corretto
        self.register_buffer('I', None)

    def forward(self, x_realimag):
        B = x_realimag.shape[0]
        device = x_realimag.device

        # Lazy init dell'identità sul device corretto
        if self.I is None or self.I.device != device:
            self.I = torch.eye(self.d, dtype=torch.complex64, device=device)

        A = x_realimag.view(B, 2, self.d, self.d)
        # Usa torch.complex invece di 1j per migliore compatibilità
        A_complex = torch.complex(A[:,0].float(), A[:,1].float())

        rho = A_complex @ A_complex.conj().transpose(-1,-2) + self.eps * self.I
        tr = rho.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real.clamp(min=1e-12)
        return rho / tr.view(B,1,1)


class MeasurementM2(nn.Module):
    def __init__(self, bases):
        super().__init__()
        with torch.no_grad():
            Ub = [build_rotation_unitary(b) for b in bases]
        self.register_buffer('Ub', torch.stack(Ub, dim=0))
        self.register_buffer('UbH', self.Ub.conj().transpose(-1,-2))
        self.d = self.Ub.shape[-1]
        self.B = self.Ub.shape[0]

    def forward(self, rho):
        rows = []
        for i in range(self.B):
            rhob = self.Ub[i] @ rho[0] @ self.UbH[i]
            p = rhob.diagonal().real.clamp(min=0.0)
            s = p.sum()
            p = p/s if s > 0 else torch.full((self.d,), 1.0/self.d, dtype=FDTYPE, device=DEVICE)
            rows.append(p)
        P = torch.stack(rows, dim=0)
        P = (P + 1e-12) / (P.sum(dim=1, keepdim=True) + 1e-12*self.d)
        return P.reshape(1, -1).to(FDTYPE)

class MeasurementM1(nn.Module):
    def __init__(self, ops_strings):
        super().__init__()
        with torch.no_grad():
            Ulist = [kron_pauli_string(s) for s in ops_strings]
        self.register_buffer('U', torch.stack(Ulist, dim=0))
        self.K = self.U.shape[0]

    def forward(self, rho):
        vals = [(rho[0] @ self.U[k]).diagonal().sum().real for k in range(self.K)]
        return torch.stack(vals, dim=0).reshape(1, -1).to(FDTYPE)

print("✓ Measurement layers defined")


# === cell #6 ===
# =============================================================================
# SPIKING NEURON LAYERS
# =============================================================================

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        alpha = 2.0
        return grad_output * (alpha / (1 + alpha*torch.abs(x))**2)

def spike_fn(x):
    return SurrogateHeaviside.apply(x)

class LIF(nn.Module):
    def __init__(self, beta=0.95, v_th=1.0):
        super().__init__()
        self.beta = beta
        self.v_th = v_th

    def forward(self, x_seq):
        T = x_seq.shape[0]
        v = torch.zeros_like(x_seq[0])
        outs = []
        for t in range(T):
            v = self.beta * v + x_seq[t]
            s = spike_fn(v - self.v_th)
            v = v - s * self.v_th
            outs.append(s)
        return torch.stack(outs, dim=0)

def rate_code(x_vec, T=8, is_prob=True):
    x = (x_vec + 1.0)/2.0 if not is_prob else x_vec
    x = x.clamp(0, 1)
    probs = x.unsqueeze(0).repeat(T, 1, 1)
    return torch.bernoulli(probs.to(torch.float64)).to(x_vec.device).to(x_vec.dtype)

print("✓ Spiking layers defined")


# === cell #7 ===
# =============================================================================
# NORSE LIF & CROSSBAR LAYERS
# =============================================================================

class NorseLIFBase(nn.Module):
    def __init__(self, output_features, T=8, tau_mem_inv=100.0, tau_syn_inv=200.0,
                 v_th=1.0, return_rate=True, enc_mode='poisson', enc_gamma=1.5,
                 is_prob_input=True):
        super().__init__()
        self.T = T
        self.return_rate = return_rate
        self.enc_mode = enc_mode
        self.enc_gamma = enc_gamma
        self.is_prob_input = is_prob_input
        p = LIFParameters(
            tau_mem_inv=torch.tensor(tau_mem_inv),
            tau_syn_inv=torch.tensor(tau_syn_inv),
            v_th=torch.tensor(v_th)
        )
        self.lif = norse.LIFCell(p)

    def encode(self, x):
        T, B = self.T, x.size(0)
        # CRITICAL: Normalize M1 inputs from [-1,1] to [0,1]
        if not self.is_prob_input:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        if self.enc_mode == 'poisson':
            probs = torch.sigmoid(self.enc_gamma * (x - 0.5))
            probs = probs.unsqueeze(0).expand(T, -1, -1)
            return torch.bernoulli(probs)
        else:
            return x.unsqueeze(0).expand(T, -1, -1)

class NorseLIFLinear(NorseLIFBase):
    def __init__(self, in_features, out_features, is_prob_input=True, **kwargs):
        super().__init__(out_features, is_prob_input=is_prob_input, **kwargs)
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def current(self, x):
        return self.fc(x)

    def forward(self, x):
        x_enc = self.encode(x)
        T, B = x_enc.shape[0], x_enc.shape[1]
        state = None
        outs = []
        for t in range(T):
            I = self.current(x_enc[t])
            out, state = self.lif(I, state)
            outs.append(out)
        spikes = torch.stack(outs, dim=0)
        return spikes.mean(dim=0) if self.return_rate else spikes.sum(dim=0)

class CrossbarLIFLinear(NorseLIFBase):
    def __init__(self, in_features, out_features, weight_bits=8, adc_bits=8,
                 dac_bits=8, noise_std=0.01, device_variation=0.02,
                 is_prob_input=True, **kwargs):
        norse_kwargs = {k: v for k, v in kwargs.items()
                       if k in ['T', 'tau_mem_inv', 'tau_syn_inv', 'v_th',
                               'return_rate', 'enc_mode', 'enc_gamma']}
        norse_kwargs['is_prob_input'] = is_prob_input
        super().__init__(out_features, **norse_kwargs)
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.device_variation = device_variation
        self.register_buffer('variation_mask',
                           1.0 + device_variation * torch.randn(out_features, in_features))

    def quantize_weights(self, w):
        n_levels = 2 ** self.weight_bits
        w_min, w_max = w.min(), w.max()
        w_range = w_max - w_min + 1e-8
        w_norm = (w - w_min) / w_range
        w_quant = torch.round(w_norm * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant * w_range + w_min
        return w + (w_quant - w).detach()

    def adc_quantize(self, y):
        if self.adc_bits >= 32:
            return y
        n_levels = 2 ** self.adc_bits
        y_min, y_max = y.min(), y.max()
        y_range = y_max - y_min + 1e-8
        y_norm = (y - y_min) / y_range
        y_quant = torch.round(y_norm * (n_levels - 1)) / (n_levels - 1)
        y_quant = y_quant * y_range + y_min
        return y + (y_quant - y).detach()

    def current(self, x):
        w_quant = self.quantize_weights(self.fc.weight)
        w_var = w_quant * self.variation_mask
        y = F.linear(x, w_var, None)
        if self.noise_std > 0 and self.training:
            y = y + self.noise_std * torch.randn_like(y)
        return self.adc_quantize(y)

    def forward(self, x):
        x_enc = self.encode(x)
        T, B = x_enc.shape[0], x_enc.shape[1]
        state = None
        outs = []
        for t in range(T):
            I = self.current(x_enc[t])
            out, state = self.lif(I, state)
            outs.append(out)
        spikes = torch.stack(outs, dim=0)
        return spikes.mean(dim=0) if self.return_rate else spikes.sum(dim=0)

print("✓ Norse & Crossbar layers defined")


# === cell #8 ===
# =============================================================================
# GENERATOR ARCHITECTURES
# =============================================================================

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)

class CNNGenBase2D(nn.Module):
    def __init__(self, d, num_points, measurement_layer, up=False):
        super().__init__()
        self.d = d
        self.measure = measurement_layer
        dense_dim = max(d // 2, 8)
        self.dense_dim = dense_dim
        self.fc = nn.Sequential(
            nn.Linear(num_points, dense_dim*dense_dim*2, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 1, 1, bias=False),
            nn.ConvTranspose2d(32, 2, 4, 1, 1, bias=False),
        )
        self.to_rho = DensityMap(d)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        z = self.fc(x).view(B, 2, self.dense_dim, self.dense_dim)
        z = self.deconv(z)[:,:,:self.d,:self.d].reshape(B, 2, self.d, self.d)
        rho_hat = self.to_rho(z)
        y_hat = self.measure(rho_hat)
        return y_hat, rho_hat

class SCNNGenNorse2D(nn.Module):
    def __init__(self, d, num_points, measurement_layer, T=8, up=False, v_th=0.3,
                 is_prob_input=True):
        super().__init__()
        self.d = d
        self.measure = measurement_layer
        self.T = T
        self.is_prob_input = is_prob_input
        dense_dim = max(d // 2, 8)
        self.dense_dim = dense_dim
        # v_th=0.3 by default for better spike generation
        kw = dict(T=T, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=v_th,
                 return_rate=True, enc_mode='poisson', enc_gamma=1.5,
                 is_prob_input=is_prob_input)
        self.fc = NorseLIFLinear(num_points, dense_dim*dense_dim*2, **kw)
        self.ct = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 1, 1, bias=False),
            nn.ConvTranspose2d(32, 2, 4, 1, 1, bias=False),
        )
        self.to_rho = DensityMap(d)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        z = self.fc(x).view(B, 2, self.dense_dim, self.dense_dim)
        z = self.ct(z)[:,:,:self.d,:self.d].reshape(B, 2, self.d, self.d)
        rho_hat = self.to_rho(z)
        y_hat = self.measure(rho_hat)
        return y_hat, rho_hat

class SCNNGenCrossbar2D(nn.Module):
    def __init__(self, d, num_points, measurement_layer, T=8, up=False,
                 v_th=0.3, weight_bits=8, adc_bits=8, dac_bits=8,
                 is_prob_input=True):
        super().__init__()
        self.d = d
        self.measure = measurement_layer
        self.T = T
        self.is_prob_input = is_prob_input
        dense_dim = max(d // 2, 8)
        self.dense_dim = dense_dim
        kw = dict(T=T, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=v_th,
                 return_rate=True, enc_mode='poisson', enc_gamma=1.5,
                 weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                 is_prob_input=is_prob_input)
        self.fc = CrossbarLIFLinear(num_points, dense_dim*dense_dim*2, **kw)
        self.ct = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 1, 1, bias=False),
            nn.ConvTranspose2d(32, 2, 4, 1, 1, bias=False),
        )
        self.to_rho = DensityMap(d)
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        z = self.fc(x).view(B, 2, self.dense_dim, self.dense_dim)
        z = self.ct(z)[:,:,:self.d,:self.d].reshape(B, 2, self.d, self.d)
        rho_hat = self.to_rho(z)
        y_hat = self.measure(rho_hat)
        return y_hat, rho_hat

print("✓ Generator architectures defined")


# === cell #9 ===
# =============================================================================
# DISCRIMINATOR ARCHITECTURES
# =============================================================================

class DiscClassic(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        input_dim = 2 * num_points
        hidden = max(128, input_dim * 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden, hidden//2), nn.LeakyReLU(0.2, True),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, x_cond, y):
        return self.net(torch.cat([x_cond, y], dim=-1))

class DiscSpiking(nn.Module):
    def __init__(self, num_points, T=8, v_th=0.3):
        super().__init__()
        self.T = T
        input_dim = 2 * num_points
        hidden = max(128, input_dim * 2)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.lif1 = LIF(0.95, v_th)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.lif2 = LIF(0.95, v_th)
        self.out = nn.Linear(hidden//2, 1)

    def forward(self, x_cond, y):
        x = torch.cat([x_cond, y], dim=-1)
        x_seq = rate_code(x, T=self.T, is_prob=True).to(self.fc1.weight.dtype)
        T, B, F = x_seq.shape
        v1 = v2 = None
        outs = []
        for t in range(T):
            h1 = self.fc1(x_seq[t])
            if v1 is None: v1 = torch.zeros_like(h1)
            v1 = self.lif1.beta * v1 + h1
            s1 = spike_fn(v1 - self.lif1.v_th)
            v1 = v1 - s1 * self.lif1.v_th
            h2 = self.fc2(s1)
            if v2 is None: v2 = torch.zeros_like(h2)
            v2 = self.lif2.beta * v2 + h2
            s2 = spike_fn(v2 - self.lif2.v_th)
            v2 = v2 - s2 * self.lif2.v_th
            outs.append(s2)
        return self.out(torch.stack(outs, dim=0).mean(dim=0))

print("✓ Discriminator architectures defined")


# === cell #10 ===
# =============================================================================
# TRAINING LOOP
# =============================================================================

bce = nn.BCEWithLogitsLoss()

def generator_loss(disc_fake_logits, gen_out, target, lam=10.0):
    gan = bce(disc_fake_logits, torch.ones_like(disc_fake_logits))
    l1 = F.l1_loss(gen_out, target)
    return gan + lam*l1, gan, l1

def discriminator_loss(disc_real_logits, disc_fake_logits):
    real = bce(disc_real_logits, torch.ones_like(disc_real_logits))
    fake = bce(disc_fake_logits, torch.zeros_like(disc_fake_logits))
    return real + fake

@torch.no_grad()
def map_to_prob(x, is_M2=True):
    if is_M2:
        return x.clamp(0, 1)
    else:
        return ((x + 1.0) / 2.0).clamp(0, 1)

def train_scgan(G, D, x_target, rho_true, steps=1000, lr=1e-3, lam=50.0,
                verbose_every=100, is_M2=True, use_amp=False):
    """Training loop with optional AMP and memory management."""
    G = G.to(DEVICE).float()  # float32 invece di double
    D = D.to(DEVICE).float()
    x_target = x_target.to(DEVICE, dtype=torch.float32)
    rho_true = rho_true.to(DEVICE)

    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
    schedulerG = torch.optim.lr_scheduler.LambdaLR(
        optG, lr_lambda=lambda t: 1.0 / (1.0 + 0.96 * t / max(steps, 1))
    )
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # AMP setup (opzionale)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    f_hist = []
    t0 = time.time()

    for t in range(1, steps+1):
        # Generator step
        if use_amp:
            with torch.amp.autocast('cuda'):
                y_hat, rho_hat = G(x_target)
                xf = map_to_prob(x_target, is_M2)
                yf = map_to_prob(y_hat.detach(), is_M2)
                d_fake = D(xf, yf)
                g_total, g_gan, g_l1 = generator_loss(d_fake, y_hat, x_target, lam=lam)
            optG.zero_grad(set_to_none=True)
            scaler.scale(g_total).backward()
            scaler.step(optG)
            scaler.update()
            schedulerG.step()
        else:
            y_hat, rho_hat = G(x_target)
            xf = map_to_prob(x_target, is_M2)
            yf = map_to_prob(y_hat.detach(), is_M2)
            d_fake = D(xf, yf)
            g_total, g_gan, g_l1 = generator_loss(d_fake, y_hat, x_target, lam=lam)
            optG.zero_grad(set_to_none=True)
            g_total.backward()
            optG.step()
            schedulerG.step()

        # Discriminator step
        with torch.no_grad():
            y_hat_det, _ = G(x_target)
        y_fake = map_to_prob(y_hat_det, is_M2)
        y_real = map_to_prob(x_target, is_M2)

        if use_amp:
            with torch.amp.autocast('cuda'):
                d_fake = D(xf, y_fake)
                d_real = D(xf, y_real)
                d_total = discriminator_loss(d_real, d_fake)
            optD.zero_grad(set_to_none=True)
            scaler.scale(d_total).backward()
            scaler.step(optD)
            scaler.update()
        else:
            d_fake = D(xf, y_fake)
            d_real = D(xf, y_real)
            d_total = discriminator_loss(d_real, d_fake)
            optD.zero_grad(set_to_none=True)
            d_total.backward()
            optD.step()

        # Fidelity tracking
        with torch.no_grad():
            _, rho_final = G(x_target)
            Fval = fidelity(rho_final, rho_true)
            f_hist.append(Fval)

        if (t % verbose_every == 0) or (t == steps):
            print(f"[{t:04d}/{steps}] G={g_total.item():.4e} D={d_total.item():.4e} F={Fval:.4f}")

    elapsed = time.time() - t0

    # ── Cleanup: free optimizer states, scaler, scheduler from GPU ──
    del optG, optD, schedulerG
    if scaler is not None:
        del scaler
    # Move models to CPU so caller's `del G, D` actually frees GPU memory
    G.cpu()
    D.cpu()
    del x_target, rho_true  # local refs only
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'F_mean': float(np.mean(f_hist[-50:])),
        'F_best': max(f_hist),
        'f_hist': f_hist,
        'time_sec': elapsed
    }


print("✓ Training loop defined")


# === cell #11 ===
# =============================================================================
# BENCHMARK FUNCTION
# =============================================================================

def gpu_cleanup():
    """Aggressive GPU memory cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_scgan_benchmark(N_list=[3,4,5], method='M2', M_M2=4, M_M1=256, steps=1000, lr=1e-3,
                        T=8, sparsity=0.1, crossbar_bits=[8, 4]):
    print("\n" + "="*80)
    print("🚀 SCGAN BENCHMARK: Mixed GHZ States (Werner p=0.5)")
    print("="*80)

    method = method.upper()
    is_M2 = (method == 'M2')

    results = {
        'method': method,
        'N_list': N_list,
        'T': T,
        'steps': steps,
        'data': []
    }

    for N in N_list:
        d = 2**N
        print(f"\n{'─'*80}")
        print(f"🔬 N={N} qubits (d={d})")
        print(f"{'─'*80}")

        # Setup target state: Werner state rho = p|GHZ><GHZ| + (1-p)I/d
        rho_true = mixed_ghz_state(N, p=0.5)

        # Fixed M from sweep studies
        if is_M2:
            M_eff = M_M2
            bases = generate_bases_M2(N, M_eff)
            x_target = measure_M2(rho_true, bases)
            measure = MeasurementM2(bases)
        else:
            M_eff = min(4**N - 1, M_M1)  # Cap at available Paulis
            ops = select_ops_nonzero_M1(rho_true, M_eff, N, seed=1234)
            x_target = measure_M1(rho_true, ops)
            measure = MeasurementM1(ops)

        num_points = x_target.shape[-1]
        print(f"  {method}: M={M_eff} "
              f"({'bases' if is_M2 else 'Pauli ops'}, {num_points} conditioning values)")
        current_T = T

        # 1. CGAN
        print(f"\n1️⃣  CGAN (GPU)")
        G = CNNGenBase2D(d, num_points, measure)
        D = DiscClassic(num_points)
        res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr, is_M2=is_M2)
        E_inf = estimate_gpu_inference_energy(G)
        E_train = estimate_gpu_training_energy(G, steps, res['time_sec'])
        print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.2f}µJ, E_train={E_train['E_total_mJ']:.1f}mJ")
        results['data'].append({
            'N': N, 'd': d, 'Category': 'CGAN',
            'F_mean': res['F_mean'], 'F_best': res['F_best'],
            'E_inference_uJ': E_inf['E_total_uJ'],
            'E_training_mJ': E_train['E_total_mJ'],
            'time_sec': res['time_sec'], 'HW': 'GPU'
        })
        del G, D, res
        gpu_cleanup()

        # 2. SCGAN-Norse (v_th=0.3 for better training)
        print(f"\n2️⃣  SCGAN-Norse (Loihi)")
        G = SCNNGenNorse2D(d, num_points, measure, T=current_T, v_th=0.3, is_prob_input=is_M2)
        D = DiscSpiking(num_points, T=current_T, v_th=0.3)
        res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr, is_M2=is_M2)
        E_inf = estimate_loihi_inference_energy(G, T=current_T, sparsity=sparsity)
        E_train = estimate_gpu_training_energy(G, steps, res['time_sec'])
        print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ, E_train={E_train['E_total_mJ']:.1f}mJ")
        results['data'].append({
            'N': N, 'd': d, 'Category': 'SCGAN-Norse',
            'F_mean': res['F_mean'], 'F_best': res['F_best'],
            'E_inference_uJ': E_inf['E_total_uJ'],
            'E_training_mJ': E_train['E_total_mJ'],
            'time_sec': res['time_sec'], 'HW': 'Loihi'
        })
        del G, D, res
        gpu_cleanup()

        # 3. SCGAN-Crossbar
        for bits in crossbar_bits:
            print(f"\n3️⃣  SCGAN-Crossbar-{bits}b (Memristor)")
            v_th = 0.2 if bits <= 4 else 0.3
            G = SCNNGenCrossbar2D(d, num_points, measure, T=current_T, v_th=v_th,
                                  weight_bits=bits, adc_bits=bits, dac_bits=bits,
                                  is_prob_input=is_M2)
            D = DiscSpiking(num_points, T=current_T, v_th=0.3)
            res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr, is_M2=is_M2)
            E_inf = estimate_crossbar_inference_energy(G, T=current_T, sparsity=sparsity, bits=bits)
            E_train = estimate_gpu_training_energy(G, steps, res['time_sec'])
            print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ, E_train={E_train['E_total_mJ']:.1f}mJ")
            results['data'].append({
                'N': N, 'd': d, 'Category': f'SCGAN-Crossbar-{bits}b',
                'F_mean': res['F_mean'], 'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'time_sec': res['time_sec'], 'HW': f'Crossbar-{bits}b'
            })
            del G, D, res
            gpu_cleanup()

        # Cleanup measurement data for this N
        if is_M2:
            del bases
        else:
            del ops
        del x_target, measure, rho_true
        gpu_cleanup()

    return results


def run_scgan_benchmark_gpu(N_list=[3,4,5,6,7,8], method='M2', M_M2=4, M_M1=256,
                            steps=1000, lr=1e-3, T=8, sparsity=0.1,
                            crossbar_bits=[8, 4], use_amp=False):
    """GPU-optimized benchmark with memory management."""
    import gc

    print("\n" + "="*80)
    print("🚀 SCGAN BENCHMARK: Mixed GHZ States (GPU OPTIMIZED)")
    print("="*80)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    method = method.upper()
    is_M2 = (method == 'M2')

    results = {
        'method': method,
        'N_list': N_list,
        'T': T,
        'steps': steps,
        'data': []
    }

    for N in N_list:
        d = 2**N
        print(f"\n{'─'*80}")
        print(f"🔬 N={N} qubits (d={d})")
        print(f"{'─'*80}")

        # Memory check
        if torch.cuda.is_available():
            print(f"  GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

        # Fair benchmark: uniform steps for all N
        current_steps = steps
        # T sweep finding: T>16 hurts fidelity, T=8 is the sweet spot
        current_T = T
        current_lr = lr

        # Setup target state: Werner state rho = p|GHZ><GHZ| + (1-p)I/d
        rho_true = mixed_ghz_state(N, p=0.5)

        # Fixed M from sweep studies
        if is_M2:
            M_eff = M_M2
            bases = generate_bases_M2(N, M_eff)
            x_target = measure_M2(rho_true, bases)
            measure = MeasurementM2(bases)
        else:
            M_eff = min(4**N - 1, M_M1)  # Cap at available Paulis
            ops = select_ops_nonzero_M1(rho_true, M_eff, N, seed=1234)
            x_target = measure_M1(rho_true, ops)
            measure = MeasurementM1(ops)

        num_points = x_target.shape[-1]
        print(f"  {method}: M={M_eff} "
              f"({'bases' if is_M2 else 'Pauli ops'}, {num_points} conditioning values)")

        # 1. CGAN
        print(f"\n1️⃣  CGAN (GPU)")
        G = CNNGenBase2D(d, num_points, measure)
        D = DiscClassic(num_points)
        res = train_scgan(G, D, x_target, rho_true, steps=current_steps,
                         lr=current_lr, is_M2=is_M2, use_amp=use_amp)
        E_inf = estimate_gpu_inference_energy(G)
        E_train = estimate_gpu_training_energy(G, current_steps, res['time_sec'])
        print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.2f}µJ")
        results['data'].append({
            'N': N, 'd': d, 'Category': 'CGAN',
            'F_mean': res['F_mean'], 'F_best': res['F_best'],
            'E_inference_uJ': E_inf['E_total_uJ'],
            'E_training_mJ': E_train['E_total_mJ'],
            'time_sec': res['time_sec'], 'HW': 'GPU'
        })

        del G, D, res
        gpu_cleanup()

        # 2. SCGAN-Norse
        print(f"\n2️⃣  SCGAN-Norse (Loihi)")
        G = SCNNGenNorse2D(d, num_points, measure, T=current_T, v_th=0.3, is_prob_input=is_M2)
        D = DiscSpiking(num_points, T=current_T, v_th=0.3)
        res = train_scgan(G, D, x_target, rho_true, steps=current_steps,
                         lr=current_lr, is_M2=is_M2, use_amp=use_amp)
        E_inf = estimate_loihi_inference_energy(G, T=current_T, sparsity=sparsity)
        E_train = estimate_gpu_training_energy(G, current_steps, res['time_sec'])
        print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ")
        results['data'].append({
            'N': N, 'd': d, 'Category': 'SCGAN-Norse',
            'F_mean': res['F_mean'], 'F_best': res['F_best'],
            'E_inference_uJ': E_inf['E_total_uJ'],
            'E_training_mJ': E_train['E_total_mJ'],
            'time_sec': res['time_sec'], 'HW': 'Loihi'
        })

        del G, D, res
        gpu_cleanup()

        # 3. SCGAN-Crossbar
        for bits in crossbar_bits:
            print(f"\n3️⃣  SCGAN-Crossbar-{bits}b (Memristor)")
            v_th = 0.2 if bits <= 4 else 0.3
            G = SCNNGenCrossbar2D(d, num_points, measure, T=current_T, v_th=v_th,
                                  weight_bits=bits, adc_bits=bits, dac_bits=bits,
                                  is_prob_input=is_M2)
            D = DiscSpiking(num_points, T=current_T, v_th=0.3)
            res = train_scgan(G, D, x_target, rho_true, steps=current_steps,
                             lr=current_lr, is_M2=is_M2, use_amp=use_amp)
            E_inf = estimate_crossbar_inference_energy(G, T=current_T, sparsity=sparsity, bits=bits)
            E_train = estimate_gpu_training_energy(G, current_steps, res['time_sec'])
            print(f"  F={res['F_mean']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ")
            results['data'].append({
                'N': N, 'd': d, 'Category': f'SCGAN-Crossbar-{bits}b',
                'F_mean': res['F_mean'], 'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'time_sec': res['time_sec'], 'HW': f'Crossbar-{bits}b'
            })

            del G, D, res
            gpu_cleanup()

        # Cleanup measurement data for this N
        if is_M2:
            del bases
        else:
            del ops
        del x_target, measure, rho_true
        gpu_cleanup()
        print(f"\n  ✓ N={N} complete. GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return results

print("✓ Benchmark function defined")


# === cell #12 ===
# =============================================================================
# ANALYSIS FUNCTION (FIXED)
# =============================================================================

def analyze_scgan_results(results):
    """Analizza risultati benchmark - FIXED version."""
    print("\n" + "="*80)
    print("📊 SCGAN BENCHMARK RESULTS")
    print("="*80)

    # Create DataFrame from data list
    df = pd.DataFrame(results['data'])

    for N in sorted(df['N'].unique()):
        df_N = df[df['N'] == N].copy()
        baseline_E = df_N[df_N['Category'] == 'CGAN']['E_inference_uJ'].iloc[0]
        baseline_E_train = df_N[df_N['Category'] == 'CGAN']['E_training_mJ'].iloc[0]

        df_N['Speedup'] = baseline_E / df_N['E_inference_uJ']
        df_N['Breakeven'] = df_N.apply(
            lambda row: (row['E_training_mJ'] - baseline_E_train) * 1e3 / (baseline_E - row['E_inference_uJ'])
            if (baseline_E - row['E_inference_uJ']) > 0 else float('inf'), axis=1
        )

        print(f"\n{'─'*80}")
        print(f"N = {N} qubits")
        print(f"{'─'*80}")
        print(f"\n{'Category':<25} {'F_mean':<10} {'E_inf(µJ)':<12} {'Speedup':<10} {'E_train(mJ)':<12} {'Breakeven'}")
        print("-"*90)

        for _, row in df_N.iterrows():
            be_str = f"{row['Breakeven']/1e6:.1f}M" if row['Breakeven'] < 1e12 else "N/A"
            print(f"{row['Category']:<25} {row['F_mean']:.4f}     {row['E_inference_uJ']:<12.4f} "
                  f"{row['Speedup']:<10.0f}× {row['E_training_mJ']:<12.1f} {be_str}")

    print(f"\n{'='*80}")
    print("📈 SUMMARY")
    print(f"{'='*80}")

    for cat in df['Category'].unique():
        cat_df = df[df['Category'] == cat]
        avg_f = cat_df['F_mean'].mean()
        avg_e = cat_df['E_inference_uJ'].mean()
        baseline_e = df[df['Category'] == 'CGAN']['E_inference_uJ'].mean()
        speedup = baseline_e / avg_e
        print(f"  {cat:<25}: F={avg_f:.4f}, E_inf={avg_e:.4f}µJ, Speedup={speedup:.0f}×")

    return df

print("✓ Analysis function defined")


# === cell #13 ===
# =============================================================================
# RUN M2 BENCHMARK
# =============================================================================

print("\n" + "="*80)
print("🚀 STARTING SCGAN BENCHMARK — Mixed GHZ States (M2)")
print("="*80)

if _seed_done('M2'):
    print(f"[multiseed] seed={SEED} method=M2 already done — skipping", flush=True)
    results_M2 = None
else:
    if _ARGS.quick:
        results_M2 = run_scgan_benchmark_gpu(
            N_list=[3],
            method='M2',
            M_M2=4,         # M-sweep: 4 bases sufficient
            steps=3,
            lr=1e-3,
            T=8,
            sparsity=0.1,
            crossbar_bits=[8, 4]
        )

    else:
        results_M2 = run_scgan_benchmark_gpu(
            N_list=[3, 4, 5, 6, 7, 8],
            method='M2',
            M_M2=4,         # M-sweep: 4 bases sufficient
            steps=1000,
            lr=1e-3,
            T=8,
            sparsity=0.1,
            crossbar_bits=[8, 4]
        )

    _save_with_seed(results_M2, 'M2', analyze_scgan_results)
df_M2 = analyze_scgan_results(results_M2)


# === cell #14 ===
# =============================================================================
# RUN M1 BENCHMARK — Mixed GHZ States
# =============================================================================

print("\n" + "="*80)
print("🚀 STARTING SCGAN BENCHMARK — Mixed GHZ States (M1)")
print("="*80)

if _seed_done('M1'):
    print(f"[multiseed] seed={SEED} method=M1 already done — skipping", flush=True)
    results_M1 = None
else:
    if _ARGS.quick:
        results_M1 = run_scgan_benchmark_gpu(
            N_list=[3],
            method='M1',
            M_M1=256,       # M-sweep: 256 Paulis
            steps=3,
            lr=1e-3,
            T=8,
            sparsity=0.1,
            crossbar_bits=[8, 4]
        )

    else:
        results_M1 = run_scgan_benchmark_gpu(
            N_list=[3, 4, 5, 6, 7, 8],
            method='M1',
            M_M1=256,       # M-sweep: 256 Paulis
            steps=1000,
            lr=1e-3,
            T=8,
            sparsity=0.1,
            crossbar_bits=[8, 4]
        )

    _save_with_seed(results_M1, 'M1', analyze_scgan_results)
df_M1 = analyze_scgan_results(results_M1)


# === cell #15 ===
# =============================================================================
# CELLA DI PLOTTING AGGIORNATA PER SCGAN
# =============================================================================
# Stile uniforme con i plot SCNN, include:
# - Dashboard completa (2x3)
# - Infidelity plot (1-F) in scala logaritmica
# - Confronto Hardware
# - Scaling Analysis
# - Summary Statistics
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

# Colori consistenti (stile SCNN)
COLORS = {
    'CGAN': '#1f77b4',               # Blu (baseline GPU)
    'SCGAN-Norse': '#2ca02c',         # Verde (Loihi)
    'SCGAN-Crossbar-8b': '#ff7f0e',   # Arancione
    'SCGAN-Crossbar-4b': '#d62728',   # Rosso
}

MARKERS = {
    'CGAN': 'o',
    'SCGAN-Norse': '^',
    'SCGAN-Crossbar-8b': 'D',
    'SCGAN-Crossbar-4b': 'p',
}

def plot_scgan_complete(df, method_name='M2', save_prefix='scgan'):
    """
    Genera plot completi dei risultati SCGAN benchmark.
    Stile uniforme con i plot SCNN.
    """

    N_values = sorted(df['N'].unique())
    categories = df['Category'].unique()

    # Organizza dati per categoria
    data_by_cat = {}
    for cat in categories:
        cat_data = df[df['Category'] == cat].sort_values('N')
        data_by_cat[cat] = {
            'N': cat_data['N'].tolist(),
            'F': cat_data['F_mean'].tolist(),
            'E_inf': cat_data['E_inference_uJ'].tolist(),
            'E_train': cat_data['E_training_mJ'].tolist(),
        }

    # =========================================================================
    # FIGURA 1: Dashboard Completa (2x3)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'SCGAN Quantum State Tomography Benchmark - Method {method_name}',
                 fontsize=14, fontweight='bold')

    # 1. Fidelity vs N
    ax = axes[0, 0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        ax.plot(data['N'], data['F'], marker=marker, color=color,
                label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity vs System Size', fontsize=12)
    ax.set_ylim([0.9, 1.005])
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='F=0.99')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)

    # 2. Infidelity vs N (NUOVO!)
    ax = axes[0, 1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        infidelity = [max(1 - f, 1e-6) for f in data['F']]
        ax.semilogy(data['N'], infidelity, marker=marker, color=color,
                   label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity vs System Size (lower is better)', fontsize=12)
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.001, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(max(N_values)+0.1, 0.01, 'F=0.99', fontsize=8, color='orange')
    ax.text(max(N_values)+0.1, 0.001, 'F=0.999', fontsize=8, color='green')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim([1e-4, 0.2])

    # 3. Inference Energy vs N
    ax = axes[0, 2]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        ax.semilogy(data['N'], data['E_inf'], marker=marker, color=color,
                   label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Inference Energy (µJ)', fontsize=11)
    ax.set_title('Inference Energy vs System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # 4. Speedup vs N
    ax = axes[1, 0]
    baseline_E = {n: e for n, e in zip(data_by_cat['CGAN']['N'], data_by_cat['CGAN']['E_inf'])}

    for cat, data in data_by_cat.items():
        if cat != 'CGAN':
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            speedups = []
            N_valid = []
            for n, e in zip(data['N'], data['E_inf']):
                if n in baseline_E and e > 0:
                    speedups.append(baseline_E[n] / e)
                    N_valid.append(n)
            if speedups:
                ax.semilogy(N_valid, speedups, marker=marker, color=color,
                           label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Speedup vs CGAN-GPU', fontsize=11)
    ax.set_title('Energy Efficiency Gain', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # 5. Training Energy vs N
    ax = axes[1, 1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        E_train_J = [e / 1000 for e in data['E_train']]  # mJ → J
        ax.semilogy(data['N'], E_train_J, marker=marker, color=color,
                   label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Training Energy (J)', fontsize=11)
    ax.set_title('Training Energy vs System Size (GPU)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # 6. Efficiency (F/E) vs N
    ax = axes[1, 2]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        efficiency = [f / e if e > 0 else 0 for f, e in zip(data['F'], data['E_inf'])]
        ax.semilogy(data['N'], efficiency, marker=marker, color=color,
                   label=cat, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Efficiency (Fidelity / µJ)', fontsize=11)
    ax.set_title('Energy Efficiency vs System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{method_name}_dashboard.png', dpi=150, bbox_inches='tight')

    # =========================================================================
    # FIGURA 2: Infidelity Analysis (dettagliata)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Infidelity Analysis (1-F) - SCGAN {method_name}', fontsize=13, fontweight='bold')

    # Left: Infidelity vs N (line plot)
    ax = axes[0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        infidelity = [max(1 - f, 1e-6) for f in data['F']]
        ax.semilogy(data['N'], infidelity, marker=marker, color=color,
                   label=cat, linewidth=2, markersize=8)

    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity vs System Size (lower is better)', fontsize=12)

    # Reference lines con zone colorate
    ax.axhspan(0.1, 1.0, alpha=0.1, color='red', label='F < 0.9')
    ax.axhspan(0.01, 0.1, alpha=0.1, color='orange')
    ax.axhspan(0.001, 0.01, alpha=0.1, color='yellow')
    ax.axhspan(1e-6, 0.001, alpha=0.1, color='green')

    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.001, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    ax.text(max(N_values)+0.15, 0.1, 'F=0.9', fontsize=8, color='red')
    ax.text(max(N_values)+0.15, 0.01, 'F=0.99', fontsize=8, color='orange')
    ax.text(max(N_values)+0.15, 0.001, 'F=0.999', fontsize=8, color='green')

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim([1e-4, 0.2])
    ax.set_xlim([min(N_values)-0.3, max(N_values)+0.8])

    # Right: Infidelity bar chart per N selezionati
    ax = axes[1]
    N_selected = [n for n in N_values if n in [3, 4, 5, 6, 7, 8]][:4]

    x = np.arange(len(N_selected))
    width = 0.18
    cats_to_plot = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']

    for i, cat in enumerate(cats_to_plot):
        if cat in data_by_cat:
            infidelities = []
            for N in N_selected:
                if N in data_by_cat[cat]['N']:
                    idx = data_by_cat[cat]['N'].index(N)
                    f = data_by_cat[cat]['F'][idx]
                    infidelities.append(max(1 - f, 1e-6))
                else:
                    infidelities.append(np.nan)

            color = COLORS.get(cat, 'gray')
            offset = (i - len(cats_to_plot)/2 + 0.5) * width
            short_name = cat.replace('SCGAN-', '').replace('Crossbar-', 'CB-')
            ax.bar(x + offset, infidelities, width, label=short_name,
                  color=color, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity Comparison (lower is better)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in N_selected])
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='F=0.99')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both', axis='y')
    ax.set_ylim([1e-4, 0.2])

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{method_name}_infidelity.png', dpi=150, bbox_inches='tight')

    # =========================================================================
    # FIGURA 3: Hardware Comparison (Bar chart per N)
    # =========================================================================
    N_to_plot = [n for n in N_values if n in [3, 5, 6, 7, 8]][:4]

    if len(N_to_plot) >= 2:
        fig, axes = plt.subplots(1, len(N_to_plot), figsize=(4*len(N_to_plot), 5))
        if len(N_to_plot) == 1:
            axes = [axes]

        fig.suptitle(f'Inference Energy by Hardware Platform - SCGAN {method_name}',
                    fontsize=13, fontweight='bold')

        hw_categories = ['CGAN\n(GPU)', 'Norse\n(Loihi)', 'Crossbar\n8-bit', 'Crossbar\n4-bit']
        cat_keys = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']
        cat_colors = [COLORS.get(k, 'gray') for k in cat_keys]

        for ax, N in zip(axes, N_to_plot):
            energies = []
            fidelities = []

            for cat in cat_keys:
                if cat in data_by_cat and N in data_by_cat[cat]['N']:
                    idx = data_by_cat[cat]['N'].index(N)
                    energies.append(data_by_cat[cat]['E_inf'][idx])
                    fidelities.append(data_by_cat[cat]['F'][idx])
                else:
                    energies.append(0)
                    fidelities.append(0)

            bars = ax.bar(hw_categories, energies, color=cat_colors)
            ax.set_ylabel('Energy (µJ)' if N == N_to_plot[0] else '')
            ax.set_title(f'N = {N} qubits\n(d = {2**N})', fontsize=11)
            ax.set_yscale('log')

            # Annota con fidelity
            for bar, f in zip(bars, fidelities):
                height = bar.get_height()
                if height > 0:
                    color = 'red' if f < 0.99 else 'black'
                    fontweight = 'bold' if f < 0.99 else 'normal'
                    ax.annotate(f'F={f:.3f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=7, rotation=0,
                               color=color, fontweight=fontweight)

            # Speedup rispetto a GPU
            if energies[0] > 0:
                for i, (bar, e) in enumerate(zip(bars[1:], energies[1:]), 1):
                    if e > 0:
                        speedup = energies[0] / e
                        ax.annotate(f'{speedup:.0f}×',
                                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()/3),
                                   ha='center', va='center', fontsize=9,
                                   fontweight='bold', color='white')

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{method_name}_hardware_comparison.png', dpi=150, bbox_inches='tight')

    # =========================================================================
    # FIGURA 4: Scaling Analysis
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Scaling Analysis - SCGAN {method_name}', fontsize=13, fontweight='bold')

    # Left: Energy scaling (log-log)
    ax = axes[0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        d_values = [2**n for n in data['N']]
        ax.loglog(d_values, data['E_inf'], marker=marker, color=color,
                 label=cat, linewidth=2, markersize=8)

    # Reference lines
    d_ref = np.array([8, 256])
    ax.loglog(d_ref, 0.1 * (d_ref/8)**2, 'k--', alpha=0.3, label='O(d²)')
    ax.loglog(d_ref, 0.1 * (d_ref/8)**3, 'k:', alpha=0.3, label='O(d³)')

    ax.set_xlabel('Hilbert space dimension d', fontsize=11)
    ax.set_ylabel('Inference Energy (µJ)', fontsize=11)
    ax.set_title('Energy Scaling with System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # Right: Fidelity vs Energy Trade-off
    ax = axes[1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        # Size proporzionale a N
        sizes = [30 + 20*n for n in data['N']]
        ax.scatter(data['E_inf'], data['F'], s=sizes, c=color,
                  marker=marker, label=cat, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Inference Energy (µJ)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity vs Energy Trade-off', fontsize=12)
    ax.set_xscale('log')
    ax.axhline(y=0.99, color='orange', linestyle='--', alpha=0.5, label='F=0.99')
    ax.axhline(y=0.999, color='green', linestyle=':', alpha=0.5, label='F=0.999')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim([0.98, 1.002])

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{method_name}_scaling.png', dpi=150, bbox_inches='tight')

    # =========================================================================
    # Print Summary Statistics
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"📈 SUMMARY STATISTICS - SCGAN {method_name}")
    print(f"{'='*70}")

    # Problem analysis
    print(f"\n⚠️  PROBLEM ANALYSIS:")
    print(f"   Models with F < 0.99:")
    has_issues = False
    for cat, data in data_by_cat.items():
        issues = [(n, f) for n, f in zip(data['N'], data['F']) if f < 0.99]
        if issues:
            has_issues = True
            for n, f in issues:
                print(f"      ⚠️  {cat} @ N={n}: F={f:.4f}")
    if not has_issues:
        print(f"      ✅ All models have F ≥ 0.99!")

    # Best performers at max N
    max_N = max(N_values)
    print(f"\n🏆 Performance at N={max_N} (d={2**max_N}):")

    best_data = []
    for cat, data in data_by_cat.items():
        if max_N in data['N']:
            idx = data['N'].index(max_N)
            F = data['F'][idx]
            E = data['E_inf'][idx]
            best_data.append((cat, F, E))

    print("\n  By Fidelity:")
    for cat, F, E in sorted(best_data, key=lambda x: -x[1]):
        print(f"    {cat}: F={F:.4f}, E={E:.2f}µJ")

    print("\n  By Energy Efficiency (F/E):")
    for cat, F, E in sorted(best_data, key=lambda x: -x[1]/x[2] if x[2] > 0 else 0):
        eff = F/E if E > 0 else 0
        print(f"    {cat}: {eff:.2f} F/µJ (F={F:.4f}, E={E:.2f}µJ)")

    # Average speedups
    print("\n  Average Speedup vs CGAN-GPU (all N):")
    baseline_E_avg = np.mean(data_by_cat['CGAN']['E_inf'])
    for cat in ['SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']:
        if cat in data_by_cat:
            speedups = []
            for n, e in zip(data_by_cat[cat]['N'], data_by_cat[cat]['E_inf']):
                if n in baseline_E and e > 0:
                    speedups.append(baseline_E[n] / e)
            if speedups:
                print(f"    {cat}: {np.mean(speedups):.0f}× (range: {min(speedups):.0f}-{max(speedups):.0f}×)")

    print(f"\n{'='*70}")
    print("✅ All plots saved!")
    print(f"{'='*70}")

    return data_by_cat


# === cell #16 ===
# =============================================================================
# ESEGUI I PLOT
# =============================================================================

# Plot M1 results (usa df_M1 generato da analyze_scgan_results)
data_organized = plot_scgan_complete(df_M1, method_name='M1', save_prefix='scgan')


# === cell #17 ===
# =============================================================================
# ESEGUI I PLOT
# =============================================================================

# Plot M1 results (usa df_M2 generato da analyze_scgan_results)
data_organized = plot_scgan_complete(df_M1, method_name='M1', save_prefix='scgan')


# === cell #18 ===
# =============================================================================
# CELLA DI PLOTTING AGGIORNATA PER SCGAN
# =============================================================================
# Stile uniforme con i plot SCNN, include:
# - Dashboard completa (2x3)
# - Infidelity plot (1-F) in scala logaritmica
# - Confronto Hardware (con fidelity drop)
# - Scaling Analysis
# - Confronto M1 vs M2 (istogrammi)
# - Summary Statistics
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# MAPPING LABELS
# =============================================================================

# Labels per grafici con tutte le metriche (fidelity, energy, etc.)
LABELS_ALL = {
    'CGAN': 'CGAN (GPU)',
    'SCGAN-Norse': 'SCGAN (Loihi)',
    'SCGAN-Crossbar-8b': 'SCGAN (Crossbar 8-bit)',
    'SCGAN-Crossbar-4b': 'SCGAN (Crossbar 4-bit)',
}

# Labels per grafici confronto hardware (istogrammi)
LABELS_HARDWARE = {
    'CGAN': 'CGAN (GPU)',
    'SCGAN-Norse': 'SCGAN (Loihi)',
    'SCGAN-Crossbar-8b': 'SCGAN (8-bit)',
    'SCGAN-Crossbar-4b': 'SCGAN (4-bit)',
}

# Colori consistenti
COLORS = {
    'CGAN': '#1f77b4',               # Blu (baseline GPU)
    'SCGAN-Norse': '#2ca02c',         # Verde (Loihi)
    'SCGAN-Crossbar-8b': '#ff7f0e',   # Arancione
    'SCGAN-Crossbar-4b': '#d62728',   # Rosso
}

MARKERS = {
    'CGAN': 'o',
    'SCGAN-Norse': '^',
    'SCGAN-Crossbar-8b': 'D',
    'SCGAN-Crossbar-4b': 'p',
}


def get_label(cat_key, use_hardware_labels=False):
    """Restituisce la label corretta per la categoria."""
    if use_hardware_labels:
        return LABELS_HARDWARE.get(cat_key, cat_key)
    return LABELS_ALL.get(cat_key, cat_key)


def collect_data_by_category(df):
    """Raccoglie i dati per categoria da un DataFrame."""
    categories = df['Category'].unique()
    data_by_cat = {}
    for cat in categories:
        cat_data = df[df['Category'] == cat].sort_values('N')
        data_by_cat[cat] = {
            'N': cat_data['N'].tolist(),
            'F': cat_data['F_mean'].tolist(),
            'E_inf': cat_data['E_inference_uJ'].tolist(),
            'E_train': cat_data['E_training_mJ'].tolist(),
        }
    return data_by_cat


def compute_fidelity_drop(data_by_cat, baseline_cat='CGAN'):
    """
    Calcola il fidelity drop rispetto alla baseline CGAN.
    Fidelity drop = (F_CGAN - F_SCGAN) / F_CGAN * 100
    """
    fidelity_drop = {}

    if baseline_cat not in data_by_cat:
        print(f"⚠️ Baseline {baseline_cat} not found!")
        return fidelity_drop

    # Crea lookup per baseline
    baseline_F = {}
    for n, f in zip(data_by_cat[baseline_cat]['N'], data_by_cat[baseline_cat]['F']):
        baseline_F[int(n)] = float(f)

    for cat, data in data_by_cat.items():
        if cat == baseline_cat:
            continue

        fidelity_drop[cat] = {'N': [], 'drop_pct': []}

        for n, f in zip(data['N'], data['F']):
            n_int = int(n)
            if n_int in baseline_F:
                drop_pct = (baseline_F[n_int] - f) / baseline_F[n_int] * 100 if baseline_F[n_int] > 0 else 0
                fidelity_drop[cat]['N'].append(n_int)
                fidelity_drop[cat]['drop_pct'].append(drop_pct)

    return fidelity_drop


# =============================================================================
# FIGURA: Dashboard Completa (2x3)
# =============================================================================
def plot_dashboard(data_by_cat, method_name='', save_prefix='scgan'):
    """Dashboard principale con labels aggiornate."""

    N_values = sorted(set(n for data in data_by_cat.values() for n in data['N']))

    # Baseline per speedup
    baseline_E = {}
    if 'CGAN' in data_by_cat:
        for n, e in zip(data_by_cat['CGAN']['N'], data_by_cat['CGAN']['E_inf']):
            baseline_E[int(n)] = float(e)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    title = f'SCGAN Quantum State Tomography Benchmark'
    if method_name:
        title += f' - {method_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Fidelity vs N
    ax = axes[0, 0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        ax.plot(data['N'], data['F'], marker=marker, color=color,
                label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity vs System Size', fontsize=12)
    ax.set_ylim([0.9, 1.005])
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='F=0.99')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)

    # 2. Infidelity vs N
    ax = axes[0, 1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        infidelity = [max(1 - f, 1e-6) for f in data['F']]
        ax.semilogy(data['N'], infidelity, marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity vs System Size (lower is better)', fontsize=12)
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.001, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(max(N_values)+0.1, 0.01, 'F=0.99', fontsize=8, color='orange')
    ax.text(max(N_values)+0.1, 0.001, 'F=0.999', fontsize=8, color='green')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim([1e-4, 0.2])

    # 3. Inference Energy vs N
    ax = axes[0, 2]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        ax.semilogy(data['N'], data['E_inf'], marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Inference Energy (µJ)', fontsize=11)
    ax.set_title('Inference Energy vs System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # 4. Speedup vs N
    ax = axes[1, 0]
    has_speedup_data = False
    for cat, data in data_by_cat.items():
        if cat != 'CGAN':
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            speedups = []
            N_valid = []
            for n, e in zip(data['N'], data['E_inf']):
                n_int = int(n)
                if n_int in baseline_E and e > 0:
                    speedups.append(baseline_E[n_int] / e)
                    N_valid.append(n_int)
            if speedups:
                has_speedup_data = True
                ax.semilogy(N_valid, speedups, marker=marker, color=color,
                           label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Speedup vs CGAN (GPU)', fontsize=11)
    ax.set_title('Energy Efficiency Gain', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    if has_speedup_data:
        ax.legend(loc='upper left', fontsize=8)

    # 5. Training Energy vs N
    ax = axes[1, 1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        E_train_J = [e / 1000 for e in data['E_train']]  # mJ → J
        ax.semilogy(data['N'], E_train_J, marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Training Energy (J)', fontsize=11)
    ax.set_title('Training Energy vs System Size (GPU)', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # 6. Efficiency (F/E) vs N
    ax = axes[1, 2]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        efficiency = [f / e if e > 0 else 0 for f, e in zip(data['F'], data['E_inf'])]
        ax.semilogy(data['N'], efficiency, marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Efficiency (Fidelity / µJ)', fontsize=11)
    ax.set_title('Energy Efficiency vs System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_dashboard.png', dpi=150, bbox_inches='tight')


# =============================================================================
# FIGURA: Hardware Comparison (con fidelity drop, senza speedup)
# =============================================================================
def plot_hardware_comparison(data_by_cat, method_name='', save_prefix='scgan'):
    """Confronto energia inference per hardware platform con fidelity drop."""

    N_values = sorted(set(n for data in data_by_cat.values() for n in data['N']))
    N_to_plot = [n for n in N_values if n in [3, 5, 6, 7, 8]][:4]

    if len(N_to_plot) < 2:
        print("⚠️ Not enough N values for hardware comparison")
        return

    fig, axes = plt.subplots(1, len(N_to_plot), figsize=(4*len(N_to_plot), 5))
    if len(N_to_plot) == 1:
        axes = [axes]

    title = f'Inference Energy by Hardware Platform'
    if method_name:
        title += f' - SCGAN {method_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Labels corte per istogrammi
    hw_categories = ['CGAN\n(GPU)', 'SCGAN\n(Loihi)', 'SCGAN\n(8-bit)', 'SCGAN\n(4-bit)']
    cat_keys = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']
    cat_colors = [COLORS.get(k, 'gray') for k in cat_keys]

    for ax, N in zip(axes, N_to_plot):
        energies = []
        fidelities = []

        for cat in cat_keys:
            if cat in data_by_cat and N in data_by_cat[cat]['N']:
                idx = data_by_cat[cat]['N'].index(N)
                energies.append(data_by_cat[cat]['E_inf'][idx])
                fidelities.append(data_by_cat[cat]['F'][idx])
            else:
                energies.append(0.001)
                fidelities.append(0)

        # Calcola fidelity drop rispetto a CGAN (primo elemento)
        baseline_F = fidelities[0]
        fidelity_drops = []
        for f in fidelities:
            if baseline_F > 0:
                drop = (baseline_F - f) / baseline_F * 100
            else:
                drop = 0
            fidelity_drops.append(drop)

        bars = ax.bar(hw_categories, energies, color=cat_colors)
        ax.set_ylabel('Energy (µJ)' if N == N_to_plot[0] else '')
        ax.set_title(f'N = {N} qubits\n(d = {2**N})', fontsize=11)
        ax.set_yscale('log')

        # Annota con fidelity drop
        for bar, drop in zip(bars, fidelity_drops):
            height = bar.get_height()
            if height > 0:
                # Colore basato sul drop
                if drop < 0.01:
                    color = 'green'
                    text = '0%'
                elif drop < 1:
                    color = 'green'
                    text = f'{drop:.2f}%'
                elif drop < 5:
                    color = 'orange'
                    text = f'{drop:.1f}%'
                else:
                    color = 'red'
                    text = f'{drop:.1f}%'

                fontweight = 'bold' if drop >= 1 else 'normal'
                ax.annotate(text,
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8,
                           rotation=0, color=color, fontweight=fontweight)

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_hardware_comparison.png', dpi=150, bbox_inches='tight')


# =============================================================================
# FIGURA: Infidelity Analysis
# =============================================================================
def plot_infidelity_analysis(data_by_cat, method_name='', save_prefix='scgan'):
    """Analisi dettagliata dell'infidelity."""

    N_values = sorted(set(n for data in data_by_cat.values() for n in data['N']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = f'Infidelity Analysis (1-F)'
    if method_name:
        title += f' - SCGAN {method_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left: Infidelity vs N (line plot)
    ax = axes[0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        infidelity = [max(1 - f, 1e-6) for f in data['F']]
        ax.semilogy(data['N'], infidelity, marker=marker, color=color,
                   label=label, linewidth=2, markersize=8)

    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity vs System Size (lower is better)', fontsize=12)

    # Reference lines con zone colorate
    ax.axhspan(0.1, 1.0, alpha=0.1, color='red', label='F < 0.9')
    ax.axhspan(0.01, 0.1, alpha=0.1, color='orange')
    ax.axhspan(0.001, 0.01, alpha=0.1, color='yellow')
    ax.axhspan(1e-6, 0.001, alpha=0.1, color='green')

    ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axhline(y=0.001, color='green', linestyle=':', alpha=0.5, linewidth=1.5)

    ax.text(max(N_values)+0.15, 0.1, 'F=0.9', fontsize=8, color='red')
    ax.text(max(N_values)+0.15, 0.01, 'F=0.99', fontsize=8, color='orange')
    ax.text(max(N_values)+0.15, 0.001, 'F=0.999', fontsize=8, color='green')

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim([1e-4, 0.2])
    ax.set_xlim([min(N_values)-0.3, max(N_values)+0.8])

    # Right: Infidelity bar chart per N selezionati
    ax = axes[1]
    N_selected = [n for n in N_values if n in [3, 4, 5, 6, 7, 8]][:4]

    x = np.arange(len(N_selected))
    width = 0.18
    cats_to_plot = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']

    for i, cat in enumerate(cats_to_plot):
        if cat in data_by_cat:
            infidelities = []
            for N in N_selected:
                if N in data_by_cat[cat]['N']:
                    idx = data_by_cat[cat]['N'].index(N)
                    f = data_by_cat[cat]['F'][idx]
                    infidelities.append(max(1 - f, 1e-6))
                else:
                    infidelities.append(np.nan)

            color = COLORS.get(cat, 'gray')
            offset = (i - len(cats_to_plot)/2 + 0.5) * width
            label = get_label(cat, use_hardware_labels=True)
            ax.bar(x + offset, infidelities, width, label=label,
                  color=color, alpha=0.8)

    ax.set_yscale('log')
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax.set_title('Infidelity Comparison (lower is better)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'N={n}' for n in N_selected])
    ax.axhline(y=0.01, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='F=0.99')
    ax.legend(loc='upper left', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which='both', axis='y')
    ax.set_ylim([1e-4, 0.2])

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_infidelity.png', dpi=150, bbox_inches='tight')


# =============================================================================
# FIGURA: Scaling Analysis
# =============================================================================
def plot_scaling_analysis(data_by_cat, method_name='', save_prefix='scgan'):
    """Analisi dello scaling con la dimensione del sistema."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    title = f'Scaling Analysis'
    if method_name:
        title += f' - SCGAN {method_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Left: Energy scaling (log-log)
    ax = axes[0]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        d_values = [2**n for n in data['N']]
        ax.loglog(d_values, data['E_inf'], marker=marker, color=color,
                 label=label, linewidth=2, markersize=8)

    # Reference lines
    d_ref = np.array([8, 256])
    ax.loglog(d_ref, 0.1 * (d_ref/8)**2, 'k--', alpha=0.3, label='O(d²)')
    ax.loglog(d_ref, 0.1 * (d_ref/8)**3, 'k:', alpha=0.3, label='O(d³)')

    ax.set_xlabel('Hilbert space dimension d', fontsize=11)
    ax.set_ylabel('Inference Energy (µJ)', fontsize=11)
    ax.set_title('Energy Scaling with System Size', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=8)

    # Right: Fidelity vs Energy Trade-off
    ax = axes[1]
    for cat, data in data_by_cat.items():
        color = COLORS.get(cat, 'gray')
        marker = MARKERS.get(cat, 'o')
        label = get_label(cat)
        # Size proporzionale a N
        sizes = [30 + 20*n for n in data['N']]
        ax.scatter(data['E_inf'], data['F'], s=sizes, c=color,
                  marker=marker, label=label, alpha=0.8, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Inference Energy (µJ)', fontsize=11)
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity vs Energy Trade-off', fontsize=12)
    ax.set_xscale('log')
    ax.axhline(y=0.99, color='orange', linestyle='--', alpha=0.5, label='F=0.99')
    ax.axhline(y=0.999, color='green', linestyle=':', alpha=0.5, label='F=0.999')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_ylim([0.98, 1.002])

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_scaling.png', dpi=150, bbox_inches='tight')


# =============================================================================
# NUOVO: Confronto M1 vs M2 (istogrammi grouped)
# =============================================================================
def plot_m1_vs_m2_bar_comparison(df_M1, df_M2, save_prefix='scgan', N_selected=None):
    """
    Crea istogrammi grouped per confrontare M1 vs M2 per energia e infidelity.

    Parametri:
    ----------
    df_M1 : DataFrame
        DataFrame con risultati M1
    df_M2 : DataFrame
        DataFrame con risultati M2
    save_prefix : str
        Prefisso per i file salvati
    N_selected : list, optional
        Valori di N da mostrare. Default: [5, 6, 7, 8]
    """

    data_M1 = collect_data_by_category(df_M1)
    data_M2 = collect_data_by_category(df_M2)

    if N_selected is None:
        N_values = sorted(set(n for data in data_M1.values() for n in data['N']))
        N_selected = [n for n in N_values if n in [5, 6, 7, 8]][:4]

    cats_to_compare = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']
    cat_labels = ['CGAN\n(GPU)', 'SCGAN\n(Loihi)', 'SCGAN\n(8-bit)', 'SCGAN\n(4-bit)']

    # ==========================================================================
    # FIGURA 1: Inference Energy M1 vs M2
    # ==========================================================================
    fig, axes = plt.subplots(1, len(N_selected), figsize=(5*len(N_selected), 5))
    if len(N_selected) == 1:
        axes = [axes]

    fig.suptitle('SCGAN Inference Energy: M1 vs M2 Comparison', fontsize=14, fontweight='bold')

    bar_width = 0.35
    x = np.arange(len(cats_to_compare))

    for ax, N in zip(axes, N_selected):
        energies_M1 = []
        energies_M2 = []

        for cat in cats_to_compare:
            # M1
            if cat in data_M1 and N in data_M1[cat]['N']:
                idx = data_M1[cat]['N'].index(N)
                energies_M1.append(data_M1[cat]['E_inf'][idx])
            else:
                energies_M1.append(np.nan)

            # M2
            if cat in data_M2 and N in data_M2[cat]['N']:
                idx = data_M2[cat]['N'].index(N)
                energies_M2.append(data_M2[cat]['E_inf'][idx])
            else:
                energies_M2.append(np.nan)

        bars1 = ax.bar(x - bar_width/2, energies_M1, bar_width, label='M1', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, energies_M2, bar_width, label='M2', color='#3498db', alpha=0.8)

        ax.set_ylabel('Inference Energy (µJ)' if N == N_selected[0] else '')
        ax.set_title(f'N = {N} qubits (d = {2**N})', fontsize=11)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Annota con ratio M1/M2
        for i, (e1, e2) in enumerate(zip(energies_M1, energies_M2)):
            if not np.isnan(e1) and not np.isnan(e2) and e2 > 0:
                ratio = e1 / e2
                max_e = max(e1, e2)
                ax.annotate(f'{ratio:.2f}×',
                           xy=(x[i], max_e),
                           xytext=(0, 5), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8,
                           color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_m1_vs_m2_energy.png', dpi=150, bbox_inches='tight')

    # ==========================================================================
    # FIGURA 2: Infidelity M1 vs M2
    # ==========================================================================
    fig, axes = plt.subplots(1, len(N_selected), figsize=(5*len(N_selected), 5))
    if len(N_selected) == 1:
        axes = [axes]

    fig.suptitle('SCGAN Infidelity (1-F): M1 vs M2 Comparison', fontsize=14, fontweight='bold')

    for ax, N in zip(axes, N_selected):
        infidelities_M1 = []
        infidelities_M2 = []

        for cat in cats_to_compare:
            # M1
            if cat in data_M1 and N in data_M1[cat]['N']:
                idx = data_M1[cat]['N'].index(N)
                f = data_M1[cat]['F'][idx]
                infidelities_M1.append(max(1 - f, 1e-6))
            else:
                infidelities_M1.append(np.nan)

            # M2
            if cat in data_M2 and N in data_M2[cat]['N']:
                idx = data_M2[cat]['N'].index(N)
                f = data_M2[cat]['F'][idx]
                infidelities_M2.append(max(1 - f, 1e-6))
            else:
                infidelities_M2.append(np.nan)

        bars1 = ax.bar(x - bar_width/2, infidelities_M1, bar_width, label='M1', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, infidelities_M2, bar_width, label='M2', color='#3498db', alpha=0.8)

        ax.set_ylabel('Infidelity (1 - F)' if N == N_selected[0] else '')
        ax.set_title(f'N = {N} qubits (d = {2**N})', fontsize=11)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Linea threshold 1%
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Annota con ratio M1/M2
        for i, (inf1, inf2) in enumerate(zip(infidelities_M1, infidelities_M2)):
            if not np.isnan(inf1) and not np.isnan(inf2) and inf2 > 1e-9:
                ratio = inf1 / inf2
                max_inf = max(inf1, inf2)
                if ratio > 1.1 or ratio < 0.9:
                    ax.annotate(f'{ratio:.2f}×',
                               xy=(x[i], max_inf),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8,
                               color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_m1_vs_m2_infidelity.png', dpi=150, bbox_inches='tight')


# =============================================================================
# NUOVO: Method Comparison (line plots con ratios) - come SCNN
# =============================================================================
def plot_method_comparison(df_M1, df_M2, save_prefix='scgan'):
    """
    Confronta i risultati tra Method M1 e M2.
    Mostra ratio e differenze percentuali.
    """

    data_M1 = collect_data_by_category(df_M1)
    data_M2 = collect_data_by_category(df_M2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SCGAN Method Comparison: M1 vs M2', fontsize=14, fontweight='bold')

    cats_to_compare = ['CGAN', 'SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']

    def get_comparison_data(data_M1, data_M2, cat, metric):
        N_common = sorted(set(data_M1[cat]['N']) & set(data_M2[cat]['N']))
        vals_M1 = []
        vals_M2 = []
        for n in N_common:
            idx1 = data_M1[cat]['N'].index(n)
            idx2 = data_M2[cat]['N'].index(n)
            vals_M1.append(data_M1[cat][metric][idx1])
            vals_M2.append(data_M2[cat][metric][idx2])
        return N_common, vals_M1, vals_M2

    # Row 1: Fidelity
    # 1. Fidelity difference %
    ax = axes[0, 0]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, cat, 'F')
            diff_pct = [(f1 - f2) / f2 * 100 if f2 > 0 else 0 for f1, f2 in zip(F_M1, F_M2)]
            if diff_pct:
                ax.plot(N_common, diff_pct, marker=marker, color=color,
                       label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Fidelity Change (%)', fontsize=11)
    ax.set_title('Fidelity: (M1 - M2) / M2 × 100%\n(positive = M1 better)', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)

    # 2. Fidelity ratio
    ax = axes[0, 1]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, cat, 'F')
            ratio = [f1 / f2 if f2 > 0 else np.nan for f1, f2 in zip(F_M1, F_M2)]
            if ratio:
                ax.plot(N_common, ratio, marker=marker, color=color,
                       label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Fidelity Ratio (M1 / M2)', fontsize=11)
    ax.set_title('Fidelity Ratio\n(>1 = M1 better)', fontsize=11)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)

    # 3. Infidelity ratio
    ax = axes[0, 2]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, cat, 'F')
            infid_ratio = [(1 - f1) / (1 - f2) if (1 - f2) > 1e-9 else np.nan
                          for f1, f2 in zip(F_M1, F_M2)]
            if infid_ratio:
                ax.semilogy(N_common, infid_ratio, marker=marker, color=color,
                           label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Infidelity Ratio', fontsize=11)
    ax.set_title('Infidelity Ratio: (1-F_M1) / (1-F_M2)\n(<1 = M1 has lower error)', fontsize=11)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=7)

    # Row 2: Energy
    # 4. Inference Energy difference %
    ax = axes[1, 0]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, cat, 'E_inf')
            diff_pct = [(e1 - e2) / e2 * 100 if e2 > 0 else 0 for e1, e2 in zip(E_M1, E_M2)]
            if diff_pct:
                ax.plot(N_common, diff_pct, marker=marker, color=color,
                       label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Inference Energy Change (%)', fontsize=11)
    ax.set_title('Inference Energy: (M1 - M2) / M2 × 100%\n(negative = M1 more efficient)', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)

    # 5. Inference Energy ratio
    ax = axes[1, 1]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, cat, 'E_inf')
            ratio = [e1 / e2 if e2 > 0 else np.nan for e1, e2 in zip(E_M1, E_M2)]
            if ratio:
                ax.plot(N_common, ratio, marker=marker, color=color,
                       label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Inference Energy Ratio (M1 / M2)', fontsize=11)
    ax.set_title('Inference Energy Ratio\n(<1 = M1 more efficient)', fontsize=11)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)

    # 6. Training Energy ratio
    ax = axes[1, 2]
    for cat in cats_to_compare:
        if cat in data_M1 and cat in data_M2:
            color = COLORS.get(cat, 'gray')
            marker = MARKERS.get(cat, 'o')
            label = get_label(cat)
            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, cat, 'E_train')
            ratio = [e1 / e2 if e2 > 0 else np.nan for e1, e2 in zip(E_M1, E_M2)]
            if ratio:
                ax.plot(N_common, ratio, marker=marker, color=color,
                       label=label, linewidth=2, markersize=8)
    ax.set_xlabel('N qubits', fontsize=11)
    ax.set_ylabel('Training Energy Ratio (M1 / M2)', fontsize=11)
    ax.set_title('Training Energy Ratio\n(<1 = M1 more efficient)', fontsize=11)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_method_comparison.png', dpi=150, bbox_inches='tight')


# =============================================================================
# Summary Statistics
# =============================================================================
def print_summary_statistics(data_by_cat, method_name=''):
    """Stampa statistiche riassuntive."""

    N_values = sorted(set(n for data in data_by_cat.values() for n in data['N']))

    # Baseline per speedup
    baseline_E = {}
    if 'CGAN' in data_by_cat:
        for n, e in zip(data_by_cat['CGAN']['N'], data_by_cat['CGAN']['E_inf']):
            baseline_E[int(n)] = float(e)

    print(f"\n{'='*70}")
    print(f"📈 SUMMARY STATISTICS - SCGAN {method_name}")
    print(f"{'='*70}")

    # Problem analysis
    print(f"\n⚠️  PROBLEM ANALYSIS:")
    print(f"   Models with F < 0.99:")
    has_issues = False
    for cat, data in data_by_cat.items():
        issues = [(n, f) for n, f in zip(data['N'], data['F']) if f < 0.99]
        if issues:
            has_issues = True
            for n, f in issues:
                print(f"      ⚠️  {get_label(cat)} @ N={n}: F={f:.4f}")
    if not has_issues:
        print(f"      ✅ All models have F ≥ 0.99!")

    # Best performers at max N
    max_N = max(N_values)
    print(f"\n🏆 Performance at N={max_N} (d={2**max_N}):")

    best_data = []
    for cat, data in data_by_cat.items():
        if max_N in data['N']:
            idx = data['N'].index(max_N)
            F = data['F'][idx]
            E = data['E_inf'][idx]
            best_data.append((cat, F, E))

    print("\n  By Fidelity:")
    for cat, F, E in sorted(best_data, key=lambda x: -x[1]):
        print(f"    {get_label(cat)}: F={F:.4f}, E={E:.2f}µJ")

    print("\n  By Energy Efficiency (F/E):")
    for cat, F, E in sorted(best_data, key=lambda x: -x[1]/x[2] if x[2] > 0 else 0):
        eff = F/E if E > 0 else 0
        print(f"    {get_label(cat)}: {eff:.2f} F/µJ (F={F:.4f}, E={E:.2f}µJ)")

    # Average speedups
    print("\n  Average Speedup vs CGAN (GPU) (all N):")
    for cat in ['SCGAN-Norse', 'SCGAN-Crossbar-8b', 'SCGAN-Crossbar-4b']:
        if cat in data_by_cat:
            speedups = []
            for n, e in zip(data_by_cat[cat]['N'], data_by_cat[cat]['E_inf']):
                n_int = int(n)
                if n_int in baseline_E and e > 0:
                    speedups.append(baseline_E[n_int] / e)
            if speedups:
                print(f"    {get_label(cat)}: {np.mean(speedups):.0f}× (range: {min(speedups):.0f}-{max(speedups):.0f}×)")

    print(f"\n{'='*70}")


# =============================================================================
# FUNZIONE PRINCIPALE: Genera tutti i plot per un singolo metodo
# =============================================================================
def plot_scgan_complete(df, method_name='', save_prefix='scgan'):
    """
    Genera tutti i plot per un singolo metodo (M1 o M2).

    Parametri:
    ----------
    df : DataFrame
        DataFrame con i risultati del benchmark
    method_name : str
        Nome del metodo per i titoli (es. 'M1', 'M2')
    save_prefix : str
        Prefisso per i file salvati

    Returns:
    --------
    data_by_cat : dict
        Dati organizzati per categoria
    """

    print(f"\n{'='*60}")
    print(f"📊 Generating SCGAN plots{' for ' + method_name if method_name else ''}")
    print(f"{'='*60}")

    # Raccogli dati
    data_by_cat = collect_data_by_category(df)

    # 1. Dashboard principale
    print("  → Dashboard...")
    plot_dashboard(data_by_cat, method_name, save_prefix)

    # 2. Hardware comparison (con fidelity drop)
    print("  → Hardware comparison...")
    plot_hardware_comparison(data_by_cat, method_name, save_prefix)

    # 3. Infidelity analysis
    print("  → Infidelity analysis...")
    plot_infidelity_analysis(data_by_cat, method_name, save_prefix)

    # 4. Scaling analysis
    print("  → Scaling analysis...")
    plot_scaling_analysis(data_by_cat, method_name, save_prefix)

    # 5. Summary statistics
    print_summary_statistics(data_by_cat, method_name)

    print("\n" + "="*60)
    print("✅ All plots saved!")
    print("="*60)

    return data_by_cat


# =============================================================================
# FUNZIONE: Genera tutti i plot + confronto M1 vs M2
# =============================================================================
def plot_scgan_complete_with_comparison(df_M1, df_M2, save_prefix='scgan', N_selected=None):
    """
    Genera tutti i plot per M1 e M2, più il confronto tra i due metodi.

    Parametri:
    ----------
    df_M1 : DataFrame
        DataFrame con risultati M1
    df_M2 : DataFrame
        DataFrame con risultati M2
    save_prefix : str
        Prefisso per i file salvati
    N_selected : list, optional
        Valori di N per bar charts M1 vs M2

    Returns:
    --------
    dict con 'M1' e 'M2' data_by_cat
    """

    # Plot per M1
    data_M1 = plot_scgan_complete(df_M1, 'M1', save_prefix)

    # Plot per M2
    data_M2 = plot_scgan_complete(df_M2, 'M2', save_prefix)

    # Confronto M1 vs M2 (line plots con ratios)
    print("\n" + "="*60)
    print("📊 Generating M1 vs M2 comparison plots")
    print("="*60)
    print("  → Method comparison (ratios)...")
    plot_method_comparison(df_M1, df_M2, save_prefix)

    # Confronto M1 vs M2 (istogrammi grouped)
    print("  → M1 vs M2 bar charts (energy & infidelity)...")
    plot_m1_vs_m2_bar_comparison(df_M1, df_M2, save_prefix, N_selected)

    return {'M1': data_M1, 'M2': data_M2}


# === cell #19 ===
# =============================================================================
# OPZIONE 1: Plot separati per M1 e M2
# =============================================================================

# Visualization M1
data_M1 = plot_scgan_complete(df_M1, method_name='M1', save_prefix='scgan')

# Visualization M2
data_M2 = plot_scgan_complete(df_M2, method_name='M2', save_prefix='scgan')

# Confronto M1 vs M2 (istogrammi)
plot_m1_vs_m2_bar_comparison(df_M1, df_M2,
                              save_prefix='scgan', N_selected=[5, 6, 7, 8])

