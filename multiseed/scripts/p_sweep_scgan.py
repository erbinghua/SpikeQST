import sys
#!/usr/bin/env python3
"""AUTO-GENERATED from p_sweep_SCGAN_Mixed.ipynb.

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
print("SCGAN ENERGY BENCHMARK: Mixed GHZ States (Werner p-sweep)")
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
# p-SWEEP RUNNER (mirrors run_scgan_benchmark_gpu, with p threaded in)
# =============================================================================

def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def werner_purity(p, d):
    """Tr(rho^2) for Werner state rho = p|GHZ><GHZ| + (1-p) I/d."""
    return p**2 + (1.0 - p)**2 / d + 2.0 * p * (1.0 - p) / d


def run_p_sweep_scgan(p_list, N=5, methods=['M1', 'M2'], M_M2=4, M_M1=256,
                     steps=1000, lr=1e-3, T=8, sparsity=0.1,
                     crossbar_bits=[8, 4], use_amp=False):
    """p-sweep matching the SCGAN benchmark methodology exactly.

    Same model configs and training params as run_scgan_benchmark_gpu, but loops
    over p_list at fixed N rather than over N_list at fixed p=0.5.
    """
    # === MULTISEED PATCHED ===
    if _ARGS.quick:
        p_list = [0.5]
        methods = ['M1']
        steps = 3
        print(f"[quick] p_list={p_list} methods={methods}", flush=True)

    if CSV_PATH.exists():
        _df_existing = pd.read_csv(CSV_PATH)
        rows = _df_existing.to_dict('records')
        _counts = {}
        for r in rows:
            if r.get('seed') == SEED:
                k = (r['method'], float(r['p']))
                _counts[k] = _counts.get(k, 0) + 1
        _done_outer = {k for k, c in _counts.items() if c >= 4}
        print(f"  [resume] {len(_done_outer)} (method, p) tuples complete for seed={SEED}", flush=True)
    else:
        rows = []
        _done_outer = set()
    rows = rows  # (was [])
    d = 2**N
    print(f"\n{'='*80}\nSCGAN p-SWEEP: N={N}, T={T}, p_list={p_list}\n{'='*80}")

    total = len(p_list) * len(methods)
    run_idx = 0

    for method in methods:
        method = method.upper()
        is_M2 = (method == 'M2')

        for p in p_list:
            run_idx += 1
            if (method, float(p)) in _done_outer:
                print(f"  [skip] seed={SEED} method={method} p={p:.3f}", flush=True)
                continue
            purity = werner_purity(p, d)
            print(f"\n[{run_idx}/{total}] method={method}  p={p:.3f}  purity={purity:.4f}")

            # Build target Werner state at this p
            rho_true = mixed_ghz_state(N, p=p)

            # Same M selection as benchmark
            if is_M2:
                M_eff = M_M2
                bases = generate_bases_M2(N, M_eff)
                x_target = measure_M2(rho_true, bases)
                measure = MeasurementM2(bases)
            else:
                M_eff = min(4**N - 1, M_M1)
                ops = select_ops_nonzero_M1(rho_true, M_eff, N, seed=1234)
                x_target = measure_M1(rho_true, ops)
                measure = MeasurementM1(ops)

            num_points = x_target.shape[-1]
            print(f"  {method}: M={M_eff} ({'bases' if is_M2 else 'Pauli ops'}, {num_points} cond values)")

            # ---- 1. CGAN (GPU) ----
            print("  [1/4] CGAN")
            G = CNNGenBase2D(d, num_points, measure)
            D = DiscClassic(num_points)
            res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr,
                              is_M2=is_M2, use_amp=use_amp)
            E_inf = estimate_gpu_inference_energy(G)
            rows.append({'seed': SEED, 'Model': 'CGAN', 'Architecture': 'SCGAN', 'HW': 'GPU',
                         'N': N, 'method': method, 'p': p, 'purity': purity,
                         'F_best': res['F_best'], 'F_last': float(res['f_hist'][-1]) if res['f_hist'] else res['F_best'],
                         'F_mean': res['F_mean'],
                         'E_inference_uJ': E_inf['E_total_uJ'],
                         'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
            print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.2f} uJ")
            del G, D, res; gpu_cleanup()

            # ---- 2. SCGAN-Norse (Loihi) ----
            print("  [2/4] SCGAN-Norse")
            G = SCNNGenNorse2D(d, num_points, measure, T=T, v_th=0.3, is_prob_input=is_M2)
            D = DiscSpiking(num_points, T=T, v_th=0.3)
            res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr,
                              is_M2=is_M2, use_amp=use_amp)
            E_inf = estimate_loihi_inference_energy(G, T=T, sparsity=sparsity)
            rows.append({'seed': SEED, 'Model': 'SCGAN-Norse', 'Architecture': 'SCGAN', 'HW': 'Loihi',
                         'N': N, 'method': method, 'p': p, 'purity': purity,
                         'F_best': res['F_best'], 'F_last': float(res['f_hist'][-1]) if res['f_hist'] else res['F_best'],
                         'F_mean': res['F_mean'],
                         'E_inference_uJ': E_inf['E_total_uJ'],
                         'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
            print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f} uJ")
            del G, D, res; gpu_cleanup()

            # ---- 3,4. SCGAN-Crossbar-{8,4}b ----
            for ci, bits in enumerate(crossbar_bits, start=3):
                print(f"  [{ci}/4] SCGAN-Crossbar-{bits}b")
                v_th = 0.2 if bits <= 4 else 0.3
                G = SCNNGenCrossbar2D(d, num_points, measure, T=T, v_th=v_th,
                                      weight_bits=bits, adc_bits=bits, dac_bits=bits,
                                      is_prob_input=is_M2)
                D = DiscSpiking(num_points, T=T, v_th=0.3)
                res = train_scgan(G, D, x_target, rho_true, steps=steps, lr=lr,
                                  is_M2=is_M2, use_amp=use_amp)
                E_inf = estimate_crossbar_inference_energy(G, T=T, sparsity=sparsity, bits=bits)
                rows.append({'seed': SEED, 'Model': f'SCGAN-Crossbar-{bits}b', 'Architecture': 'SCGAN',
                             'HW': f'Crossbar-{bits}b', 'N': N, 'method': method, 'p': p, 'purity': purity,
                             'F_best': res['F_best'], 'F_last': float(res['f_hist'][-1]) if res['f_hist'] else res['F_best'],
                             'F_mean': res['F_mean'],
                             'E_inference_uJ': E_inf['E_total_uJ'],
                             'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
                print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f} uJ")
                del G, D, res; gpu_cleanup()

            # Cleanup measurement structures for this p
            if is_M2:
                del bases
            else:
                del ops
            del x_target, measure, rho_true
            gpu_cleanup()

            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)  # multiseed: save after each (method, p)

    return pd.DataFrame(rows)


print("Defined: run_p_sweep_scgan")


# === cell #12 ===
# =============================================================================
# p-SWEEP CONFIGURATION
# =============================================================================
P_LIST = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
N_FIXED = 5
T_FIXED = 8
STEPS = 1000  # SCGAN benchmark default
LR = 1e-3
print(f"p_list = {P_LIST}")
print(f"N = {N_FIXED}, T = {T_FIXED}, steps = {STEPS}")
print(f"Models per (p, method): 4  (CGAN, SCGAN-Norse, SCGAN-Crossbar-8b, SCGAN-Crossbar-4b)")
print(f"Total runs: {len(P_LIST)} p-values * 2 methods * 4 models = {len(P_LIST)*2*4}")


# === cell #13 ===
# =============================================================================
# RUN p-SWEEP (M1 + M2 in one pass)
# =============================================================================
df_p_sweep = run_p_sweep_scgan(
    p_list=P_LIST, N=N_FIXED, methods=['M1', 'M2'],
    M_M2=4, M_M1=256,
    steps=STEPS, lr=LR,
    T=T_FIXED, sparsity=0.1,
    crossbar_bits=[8, 4], use_amp=False)

# === MULTISEED quick-exit ===
# Skip the post-sweep analysis / projection / "best M" cells in --quick mode:
# they assume a full sweep grid and crash on the smoke-test subset.
if _ARGS.quick:
    print("[multiseed] --quick: skipping post-sweep analysis cells", flush=True)
    sys.exit(0)


# === cell #14 ===
# =============================================================================
# SAVE CSV
# =============================================================================
import os
csv_path = CSV_PATH
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df_p_sweep.to_csv(csv_path, index=False)
print(f"Saved {len(df_p_sweep)} rows to {csv_path}")


# === cell #15 ===
# =============================================================================
# PLOT: F vs p for each (Model, method)
# =============================================================================
COLORS = {'CGAN':'#1f77b4', 'SCGAN-Norse':'#2ca02c',
          'SCGAN-Crossbar-8b':'#ff7f0e', 'SCGAN-Crossbar-4b':'#d62728'}
MARKERS = {'CGAN':'o', 'SCGAN-Norse':'s',
           'SCGAN-Crossbar-8b':'P', 'SCGAN-Crossbar-4b':'X'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax_idx, method in enumerate(['M1', 'M2']):
    ax = axes[ax_idx]
    df_m = df_p_sweep[df_p_sweep['method'] == method]
    for model_name in df_m['Model'].unique():
        df_model = df_m[df_m['Model'] == model_name].sort_values('p')
        ax.plot(df_model['p'], df_model['F_best'],
                marker=MARKERS.get(model_name, 'o'),
                color=COLORS.get(model_name, 'gray'),
                label=model_name, linewidth=2, markersize=8)
    ax.set_xlabel('Werner mixing parameter p', fontsize=12)
    ax.set_ylabel('Fidelity (best)', fontsize=12)
    ax.set_title(f'SCGAN p-sweep — {method} (N={N_FIXED}, T={T_FIXED})', fontsize=13)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='F=0.8')
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.4, label='Benchmark p=0.5')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('p_sweep_SCGAN_fidelity_vs_p.png', dpi=150, bbox_inches='tight')
plt.savefig('p_sweep_SCGAN_fidelity_vs_p.pdf', bbox_inches='tight')
print("Saved: p_sweep_SCGAN_fidelity_vs_p.png/pdf")

