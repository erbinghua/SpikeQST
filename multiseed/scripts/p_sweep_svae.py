import sys
#!/usr/bin/env python3
"""AUTO-GENERATED from p_sweep_SVAE_Mixed.ipynb.

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
print(f"[multiseed] seed={SEED}  out_dir={OUT_DIR}", flush=True)


# === cell #0 ===
# =============================================================================
# 1. SETUP & IMPORTS
# =============================================================================
import os, math, random, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path
import norse.torch as norse

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# GPU Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Checkpoint directory
# (multiseed: redirected to per-seed OUT_DIR)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "results.csv"
print("="*70)
print("🚀 QUANTUM STATE TOMOGRAPHY: VAE vs SVAE ENERGY BENCHMARK - MIXED GHZ")
print("="*70)
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✓ Norse: Installed")


# === cell #1 ===
# =============================================================================
# 2. HARDWARE ENERGY PARAMETERS
# =============================================================================

"""
Riferimenti:
- GPU: Horowitz ISSCC 2014 (scalato a 7nm), NVIDIA A100 whitepaper
- Loihi: Davies+ IEEE Micro 2018 "Loihi: A Neuromorphic Manycore Processor"
- Memristor: Cai+ Nature Electronics 2019, Yao+ Nature 2020
"""

# GPU (NVIDIA A100, 7nm)
GPU_PARAMS = {
    'name': 'NVIDIA A100 (7nm)',
    'E_MAC': 35e-12,              # 35 pJ/MAC (FP16/FP32 fused)
    'E_DRAM': 640e-12,            # 640 pJ/DRAM access
    'E_L2_cache': 5e-12,          # 5 pJ/L2 access
    'cache_hit_rate': 0.85,       # 85% cache hit rate
    'TDP_W': 400,                 # 400W TDP
    'util_small_model': 0.05,     # 5% utilization for small models
}

# Intel Loihi (14nm)
LOIHI_PARAMS = {
    'name': 'Intel Loihi (14nm)',
    'E_spike': 23.6e-12,          # 23.6 pJ/spike (axon + synapse + dendrite)
    'E_synapse': 81e-15,          # 81 fJ/synaptic operation
    'E_neuron_leak': 0.5e-12,     # 0.5 pJ/neuron/timestep leakage
    'E_router': 10e-12,           # 10 pJ/spike routing
}

# Memristor Crossbar (65nm CMOS + RRAM)
MEMRISTOR_PARAMS = {
    'name': 'Memristor Crossbar (65nm)',
    'E_MAC_analog': 2e-15,        # 2 fJ/MAC (resistive MVM)
    'E_ADC_8bit': 20e-12,         # 20 pJ/conversion (8-bit SAR)
    'E_ADC_4bit': 5e-12,          # 5 pJ/conversion (4-bit)
    'E_DAC_8bit': 10e-12,         # 10 pJ/conversion (8-bit)
    'E_DAC_4bit': 2.5e-12,        # 2.5 pJ/conversion (4-bit)
    'E_spike_gen': 3e-12,         # 3 pJ/spike (LIF analog circuit)
    'E_write': 100e-12,           # 100 pJ/weight write (RRAM program)
    'E_leakage': 0.1e-12,         # 0.1 pJ/neuron/timestep
}

print("="*70)
print("📊 HARDWARE ENERGY PARAMETERS")
print("="*70)

print(f"\n🖥️  GPU ({GPU_PARAMS['name']}):")
print(f"   E_MAC = {GPU_PARAMS['E_MAC']*1e12:.0f} pJ")
print(f"   E_DRAM = {GPU_PARAMS['E_DRAM']*1e12:.0f} pJ")
print(f"   E_L2 = {GPU_PARAMS['E_L2_cache']*1e12:.0f} pJ")

print(f"\n🧠 Loihi ({LOIHI_PARAMS['name']}):")
print(f"   E_spike = {LOIHI_PARAMS['E_spike']*1e12:.1f} pJ")
print(f"   E_synapse = {LOIHI_PARAMS['E_synapse']*1e15:.0f} fJ")
print(f"   E_leak = {LOIHI_PARAMS['E_neuron_leak']*1e12:.1f} pJ/neuron/ts")

print(f"\n⚡ Crossbar ({MEMRISTOR_PARAMS['name']}):")
print(f"   E_MAC_analog = {MEMRISTOR_PARAMS['E_MAC_analog']*1e15:.0f} fJ")
print(f"   E_ADC_8bit = {MEMRISTOR_PARAMS['E_ADC_8bit']*1e12:.0f} pJ")
print(f"   E_ADC_4bit = {MEMRISTOR_PARAMS['E_ADC_4bit']*1e12:.0f} pJ")
print(f"   E_DAC_8bit = {MEMRISTOR_PARAMS['E_DAC_8bit']*1e12:.0f} pJ")


# === cell #2 ===
# =============================================================================
# ENERGY ESTIMATION FUNCTIONS (CORRECTED: per-layer MAC counting)
# =============================================================================

def _is_linear_like(m):
    """Return True for nn.Linear OR any custom layer that exposes
    in_features/out_features/weight (e.g. QuantizedLinear).
    Needed because the crossbar models use QuantizedLinear which does not
    inherit from nn.Linear, so plain isinstance checks miss them."""
    import torch.nn as _nn
    return isinstance(m, _nn.Linear) or (
        hasattr(m, 'in_features') and hasattr(m, 'out_features') and hasattr(m, 'weight')
    )

def count_macs_per_inference(model, T=8):
    """Count MAC ops per inference with T spiking timesteps (Linear-only model)."""
    import torch.nn as nn
    macs = {}
    for name, m in model.named_modules():
        if _is_linear_like(m):
            macs[name] = m.in_features * m.out_features * T
        # Also catch CrossbarLinear (not subclass of nn.Linear)
        elif hasattr(m, 'weight') and hasattr(m, 'in_features') and not _is_linear_like(m):
            macs[name] = m.in_features * m.out_features * T
        elif hasattr(m, 'crossbar') and hasattr(m.crossbar, 'weight'):
            w = m.crossbar.weight
            macs[name] = w.size(1) * w.size(0) * T
    return macs


def estimate_gpu_inference_energy(model, batch_size=1):
    """GPU inference energy for VAE (non-spiking, T=1)."""
    n_params = sum(p.numel() for p in model.parameters())
    macs_dict = count_macs_per_inference(model, T=1)
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
    if measured_time_s is not None:
        E_total = GPU_PARAMS['TDP_W'] * util * measured_time_s
    else:
        n_macs_per_step = n_params * 6 * batch_size
        total_macs = n_macs_per_step * steps
        E_total = total_macs * GPU_PARAMS['E_MAC'] * (1 / util)
    return {'E_total_J': E_total, 'E_total_mJ': E_total * 1e3,
            'n_params': n_params, 'utilization': util, 'time_s': measured_time_s}


def estimate_loihi_inference_energy(model, T=8, sparsity=0.1, batch_size=1):
    """Loihi inference energy for SVAE."""
    n_params = sum(p.numel() for p in model.parameters())
    # Count neurons from all linear-like layers
    n_neurons = 0
    for m in model.modules():
        if _is_linear_like(m):
            n_neurons += m.out_features
        elif hasattr(m, 'fc') and isinstance(m.fc, nn.Linear):
            n_neurons += m.fc.out_features
        elif hasattr(m, 'crossbar') and hasattr(m.crossbar, 'weight'):
            n_neurons += m.crossbar.weight.size(0)
        elif hasattr(m, 'weight') and hasattr(m, 'in_features') and not _is_linear_like(m):
            n_neurons += m.weight.size(0)

    macs_dict = count_macs_per_inference(model, T=T)
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
            'E_spikes_J': E_spikes, 'E_syn_J': E_syn,
            'E_leak_J': E_leak, 'E_routing_J': E_routing,
            'n_params': n_params, 'n_neurons': n_neurons, 'n_spikes': n_spikes,
            'T': T, 'sparsity': sparsity,
            'breakdown': {
                'spikes_%': E_spikes/E_total*100 if E_total > 0 else 0,
                'syn_%': E_syn/E_total*100 if E_total > 0 else 0,
                'leak_%': E_leak/E_total*100 if E_total > 0 else 0,
                'routing_%': E_routing/E_total*100 if E_total > 0 else 0}}


def estimate_crossbar_inference_energy(model, bits=8, T=8, sparsity=0.1, batch_size=1):
    """Memristor crossbar inference energy for SVAE."""
    E_ADC = MEMRISTOR_PARAMS['E_ADC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_ADC_4bit']
    E_DAC = MEMRISTOR_PARAMS['E_DAC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_DAC_4bit']
    n_params = sum(p.numel() for p in model.parameters())

    total_in, total_out = 0, 0
    for m in model.modules():
        if _is_linear_like(m):
            total_in += m.in_features; total_out += m.out_features
        elif hasattr(m, 'crossbar') and hasattr(m.crossbar, 'weight'):
            total_in += m.crossbar.weight.size(1); total_out += m.crossbar.weight.size(0)
        elif hasattr(m, 'weight') and hasattr(m, 'in_features') and not _is_linear_like(m):
            total_in += m.in_features; total_out += m.out_features

    macs_dict = count_macs_per_inference(model, T=T)
    total_macs = sum(macs_dict.values())
    active_in = int(total_in * sparsity)
    E_dac = active_in * E_DAC * T * batch_size
    E_mvm = int(total_macs * sparsity) * MEMRISTOR_PARAMS['E_MAC_analog'] * batch_size
    E_adc = total_out * E_ADC * T * batch_size  # NO sparsity
    n_spikes = int(total_out * T * sparsity * batch_size)
    E_spike = n_spikes * MEMRISTOR_PARAMS['E_spike_gen']
    E_leak = total_out * T * batch_size * MEMRISTOR_PARAMS['E_leakage']
    E_total = E_dac + E_mvm + E_adc + E_spike + E_leak
    return {'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
            'E_total_mJ': E_total * 1e3,
            'E_dac_J': E_dac, 'E_mvm_J': E_mvm, 'E_adc_J': E_adc,
            'E_spike_gen_J': E_spike, 'E_leak_J': E_leak,
            'n_params': n_params, 'bits': bits, 'T': T, 'sparsity': sparsity,
            'breakdown': {
                'DAC_%': E_dac/E_total*100 if E_total > 0 else 0,
                'MVM_%': E_mvm/E_total*100 if E_total > 0 else 0,
                'ADC_%': E_adc/E_total*100 if E_total > 0 else 0,
                'spike_gen_%': E_spike/E_total*100 if E_total > 0 else 0,
                'leak_%': E_leak/E_total*100 if E_total > 0 else 0}}

print("✓ Energy estimation functions defined (corrected: per-layer MAC counting)")


# === cell #3 ===
# =============================================================================
# 4. QUANTUM UTILITIES
# =============================================================================

# Pauli matrices
I_np = np.eye(2, dtype=np.complex128)
X_np = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y_np = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z_np = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULIS = {'I': I_np, 'X': X_np, 'Y': Y_np, 'Z': Z_np}


def pauli_op(string):
    """Create N-qubit Pauli operator from string like 'XYZ'"""
    out = PAULIS[string[0]]
    for s in string[1:]:
        out = np.kron(out, PAULIS[s])
    return out


def ghz_density(N):
    """Create GHZ state density matrix: |GHZ⟩ = (|0...0⟩ + |1...1⟩)/√2"""
    d = 2**N
    psi = np.zeros(d, dtype=np.complex128)
    psi[0] = 1/np.sqrt(2)
    psi[-1] = 1/np.sqrt(2)
    return np.outer(psi, psi.conj())


def mixed_ghz_state(N, p=0.5):
    """
    Generalized Werner state (Eq. A4 from Hua et al., arXiv:2507.23007).

    rho = p |GHZ><GHZ| + (1 - p) I_N / 2^N

    Args:
        N: Number of qubits
        p: Pure-state weight (1 = pure GHZ, 0 = maximally mixed)

    Returns:
        rho: (d, d) complex128 density matrix
    """
    d = 2**N
    rho_pure = ghz_density(N)
    I_d = np.eye(d, dtype=np.complex128) / d
    rho = p * rho_pure + (1 - p) * I_d
    rho = 0.5 * (rho + rho.conj().T)
    return (rho / np.real(np.trace(rho))).astype(np.complex128)



def fidelity_np(rho1, rho2, eps=1e-12):
    """Quantum fidelity between two density matrices (Uhlmann formula)."""
    from scipy.linalg import sqrtm
    rho1 = 0.5 * (rho1 + rho1.conj().T)
    rho2 = 0.5 * (rho2 + rho2.conj().T)
    sqrt_rho1 = sqrtm(rho1 + eps * np.eye(rho1.shape[0]))
    M = sqrt_rho1 @ rho2 @ sqrt_rho1
    M = 0.5 * (M + M.conj().T)
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.maximum(eigvals.real, 0)
    F = float(np.sum(np.sqrt(eigvals))**2)
    if F > 1.0 + 1e-6:
        print(f"WARNING fidelity_np: F={F:.6f} > 1, possible numerical issue")
    return float(np.clip(F, 0, 1))


def select_paulis_nonzero(rho, N, M, tol=1e-8, tries=100):
    """Select M Pauli operators with non-zero expectation values."""
    from itertools import product as iterprod
    L = 4**N
    strings_all = [''.join(p) for p in iterprod('IXYZ', repeat=N)]
    rng = np.random.default_rng(1234)
    chosen, seen = [], set()

    for _ in range(tries):
        pool_sz = min(5*M + 32, L)
        idx = rng.choice(L, size=pool_sz, replace=False)
        for i in idx:
            s = strings_all[i]
            if s in seen: continue
            A = pauli_op(s)
            if abs(np.real(np.trace(rho @ A))) > tol:
                chosen.append(A); seen.add(s)
                if len(chosen) >= M: break
        if len(chosen) >= M: break

    while len(chosen) < M:
        for s in strings_all:
            if s not in seen:
                chosen.append(pauli_op(s))
                if len(chosen) >= M: break

    return chosen[:M]


def select_bases_nonzero_M2(rho, N, M, seed=0):
    """Select M measurement bases for M2 method."""
    rng = np.random.default_rng(seed)
    bases = []
    if M >= 1: bases.append('Z'*N)
    if M >= 2: bases.append('X'*N)
    if M >= 3: bases.append('Y'*N)
    letters = np.array(list('XYZ'))
    while len(bases) < M:
        cand = ''.join(rng.choice(letters, size=N))
        if cand not in bases: bases.append(cand)
    return bases[:M]


def _eigenbasis_1q(axis, device=None, dtype=torch.complex64):
    """Get single-qubit eigenbasis transformation."""
    if axis == 'Z':
        return torch.eye(2, dtype=dtype, device=device)
    elif axis == 'X':
        return (1/torch.sqrt(torch.tensor(2., device=device))) * torch.tensor(
            [[1,1],[1,-1]], dtype=dtype, device=device)
    elif axis == 'Y':
        return (1/torch.sqrt(torch.tensor(2., device=device))) * torch.tensor(
            [[1,1],[1j,-1j]], dtype=dtype, device=device)
    raise ValueError(f"axis must be X/Y/Z, got {axis}")


def _kronN_torch(mats):
    """Kronecker product of list of matrices."""
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out


def probs_from_bases_torch(rho_ri, bases):
    """Calculate probability distributions from density matrix in given bases."""
    device, B, _, d, _ = rho_ri.device, *rho_ri.shape
    N = int(math.log2(d))

    with torch.amp.autocast('cuda', enabled=False):
        rho = torch.complex(rho_ri[:,0].float(), rho_ri[:,1].float())
        plist = []
        for b in bases:
            E = _kronN_torch([_eigenbasis_1q(ax, device) for ax in b])
            Ec = E.conj().transpose(0,1)
            rho_p = Ec @ rho @ E
            p = rho_p.diagonal(dim1=-2, dim2=-1).real
            p = torch.clamp(p, min=0)
            p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            plist.append(p)

    return torch.cat(plist, dim=-1)


def fidelity_batch(rho_pred_ri, rho_true_ri, eps=1e-12):
    """Quantum fidelity with double-precision arithmetic.
    Uses a pure-state fast path when rho_true is pure (rank 1):
        F = Tr(rho_true @ rho_pred)
    For mixed target states falls back to the general Uhlmann formula
    computed in complex128 with eigenvalues clamped to min=0 (not eps).
    """
    with torch.amp.autocast('cuda', enabled=False):
        c_dtype = torch.complex128
        rho_p = torch.complex(rho_pred_ri[:,0].double(), rho_pred_ri[:,1].double()).to(c_dtype)
        rho_t = torch.complex(rho_true_ri[:,0].double(), rho_true_ri[:,1].double()).to(c_dtype)
        d = rho_t.shape[-1]
        batch_size = rho_pred_ri.shape[0]

        # Pure-state fast path
        purity = torch.real(torch.diagonal(rho_t @ rho_t, dim1=-2, dim2=-1).sum(-1))
        is_pure = (purity > 1.0 - 1e-6)

        if is_pure.all():
            fid = torch.real(torch.diagonal(rho_t @ rho_p, dim1=-2, dim2=-1).sum(-1))
            fid = torch.clamp(fid, min=0.0)
            if (fid > 1.0 + 1e-6).any():
                print(f"WARNING fidelity_batch: pure-state fidelity "
                      f"{fid.max().item():.6f} > 1 (d={d}), possible numerical issue")
            fid = torch.clamp(fid, max=1.0)
            return fid, fid.mean().item()

        # General Uhlmann fidelity (mixed states)
        def sqrtm_psd(A):
            A = 0.5*(A + A.conj().transpose(-1,-2))
            A = A + eps * torch.eye(d, device=A.device, dtype=A.dtype)
            try:
                ev, V = torch.linalg.eigh(A)
            except torch.linalg.LinAlgError as e:
                print(f"WARNING sqrtm_psd: eigh failed (d={d}): {e}")
                return None
            ev = torch.clamp(ev.real, min=0.0)
            D = torch.diag_embed(torch.sqrt(ev)).to(c_dtype)
            return V @ D @ V.conj().transpose(-1,-2)

        try:
            sqrt_t = sqrtm_psd(rho_t)
            if sqrt_t is None:
                fid = torch.zeros(batch_size, device=rho_pred_ri.device)
                return fid, 0.0
            M = sqrt_t @ rho_p @ sqrt_t
            M = 0.5*(M + M.conj().transpose(-1,-2))
            ev = torch.linalg.eigvalsh(M)
            ev = torch.clamp(ev.real, min=0.0)
            fid = torch.sum(torch.sqrt(ev), dim=-1)**2
            if (fid > 1.0 + 1e-6).any():
                print(f"WARNING fidelity_batch: Uhlmann fidelity "
                      f"{fid.max().item():.6f} > 1 (d={d}), possible numerical issue")
            fid = torch.clamp(fid, min=0.0, max=1.0)
        except torch.linalg.LinAlgError as e:
            print(f"WARNING fidelity_batch: LinAlgError (d={d}): {e}")
            fid = torch.zeros(batch_size, device=rho_pred_ri.device)
        except RuntimeError as e:
            print(f"WARNING fidelity_batch: RuntimeError (d={d}): {e}")
            fid = torch.zeros(batch_size, device=rho_pred_ri.device)

    return fid, fid.mean().item()

print("✓ Quantum utilities defined")


# === cell #4 ===
# =============================================================================
# 5. DENSITY MATRIX RECONSTRUCTION LAYERS
# =============================================================================

class DensityMap(nn.Module):
    """
    Maps output to valid density matrix via AA†/Tr(AA†).
    Input: [B, 2, d, d] (real and imaginary parts)
    Output: [B, 2, d, d] (normalized density matrix)
    """
    def forward(self, aa_ri):
        Ar, Ai = aa_ri[:,0], aa_ri[:,1]

        with torch.amp.autocast('cuda', enabled=False):
            A = torch.complex(Ar.float(), Ai.float())
            M = A @ A.conj().transpose(-1,-2)
            M = 0.5*(M + M.conj().transpose(-1,-2))
            tr = torch.real(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)).clamp_min(1e-12)
            M = M / tr.view(-1,1,1)
            result = torch.stack([M.real, M.imag], dim=1)

        return result


class ExpectationLayer(nn.Module):
    """Calculate expectation values ⟨A⟩ = Tr(ρA) for fixed operators."""
    def __init__(self):
        super().__init__()
        self.register_buffer('Acmplx', None)

    @torch.no_grad()
    def set_ops(self, ops_ri_fixed):
        Ar = ops_ri_fixed[:, 0]
        Ai = ops_ri_fixed[:, 1]
        self.Acmplx = torch.complex(Ar.float(), Ai.float())

    def forward(self, rho_ri, ops_ri=None):
        with torch.amp.autocast('cuda', enabled=False):
            rho = torch.complex(rho_ri[:,0].float(), rho_ri[:,1].float())
            if self.Acmplx is not None and ops_ri is None:
                tr = torch.einsum('bij,mji->bm', rho, self.Acmplx)
            else:
                Ar = ops_ri[:, 0].float()
                Ai = ops_ri[:, 1].float()
                A = torch.complex(Ar, Ai)
                tr = torch.einsum('bij,bijm->bm', rho, A)
            result = torch.real(tr)

        return result

print("✓ Density matrix layers defined")


# === cell #5 ===
# =============================================================================
# 6. NORSE LIF BASE CLASSES - HARDWARE REALISTIC SPIKING NEURONS
# =============================================================================

def _poisson_st(x, gamma=1.5, pmin=0.02, pmax=0.98):
    """
    Poisson encoding with Straight-Through gradient.
    Converts continuous values to spike probabilities.
    """
    # Standardize input
    x_mean = x.mean()
    x_std = x.std() + 1e-8
    z = (x - x_mean) / x_std

    # Sigmoid to get probabilities
    p = torch.sigmoid(gamma * z)
    p = torch.clamp(p, pmin, pmax)

    # Stochastic spike generation with STE
    spikes = (torch.rand_like(p) < p).float()
    return p + (spikes - p).detach()


class NorseLIFBase(nn.Module):
    """
    Base class for Norse LIF neurons with Loihi-compatible parameters.

    Features:
    - Hardware-realistic LIF dynamics via Norse
    - 8-bit weight quantization (Loihi INT8)
    - Poisson spike encoding
    - SuperSpike surrogate gradients

    Args:
        T: Number of timesteps
        tau_mem_inv: Membrane decay rate (1/\u03c4_mem)
        tau_syn_inv: Synaptic decay rate (1/\u03c4_syn)
        v_th: Firing threshold
        return_rate: If True, return spike rate; else spike count
        weight_bits: Weight quantization bits (8 for Loihi)
        enc_mode: Encoding mode ('poisson' or 'none')
        enc_gamma: Poisson encoding sharpness
    """

    def __init__(self, output_features, T=8, tau_mem_inv=100.0, tau_syn_inv=200.0,
                 v_th=1.0, return_rate=True, weight_bits=8,
                 enc_mode='poisson', enc_gamma=1.5, enc_pmin=0.02, enc_pmax=0.98,
                 **kwargs):
        super().__init__()

        self.T = T
        self.tau_mem_inv = tau_mem_inv
        self.tau_syn_inv = tau_syn_inv
        self.v_th = v_th
        self.return_rate = return_rate
        self.weight_bits = weight_bits
        self.enc_mode = enc_mode
        self.enc_gamma = enc_gamma
        self.enc_pmin = enc_pmin
        self.enc_pmax = enc_pmax

        # Norse LIF parameters
        self.lif_params = norse.LIFParameters(
            tau_mem_inv=torch.tensor(tau_mem_inv),
            tau_syn_inv=torch.tensor(tau_syn_inv),
            v_th=torch.tensor(v_th),
            v_reset=torch.tensor(0.0),
            method="super",
            alpha=torch.tensor(100.0)
        )

        self.lif_cell = norse.LIFCell(p=self.lif_params)
        self.output_features = output_features

    def quantize_weights(self, w):
        """
        Quantize weights to n-bit resolution with Straight-Through Estimator.
        Loihi uses 8-bit signed weights.
        """
        if self.weight_bits >= 32:
            return w

        n_levels = 2 ** self.weight_bits

        w_max = w.abs().max()
        if w_max == 0:
            return w

        w_normalized = w / w_max
        w_quant_norm = torch.round(w_normalized * (n_levels // 2 - 1)) / (n_levels // 2 - 1)
        w_quant = w_quant_norm * w_max

        return w + (w_quant - w).detach()

    def current(self, x):
        raise NotImplementedError

    def _encode_x(self, x):
        if self.enc_mode == 'poisson':
            return _poisson_st(x, self.enc_gamma, self.enc_pmin, self.enc_pmax)
        return x

    def forward(self, x):
        state = None
        acc = None

        for t in range(self.T):
            x_encoded = self._encode_x(x)
            I_t = self.current(x_encoded)
            spikes, state = self.lif_cell(I_t, state)
            if acc is None:
                acc = torch.zeros_like(spikes)
            acc = acc + spikes

        return acc / float(self.T) if self.return_rate else acc


class NorseLIFLinear(NorseLIFBase):
    """Norse LIF with Linear layer and Loihi 8-bit weight quantization."""
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(output_features=out_features, **kwargs)
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def current(self, x):
        w_quant = self.quantize_weights(self.fc.weight)
        return F.linear(x, w_quant, self.fc.bias)


def init_normal_002(m):
    """Initialize weights with N(0, 0.02)."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def warm_init_spiking(model, w_scale=5.0, bias=0.2):
    """Warm initialization for spiking networks to push neurons above threshold."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(w_scale)
                if m.bias is not None:
                    m.bias.add_(bias)

print("\u2713 Norse LIF base classes defined")


# === cell #6 ===
# =============================================================================
# 7. CROSSBAR LAYERS - MEMRISTOR SIMULATION
# =============================================================================

class CrossbarLinear(nn.Module):
    """
    Linear layer with memristor crossbar simulation.

    Simulates:
    - Weight quantization (memristor conductance levels)
    - DAC/ADC quantization
    - Device variation (manufacturing imperfections)
    - Read noise (cycle-to-cycle variation)
    - Wire resistance (IR drop)

    References:
    - Cai+ Nature Electronics 2019
    - Yao+ Nature 2020
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02, wire_resistance=0.0):
        super().__init__()

        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.device_variation = device_variation
        self.wire_resistance = wire_resistance

        # Learnable weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Fixed device variation (manufacturing, doesn't change)
        self.register_buffer('device_var_mask',
            torch.randn(out_features, in_features) * device_variation)

    def quantize_weights(self, w):
        """Quantize weights with STE."""
        if self.weight_bits >= 32:
            return w

        n_levels = 2 ** self.weight_bits
        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8

        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min

        return w + (w_quant - w).detach()

    def add_device_variation(self, w):
        """Add fixed device variation."""
        if self.device_variation > 0:
            return w * (1 + self.device_var_mask)
        return w

    def add_read_noise(self, w):
        """Add stochastic read noise (only during training)."""
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(w) * self.noise_std
            return w * (1 + noise)
        return w

    def dac_quantize(self, x):
        """DAC: digital to analog conversion."""
        if self.dac_bits >= 32:
            return x

        n_levels = 2 ** self.dac_bits
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min + 1e-8

        x_normalized = (x - x_min) / x_range
        x_quant_norm = torch.round(x_normalized * (n_levels - 1)) / (n_levels - 1)
        x_quant = x_quant_norm * x_range + x_min

        return x + (x_quant - x).detach()

    def adc_quantize(self, y):
        """ADC: analog to digital conversion."""
        if self.adc_bits >= 32:
            return y

        n_levels = 2 ** self.adc_bits
        y_min = y.min()
        y_max = y.max()
        y_range = y_max - y_min + 1e-8

        y_normalized = (y - y_min) / y_range
        y_quant_norm = torch.round(y_normalized * (n_levels - 1)) / (n_levels - 1)
        y_quant = y_quant_norm * y_range + y_min

        return y + (y_quant - y).detach()

    def forward(self, x):
        # 1. DAC: Quantize input
        x_dac = self.dac_quantize(x)

        # 2. Weight quantization
        w_quant = self.quantize_weights(self.weight)

        # 3. Device variation
        w_varied = self.add_device_variation(w_quant)

        # 4. Read noise
        w_noisy = self.add_read_noise(w_varied)

        # 5. Analog MVM (this is where memristor magic happens!)
        y = F.linear(x_dac, w_noisy, None)

        # 6. ADC: Quantize output
        y_adc = self.adc_quantize(y)

        # 7. Bias (digital)
        if self.bias is not None:
            y_adc = y_adc + self.bias

        return y_adc


class CrossbarLIFLinear(NorseLIFBase):
    """
    Norse LIF neuron with memristor crossbar for synaptic weights.
    Combines CrossbarLinear + Norse LIF dynamics.
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02, **kwargs):

        # Extract crossbar-specific params
        kwargs_norse = {k: v for k, v in kwargs.items()
                       if k not in ['weight_bits', 'adc_bits', 'dac_bits',
                                   'noise_std', 'device_variation', 'wire_resistance']}

        super().__init__(output_features=out_features, **kwargs_norse)

        self.crossbar = CrossbarLinear(
            in_features, out_features, bias=bias,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )

    def current(self, x):
        return self.crossbar(x)

print("✓ Crossbar layers defined")


# === cell #7 ===
# =============================================================================
# 8. VAE ARCHITECTURES - CLASSICAL BASELINE
# =============================================================================

class VAE_Encoder(nn.Module):
    """Classical VAE Encoder: input → hidden → (mean, log_var)"""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)
        return mean, log_var


class VAE_Decoder(nn.Module):
    """Classical VAE Decoder: latent → hidden → output"""

    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return self.fc2(h)


class VAE_QST(nn.Module):
    """
    Variational Autoencoder for Quantum State Tomography.

    Architecture:
    - Encoder: measurement → hidden → (μ, σ²) → z
    - Decoder: z → hidden → density matrix elements
    - DensityMap: elements → valid ρ via AA†/Tr(AA†)

    Args:
        cond_dim: Input dimension (M1: 12, M2: 3×2^N)
        d: Hilbert space dimension (2^N)
        hidden_dim: Hidden layer size
        latent_dim: Latent space dimension
    """

    def __init__(self, cond_dim, d, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        output_dim = 2 * d * d  # Real + Imag parts

        self.encoder = VAE_Encoder(cond_dim, hidden_dim, latent_dim)
        self.decoder = VAE_Decoder(latent_dim, hidden_dim, output_dim)
        self.dm = DensityMap()

        # Initialize
        self.apply(init_normal_002)

    def reparameterize(self, mean, log_var):
        """Reparameterization trick: z = μ + σ * ε, ε ~ N(0,1)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        # Encode
        mean, log_var = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mean, log_var)

        # Decode
        out = self.decoder(z)

        # Reshape to [B, 2, d, d]
        out = out.view(-1, 2, self.d, self.d)

        # Map to valid density matrix
        rho = self.dm(out)

        return rho, mean, log_var

    def loss_function(self, rho_pred, rho_true, mean, log_var, beta=1.0):
        """
        VAE Loss = Reconstruction Loss + β × KL Divergence
        """
        # Reconstruction loss (MSE on density matrix elements)
        recon_loss = F.mse_loss(rho_pred, rho_true)

        # KL divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        return recon_loss + beta * kl_loss, recon_loss, kl_loss

print("✓ Classical VAE architecture defined")


# === cell #8 ===
# =============================================================================
# 9. SVAE ARCHITECTURES - SPIKING VAE WITH NORSE LIF
# =============================================================================

class SVAE_Encoder_Norse(nn.Module):
    """
    Spiking VAE Encoder using Norse LIF neurons.
    Outputs spike rates that encode mean and log_var.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, enc_mode='poisson', enc_gamma=1.5):
        super().__init__()

        kw = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                  v_th=v_th, return_rate=True, weight_bits=weight_bits,
                  enc_mode=enc_mode, enc_gamma=enc_gamma)

        self.lif1 = NorseLIFLinear(input_dim, hidden_dim, **kw)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # Spiking hidden layer
        h = self.lif1(x)  # Returns spike rate

        # Linear readout for mean/logvar
        mean = self.fc_mean(h)
        log_var = self.fc_logvar(h)

        return mean, log_var


class SVAE_Decoder_Norse(nn.Module):
    """
    Spiking VAE Decoder using Norse LIF neurons.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, enc_mode='poisson', enc_gamma=1.5):
        super().__init__()

        kw = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                  v_th=v_th, return_rate=True, weight_bits=weight_bits,
                  enc_mode=enc_mode, enc_gamma=enc_gamma)

        self.lif1 = NorseLIFLinear(latent_dim, hidden_dim, **kw)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.lif1(z)
        return self.fc_out(h)


class SVAE_QST_Norse(nn.Module):
    """
    Spiking Variational Autoencoder for QST using Norse LIF.
    Target hardware: Intel Loihi with 8-bit weight quantization.

    Args:
        cond_dim: Input dimension
        d: Hilbert space dimension
        hidden_dim: Hidden layer size
        latent_dim: Latent space dimension
        T: Number of timesteps
        tau_mem_inv: Membrane decay rate
        tau_syn_inv: Synaptic decay rate
        v_th: Firing threshold
        weight_bits: Weight quantization (8 for Loihi)
        enc_mode: Spike encoding mode
        enc_gamma: Encoding sharpness
    """

    def __init__(self, cond_dim, d, hidden_dim=128, latent_dim=32,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        output_dim = 2 * d * d

        enc_kw = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, weight_bits=weight_bits, enc_mode=enc_mode,
                      enc_gamma=enc_gamma)

        self.encoder = SVAE_Encoder_Norse(cond_dim, hidden_dim, latent_dim, **enc_kw)
        self.decoder = SVAE_Decoder_Norse(latent_dim, hidden_dim, output_dim, **enc_kw)
        self.dm = DensityMap()

        # Initialize weights only. warm_init_spiking is called once by
        # optimize_single_state(warm_spiking=True), NOT here, to avoid
        # double-scaling weights (5x * 5x = 25x would saturate all neurons).
        self.apply(init_normal_002)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        out = out.view(-1, 2, self.d, self.d)
        rho = self.dm(out)
        return rho, mean, log_var

    def loss_function(self, rho_pred, rho_true, mean, log_var, beta=1.0):
        recon_loss = F.mse_loss(rho_pred, rho_true)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

print("\u2713 SVAE Norse (Loihi) architecture defined")


# === cell #9 ===
# =============================================================================
# 10. SVAE ARCHITECTURES - CROSSBAR (MEMRISTOR)
# =============================================================================

class SVAE_Encoder_Crossbar(nn.Module):
    """
    Spiking VAE Encoder with memristor crossbar.
    """

    def __init__(self, input_dim, hidden_dim, latent_dim,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()

        kw = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                  v_th=v_th, return_rate=True,
                  weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                  noise_std=noise_std, device_variation=device_variation,
                  enc_mode=enc_mode, enc_gamma=enc_gamma)

        self.lif1 = CrossbarLIFLinear(input_dim, hidden_dim, **kw)

        # Readout layers also use crossbar
        self.crossbar_mean = CrossbarLinear(
            hidden_dim, latent_dim,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )
        self.crossbar_logvar = CrossbarLinear(
            hidden_dim, latent_dim,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )

    def forward(self, x):
        h = self.lif1(x)
        mean = self.crossbar_mean(h)
        log_var = self.crossbar_logvar(h)
        return mean, log_var


class SVAE_Decoder_Crossbar(nn.Module):
    """
    Spiking VAE Decoder with memristor crossbar.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()

        kw = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                  v_th=v_th, return_rate=True,
                  weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                  noise_std=noise_std, device_variation=device_variation,
                  enc_mode=enc_mode, enc_gamma=enc_gamma)

        self.lif1 = CrossbarLIFLinear(latent_dim, hidden_dim, **kw)

        self.crossbar_out = CrossbarLinear(
            hidden_dim, output_dim,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )

    def forward(self, z):
        h = self.lif1(z)
        return self.crossbar_out(h)


class SVAE_QST_Crossbar(nn.Module):
    """
    Spiking VAE for QST with memristor crossbar simulation.
    Target hardware: Memristor-based CiM accelerator.

    Args:
        cond_dim: Input dimension
        d: Hilbert space dimension
        hidden_dim: Hidden layer size
        latent_dim: Latent space dimension
        T: Number of timesteps
        weight_bits: Weight quantization (8 or 4)
        adc_bits: ADC resolution
        dac_bits: DAC resolution
        noise_std: Read noise std
        device_variation: Manufacturing variation
    """

    def __init__(self, cond_dim, d, hidden_dim=128, latent_dim=32,
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        self.d = d
        self.latent_dim = latent_dim
        self.weight_bits = weight_bits
        output_dim = 2 * d * d

        enc_kw = dict(
            T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv, v_th=v_th,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation,
            enc_mode=enc_mode, enc_gamma=enc_gamma
        )

        self.encoder = SVAE_Encoder_Crossbar(cond_dim, hidden_dim, latent_dim, **enc_kw)
        self.decoder = SVAE_Decoder_Crossbar(latent_dim, hidden_dim, output_dim, **enc_kw)
        self.dm = DensityMap()

        # Note: init_normal_002 and warm_init_spiking only affect nn.Linear
        # layers, which CrossbarLinear is not (it uses nn.Parameter directly).
        # These calls are kept for consistency but are effectively no-ops here.
        self.apply(init_normal_002)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        out = out.view(-1, 2, self.d, self.d)
        rho = self.dm(out)
        return rho, mean, log_var

    def loss_function(self, rho_pred, rho_true, mean, log_var, beta=1.0):
        recon_loss = F.mse_loss(rho_pred, rho_true)
        kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

print("\u2713 SVAE Crossbar (Memristor) architecture defined")


# === cell #10 ===
# =============================================================================
# 11. TRAINING FUNCTIONS
# =============================================================================

def optimize_single_state(model_class, N, method='M1', M=12, steps=400,
                          lr=1e-3, beta_vae=1.0, warm_spiking=False,
                          use_amp=True, log_interval=100, p=0.5):
    """
    Train a VAE/SVAE model to reconstruct a single GHZ state.

    Args:
        model_class: Model constructor function
        N: Number of qubits
        method: 'M1' (expectation) or 'M2' (probability)
        M: Number of measurements
        steps: Training steps
        lr: Learning rate
        beta_vae: KL divergence weight
        warm_spiking: Apply warm initialization for spiking models
        use_amp: Use automatic mixed precision
        log_interval: Logging frequency

    Returns:
        dict with results
    """
    d = 2**N

    # Create target state: Mixed GHZ (Werner state, p=0.5)
    rho_np = mixed_ghz_state(N, p=p)
    rho_ri = torch.stack([torch.from_numpy(rho_np.real).float(),
                          torch.from_numpy(rho_np.imag).float()], dim=0)
    rho_ri = rho_ri.unsqueeze(0).to(DEVICE)  # [1, 2, d, d]

    # Setup measurement operators/bases
    if method == 'M1':
        ops = select_paulis_nonzero(rho_np, N, M)
        ops_np = np.stack(ops)
        ops_ri = torch.stack([torch.from_numpy(ops_np.real).float(),
                              torch.from_numpy(ops_np.imag).float()], dim=1).to(DEVICE)
        exp_layer = ExpectationLayer()
        exp_layer.set_ops(ops_ri)
        exp_layer = exp_layer.to(DEVICE)

        # Calculate target expectations
        with torch.no_grad():
            cond = exp_layer(rho_ri)  # [1, M]
        cond_dim = M
        bases = None
    else:  # M2
        bases = select_bases_nonzero_M2(rho_np, N, M)
        with torch.no_grad():
            cond = probs_from_bases_torch(rho_ri, bases)  # [1, M*d]
        cond_dim = M * d
        exp_layer = None

    # Create model
    model = model_class(cond_dim, d).to(DEVICE)

    if warm_spiking:
        warm_init_spiking(model)

    # Optimizer and scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler() if use_amp and DEVICE.type == 'cuda' else None

    # Training loop
    F_hist = []
    loss_hist = []
    best_F = 0.0
    best_state = None

    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                rho_pred, mean, log_var = model(cond)
                loss, recon_loss, kl_loss = model.loss_function(
                    rho_pred, rho_ri, mean, log_var, beta=beta_vae)

            scaler.scale(loss).backward()
            # Gradient clipping to prevent NaN
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            rho_pred, mean, log_var = model(cond)
            loss, recon_loss, kl_loss = model.loss_function(
                rho_pred, rho_ri, mean, log_var, beta=beta_vae)
            loss.backward()
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluate fidelity
        model.eval()
        with torch.no_grad():
            rho_pred_eval, _, _ = model(cond)
            _, F_mean = fidelity_batch(rho_pred_eval, rho_ri)

        F_hist.append(F_mean)
        loss_hist.append(loss.item())

        if F_mean > best_F:
            best_F = F_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if step % log_interval == 0:
            print(f"    Step {step}: Loss={loss.item():.6f}, F={F_mean:.4f}")

    elapsed = time.time() - t0

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return {
        'model': model,
        'F_best': best_F,
        'F_hist': F_hist,
        'loss_hist': loss_hist,
        'time_sec': elapsed,
        'config': {'N': N, 'method': method, 'M': M, 'steps': steps}
    }

print("✓ Training functions defined")


# === cell #11 ===
# =============================================================================
# p-SWEEP RUNNER (mirrors run_vae_svae_benchmark, with p threaded in)
# =============================================================================

import gc

def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def werner_purity(p, d):
    """Tr(rho^2) for Werner state rho = p|GHZ><GHZ| + (1-p) I/d."""
    return p**2 + (1.0 - p)**2 / d + 2.0 * p * (1.0 - p) / d


def run_p_sweep_svae(p_list, N=5, methods=['M1', 'M2'],
                    M_M1=256, M_M2=4, steps=500, lr=1e-3,
                    crossbar_bits=[8, 4], T=8, sparsity=0.1, use_amp=True):
    """p-sweep matching the SVAE benchmark methodology exactly.

    Same model configs and per-N hyperparams as run_vae_svae_benchmark, but
    loops over p_list at fixed N instead of over N_list at fixed p=0.5.
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
    print(f"\n{'='*80}\nSVAE p-SWEEP: N={N}, T={T}, p_list={p_list}\n{'='*80}")

    # Hidden/latent dims fixed by N (same formulas as benchmark)
    hidden_dim = max(64, min(256, 32 * N))
    latent_dim = max(16, min(64, 8 * N))
    current_T = T
    current_vth = 1.0
    current_gamma = 1.5

    total = len(p_list) * len(methods)
    run_idx = 0

    for method in methods:
        M = M_M1 if method == 'M1' else M_M2
        cond_dim = M if method == 'M1' else M * d

        for p in p_list:
            run_idx += 1
            if (method, float(p)) in _done_outer:
                print(f"  [skip] seed={SEED} method={method} p={p:.3f}", flush=True)
                continue
            purity = werner_purity(p, d)
            print(f"\n[{run_idx}/{total}] method={method}  p={p:.3f}  purity={purity:.4f}  M={M}")

            # ---- 1. VAE (GPU) ----
            print("  [1/4] VAE")
            model_fn = lambda cd, d_: VAE_QST(
                cond_dim=cd, d=d_, hidden_dim=hidden_dim, latent_dim=latent_dim)
            res = optimize_single_state(
                model_class=model_fn, N=N, method=method, M=M, p=p,
                steps=steps, lr=lr, warm_spiking=False, use_amp=use_amp)
            E_inf = estimate_gpu_inference_energy(res['model'])
            rows.append({'seed': SEED, 'Model': 'VAE', 'Architecture': 'SVAE', 'HW': 'GPU',
                         'N': N, 'method': method, 'p': p, 'purity': purity,
                         'F_best': res['F_best'],
                         'F_last': float(res['F_hist'][-1]) if res['F_hist'] else res['F_best'],
                         'E_inference_uJ': E_inf['E_total_uJ'],
                         'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
            print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.2f} uJ")
            del res; gpu_cleanup()

            # ---- 2. SVAE-Norse (Loihi) ----
            print("  [2/4] SVAE-Norse")
            model_fn = lambda cd, d_: SVAE_QST_Norse(
                cond_dim=cd, d=d_, hidden_dim=hidden_dim, latent_dim=latent_dim,
                T=current_T, v_th=current_vth, enc_gamma=current_gamma)
            res = optimize_single_state(
                model_class=model_fn, N=N, method=method, M=M, p=p,
                steps=steps, lr=lr, beta_vae=0.1,
                warm_spiking=True, use_amp=use_amp)
            E_inf = estimate_loihi_inference_energy(res['model'], T=current_T, sparsity=sparsity)
            rows.append({'seed': SEED, 'Model': 'SVAE-Norse', 'Architecture': 'SVAE', 'HW': 'Loihi',
                         'N': N, 'method': method, 'p': p, 'purity': purity,
                         'F_best': res['F_best'],
                         'F_last': float(res['F_hist'][-1]) if res['F_hist'] else res['F_best'],
                         'E_inference_uJ': E_inf['E_total_uJ'],
                         'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
            print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f} uJ")
            del res; gpu_cleanup()

            # ---- 3,4. SVAE-Crossbar-{8,4}b ----
            for ci, bits in enumerate(crossbar_bits, start=3):
                # Same crossbar hyperparams as benchmark (N=5 branch -> N<=6)
                noise_std, device_var = 0.001, 0.003
                cb_T = 8
                if bits == 8:
                    cb_vth, cb_gamma = 0.3, 3.0
                else:  # 4-bit
                    cb_vth, cb_gamma = 0.3, 3.5
                print(f"  [{ci}/4] SVAE-Crossbar-{bits}b  (v_th={cb_vth}, gamma={cb_gamma})")
                _bits, _T, _vth, _gamma, _ns, _dv = bits, cb_T, cb_vth, cb_gamma, noise_std, device_var
                model_fn = lambda cd, d_: SVAE_QST_Crossbar(
                    cond_dim=cd, d=d_, hidden_dim=hidden_dim, latent_dim=latent_dim,
                    T=_T, v_th=_vth, enc_gamma=_gamma,
                    weight_bits=_bits, adc_bits=_bits, dac_bits=_bits,
                    noise_std=_ns, device_variation=_dv)
                res = optimize_single_state(
                    model_class=model_fn, N=N, method=method, M=M, p=p,
                    steps=steps, lr=lr, beta_vae=0.1,
                    warm_spiking=True, use_amp=use_amp)
                E_inf = estimate_crossbar_inference_energy(res['model'], bits=bits, T=current_T, sparsity=sparsity)
                rows.append({'seed': SEED, 'Model': f'SVAE-Crossbar-{bits}b', 'Architecture': 'SVAE',
                             'HW': f'Crossbar-{bits}b', 'N': N, 'method': method, 'p': p, 'purity': purity,
                             'F_best': res['F_best'],
                             'F_last': float(res['F_hist'][-1]) if res['F_hist'] else res['F_best'],
                             'E_inference_uJ': E_inf['E_total_uJ'],
                             'n_params': E_inf['n_params'], 'time_sec': res['time_sec']})
                print(f"      F_best={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f} uJ")
                del res; gpu_cleanup()

            pd.DataFrame(rows).to_csv(CSV_PATH, index=False)  # multiseed: save after each (method, p)

    return pd.DataFrame(rows)


print("Defined: run_p_sweep_svae")


# === cell #12 ===
# =============================================================================
# p-SWEEP CONFIGURATION
# =============================================================================
P_LIST = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
N_FIXED = 5
T_FIXED = 8
STEPS = 500
LR = 1e-3
print(f"p_list = {P_LIST}")
print(f"N = {N_FIXED}, T = {T_FIXED}, steps = {STEPS}")
print(f"Models per (p, method): 4  (VAE, SVAE-Norse, SVAE-Crossbar-8b, SVAE-Crossbar-4b)")
print(f"Total runs: {len(P_LIST)} p-values * 2 methods * 4 models = {len(P_LIST)*2*4}")


# === cell #13 ===
# =============================================================================
# RUN p-SWEEP (M1 + M2 in one pass)
# =============================================================================
df_p_sweep = run_p_sweep_svae(
    p_list=P_LIST, N=N_FIXED, methods=['M1', 'M2'],
    M_M1=256, M_M2=4,
    steps=STEPS, lr=LR,
    crossbar_bits=[8, 4],
    T=T_FIXED, sparsity=0.1, use_amp=True)

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
COLORS = {'VAE':'#1f77b4', 'SVAE-Norse':'#2ca02c',
          'SVAE-Crossbar-8b':'#ff7f0e', 'SVAE-Crossbar-4b':'#d62728'}
MARKERS = {'VAE':'o', 'SVAE-Norse':'s',
           'SVAE-Crossbar-8b':'P', 'SVAE-Crossbar-4b':'X'}

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
    ax.set_title(f'SVAE p-sweep — {method} (N={N_FIXED}, T={T_FIXED})', fontsize=13)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='F=0.8')
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.4, label='Benchmark p=0.5')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.05, 1.05)
plt.tight_layout()
plt.savefig('p_sweep_SVAE_fidelity_vs_p.png', dpi=150, bbox_inches='tight')
plt.savefig('p_sweep_SVAE_fidelity_vs_p.pdf', bbox_inches='tight')
print("Saved: p_sweep_SVAE_fidelity_vs_p.png/pdf")

