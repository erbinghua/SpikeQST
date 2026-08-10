#!/usr/bin/env python3
"""AUTO-GENERATED from SCNN_GHZ_Mixed_V1.ipynb.

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
"""
Quantum State Tomography: CNN vs SCNN Energy Benchmark - Mixed GHZ States
==========================================================================
Confronto energetico completo tra CNN (GPU) e SCNN (neuromorphic) per
quantum state tomography su stati GHZ MISTI con depolarizing noise.

Werner state (Eq. A4, Hua et al.): rho = p |GHZ><GHZ| + (1-p) I/2^N
with p = 0.5 (pure-state weight)

Optimal hyperparameters from sweep studies:
  - T = 8 SNN timesteps (T-sweep: sweet spot, energy scales linearly)
  - M1: 256 Pauli operators (M-sweep: saturates at F~0.83 for N=8)
  - M2: 4 measurement bases (M-sweep: sufficient for Norse/Crossbar-4b)
  - Norse: v_th=0.9, enc_gamma=1.5, tau_mem=100, tau_syn=200
  - Crossbar-8b: v_th=0.5, enc_gamma=2.25
  - Crossbar-4b: v_th=0.3, enc_gamma=3.0
  - Capacity: ch=(32,64), depth=2 conv layers

Test:
- Metodi: M1 (expectation values) e M2 (probability distributions)
- Scaling: N=3,4,5,6,7,8 qubit
- 4 architetture: CNN-Simple2D, CNN-Up2D, SCNN-Simple2D, SCNN-Up2D
- Target: Mixed GHZ states (Werner state, p=0.5, Eq. A4 Hua et al.)

Riferimenti:
- Davies+ "Loihi" IEEE Micro 2018
- Horowitz "Computing's Energy Problem" ISSCC 2014
"""


# === cell #1 ===
# =============================================================================
# 1. SETUP & IMPORTS - GPU OPTIMIZED
# =============================================================================
import os, math, random, time, numpy as np, gc
# Anti-fragmentation: must be set BEFORE any CUDA call
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from collections import defaultdict
import pickle
from pathlib import Path

plt.rcParams['figure.figsize'] = (5.2, 4.2)

# GPU Setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-ottimizzazione
    torch.backends.cudnn.deterministic = False  # Per velocità

# Checkpoint directory
# (multiseed: redirected to per-seed OUT_DIR)
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "results.csv"
print("="*70)
print("🚀 QUANTUM STATE TOMOGRAPHY: GPU-OPTIMIZED BENCHMARK (N=3-10)")
print("="*70)
print(f"✓ PyTorch {torch.__version__}")
print(f"✓ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ cuDNN: {torch.backends.cudnn.version()}")
    print(f"✓ Mixed Precision: Enabled")
print(f"✓ Checkpoint Dir: {CHECKPOINT_DIR}")


# === cell #2 ===
# Norse imports (dopo torch imports)
import norse.torch as norse
from norse.torch import LIFParameters


# === cell #3 ===
# =============================================================================
# CELLA 1: PARAMETRI ENERGETICI CORRETTI (sostituisce la cella esistente)
# =============================================================================

"""
# =============================================================================
# PARAMETRI ENERGETICI HARDWARE (CORRETTI E VERIFICATI)
# =============================================================================

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
    'util_small_model': 0.05,     # 5% utilization for small models (<100k params)
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
    'E_MAC_analog': 2e-15,        # 2 fJ/MAC (resistive MVM) - quasi gratis!
    'E_ADC_8bit': 20e-12,         # 20 pJ/conversion (8-bit SAR)
    'E_ADC_4bit': 5e-12,          # 5 pJ/conversion (4-bit)
    'E_DAC_8bit': 10e-12,         # 10 pJ/conversion (8-bit)
    'E_DAC_4bit': 2.5e-12,        # 2.5 pJ/conversion (4-bit)
    'E_spike_gen': 3e-12,         # 3 pJ/spike (LIF analog circuit)
    'E_write': 100e-12,           # 100 pJ/weight write (RRAM program) ⚠️
    'E_leakage': 0.1e-12,         # 0.1 pJ/neuron/timestep
}

print("="*70)
print("📊 PARAMETRI ENERGETICI HARDWARE")
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
print(f"   E_MAC_analog = {MEMRISTOR_PARAMS['E_MAC_analog']*1e15:.0f} fJ (17500× < GPU!)")
print(f"   E_ADC_8bit = {MEMRISTOR_PARAMS['E_ADC_8bit']*1e12:.0f} pJ ← DOMINANTE")
print(f"   E_DAC_8bit = {MEMRISTOR_PARAMS['E_DAC_8bit']*1e12:.0f} pJ")
print(f"   E_write = {MEMRISTOR_PARAMS['E_write']*1e12:.0f} pJ (weight programming)")


# === cell #4 ===
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


# === cell #5 ===
# =============================================================================
# 3. FUNZIONI STIMA ENERGIA (CORRETTE - USA CHIAVI DA CELLA 4)
# =============================================================================

def estimate_cnn_energy(model, input_shape, steps=1):
    """
    Stima energia per CNN/ANN su GPU.

    USA le chiavi corrette da GPU_PARAMS (Cella 4):
    - E_MAC (non E_per_MAC)
    - E_DRAM (non E_DRAM_access)
    - E_L2_cache (non E_cache_access)
    """
    try:
        from torchinfo import summary
        with torch.no_grad():
            stats = summary(model, input_size=(1, *input_shape), verbose=0)
        n_macs = stats.total_mult_adds
    except Exception as e:
        print(f"WARNING estimate_cnn_energy: torchinfo failed ({e}), using parameter-based estimate")
        n_params = sum(p.numel() for p in model.parameters())
        n_macs = n_params * 2  # Approssimazione: 2 ops per peso

    # Energia computazione
    # Per modelli piccoli (<100k params), GPU è sottoutilizzata
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 100_000:
        utilization = 0.05
    elif n_params < 1_000_000:
        utilization = 0.15
    else:
        utilization = 0.30

    E_compute = n_macs * GPU_PARAMS['E_MAC']

    # Energia memoria (weighted by cache hit rate)
    n_memory_accesses = n_params * 2
    cache_hit_rate = GPU_PARAMS['cache_hit_rate']  # 0.85

    E_memory = (
        n_memory_accesses * (1 - cache_hit_rate) * GPU_PARAMS['E_DRAM'] +
        n_memory_accesses * cache_hit_rate * GPU_PARAMS['E_L2_cache']
    )

    E_total = (E_compute + E_memory) * steps

    return E_total, {
        'n_macs': n_macs,
        'n_params': n_params,
        'E_compute': E_compute,
        'E_memory': E_memory,
        'E_MAC': GPU_PARAMS['E_MAC'],
        'utilization': utilization
    }


def estimate_memristor_snn_energy(model, T_spike=8, sparsity=0.1, bits=8, steps=1):
    """
    Stima energia per SNN su memristor crossbar.

    USA le chiavi corrette da MEMRISTOR_PARAMS (Cella 4):
    - E_MAC_analog (2 fJ)
    - E_ADC_8bit / E_ADC_4bit
    - E_DAC_8bit / E_DAC_4bit
    - E_spike_gen (3 pJ)
    - E_leakage (0.1 pJ)

    Modella correttamente:
    - Analog MVM (molto efficiente: ~2 fJ/MAC)
    - ADC/DAC overhead (dominante: 65-70%)
    - Spike generation
    """
    params = MEMRISTOR_PARAMS

    # Select ADC/DAC energy based on bits
    E_ADC = params['E_ADC_8bit'] if bits == 8 else params['E_ADC_4bit']
    E_DAC = params['E_DAC_8bit'] if bits == 8 else params['E_DAC_4bit']

    # ========== ARCHITETTURA ==========
    n_synapses = sum(p.numel() for p in model.parameters())

    # Conta neuroni per layer
    layer_info = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layer_info.append({
                'seed': SEED,
                'type': 'Linear',
                'in_features': m.in_features,
                'out_features': m.out_features,
                'n_synapses': m.in_features * m.out_features
            })
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fm_size = 16  # Assume 16×16 feature map
            layer_info.append({
                'seed': SEED,
                'type': 'Conv',
                'in_channels': m.in_channels,
                'out_channels': m.out_channels,
                'n_synapses': m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1],
                'n_neurons': m.out_channels * fm_size
            })

    # Totali
    n_neurons = sum(l.get('out_features', l.get('n_neurons', 0)) for l in layer_info)
    n_layers = len(layer_info)
    total_in = sum(l.get('in_features', l.get('in_channels', 0) * 16) for l in layer_info)
    total_out = sum(l.get('out_features', l.get('n_neurons', 0)) for l in layer_info)

    # ========== 1. SPIKE GENERATION ==========
    n_spikes = n_neurons * T_spike * sparsity
    E_spikes = n_spikes * params['E_spike_gen']

    # ========== 2. ANALOG MVM (quasi gratis!) ==========
    n_mvm_ops = n_synapses * T_spike * sparsity
    E_analog_compute = n_mvm_ops * params['E_MAC_analog']

    # ========== 3. ADC (DOMINANTE!) ==========
    # ADC: ogni output neuron, ogni timestep (non dipende da sparsity input)
    n_adc_conversions = total_out * T_spike
    E_adc = n_adc_conversions * E_ADC

    # ========== 4. DAC ==========
    # DAC: solo input attivi
    n_dac_conversions = total_in * T_spike * sparsity
    E_dac = n_dac_conversions * E_DAC

    # ========== 5. LEAKAGE ==========
    E_leak = n_neurons * T_spike * params['E_leakage']

    # ========== TOTALE ==========
    E_total_per_inference = E_analog_compute + E_adc + E_dac + E_spikes + E_leak
    E_total = E_total_per_inference * steps

    # Breakdown percentages
    breakdown = {
        'analog_MVM_%': 100 * E_analog_compute / E_total_per_inference,
        'ADC_%': 100 * E_adc / E_total_per_inference,
        'DAC_%': 100 * E_dac / E_total_per_inference,
        'spikes_%': 100 * E_spikes / E_total_per_inference,
        'leak_%': 100 * E_leak / E_total_per_inference,
    }

    return E_total, {
        'n_synapses': n_synapses,
        'n_neurons': n_neurons,
        'n_layers': n_layers,
        'T_spike': T_spike,
        'sparsity': sparsity,
        'bits': bits,
        'E_analog_compute': E_analog_compute,
        'E_ADC': E_adc,
        'E_DAC': E_dac,
        'E_spikes': E_spikes,
        'E_leak': E_leak,
        'breakdown': breakdown,
    }


def estimate_write_energy(model):
    """Stima energia per programmare i pesi nel crossbar (one-time)."""
    n_params = sum(p.numel() for p in model.parameters())
    E_total = n_params * MEMRISTOR_PARAMS['E_write']
    return E_total, {'n_params': n_params, 'E_per_weight': MEMRISTOR_PARAMS['E_write']}


print("✓ Funzioni stima energia CORRETTE caricate")
print(f"  GPU: E_MAC = {GPU_PARAMS['E_MAC']*1e12:.0f} pJ")
print(f"  Crossbar 8-bit: E_ADC = {MEMRISTOR_PARAMS['E_ADC_8bit']*1e12:.0f} pJ, E_DAC = {MEMRISTOR_PARAMS['E_DAC_8bit']*1e12:.0f} pJ")
print(f"  Crossbar 4-bit: E_ADC = {MEMRISTOR_PARAMS['E_ADC_4bit']*1e12:.0f} pJ, E_DAC = {MEMRISTOR_PARAMS['E_DAC_4bit']*1e12:.0f} pJ")


# === cell #6 ===
# =============================================================================
# 4-7. UTILITIES (compatte)
# =============================================================================

def pauli_I(): return np.array([[1,0],[0,1]], dtype=np.complex128)
def pauli_X(): return np.array([[0,1],[1,0]], dtype=np.complex128)
def pauli_Y(): return np.array([[0,-1j],[1j,0]], dtype=np.complex128)
def pauli_Z(): return np.array([[1,0],[0,-1]], dtype=np.complex128)

PAULI = {'I':pauli_I(), 'X':pauli_X(), 'Y':pauli_Y(), 'Z':pauli_Z()}

def kron_n(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

def all_pauli_strings(N, include_I=True):
    import itertools
    letters = ['I','X','Y','Z'] if include_I else ['X','Y','Z']
    return [''.join(p) for p in itertools.product(letters, repeat=N)]

def pauli_op(string):
    return kron_n([PAULI[c] for c in string]).astype(np.complex128)

def ghz_state(N):
    d = 2**N
    v0 = np.zeros((d,1), dtype=np.complex128); v0[0,0]=1.0
    v1 = np.zeros((d,1), dtype=np.complex128); v1[-1,0]=1.0
    return (v0+v1)/np.sqrt(2)

def density_from_ket(psi):
    rho = psi @ psi.conj().T
    rho = 0.5*(rho + rho.conj().T)
    tr = np.real(np.trace(rho))
    return (rho / tr).astype(np.complex128)

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
    psi = ghz_state(N)
    rho_pure = density_from_ket(psi)
    I_d = np.eye(d, dtype=np.complex128) / d
    rho = p * rho_pure + (1 - p) * I_d
    rho = 0.5 * (rho + rho.conj().T)
    return (rho / np.real(np.trace(rho))).astype(np.complex128)

def compute_expectations(rho, ops):
    return np.real(np.einsum('ij,mji->m', rho, np.stack(ops,0))).astype(np.float32)

def select_ops_nonzero_M1(rho, M, N, tries=8, tol=1e-12):
    strings_all = all_pauli_strings(N, include_I=False)
    L = len(strings_all)
    rng = np.random.default_rng(1234)
    chosen, seen = [], set()

    # Phase 1: pick operators with non-zero expectation value
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

    # Phase 2: fill remaining with zero-expectation operators (no duplicates)
    if len(chosen) < M:
        for s in strings_all:
            if s not in seen:
                chosen.append(pauli_op(s)); seen.add(s)
                if len(chosen) >= M: break

    # Cap at min(M, total available) to avoid requesting more than exist
    return chosen[:min(M, L)]

def _eigenbasis_1q(axis, device=None, dtype=torch.complex64):
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
    out = mats[0]
    for m in mats[1:]:
        out = torch.kron(out, m)
    return out

def select_bases_nonzero_M2(rho, N, M, seed=0):
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



def probs_from_bases_torch(rho_ri, bases):
    """Fixed: handles AMP correctly."""
    device, B, _, d, _ = rho_ri.device, *rho_ri.shape
    N = int(math.log2(d))

    # FIX: Disable AMP for complex operations
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


class DensityMap(nn.Module):
    """Fixed: handles AMP correctly."""
    def forward(self, aa_ri):
        Ar, Ai = aa_ri[:,0], aa_ri[:,1]

        # FIX: Disable AMP for complex operations
        with torch.amp.autocast('cuda', enabled=False):
            A = torch.complex(Ar.float(), Ai.float())
            M = A @ A.conj().transpose(-1,-2)
            M = 0.5*(M + M.conj().transpose(-1,-2))
            tr = torch.real(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)).clamp_min(1e-12)
            M = M / tr.view(-1,1,1)
            result = torch.stack([M.real, M.imag], dim=1)

        return result


class ExpectationLayer(nn.Module):
    """Fixed: handles AMP correctly."""
    def __init__(self):
        super().__init__()
        self.register_buffer('Acmplx', None)

    @torch.no_grad()
    def set_ops(self, ops_ri_fixed):
        Ar = ops_ri_fixed[:, 0]
        Ai = ops_ri_fixed[:, 1]
        self.Acmplx = torch.complex(Ar.float(), Ai.float())

    def forward(self, rho_ri, ops_ri=None):
        # FIX: Disable AMP for complex operations
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


def fidelity_batch(rho_pred_ri, rho_true_ri, eps=1e-12):
    """Quantum fidelity with double-precision arithmetic.

    Uses a pure-state fast path when rho_true is pure (rank 1):
        F = Tr(rho_true @ rho_pred)
    This avoids eigendecomposition and eliminates the numerical bias
    caused by regularisation of near-zero eigenvalues in large Hilbert spaces.

    For mixed target states falls back to the general Uhlmann formula:
        F = [Tr(sqrt(sqrt(rho_true) @ rho_pred @ sqrt(rho_true)))]^2
    computed in complex128 with eigenvalues clamped to min=0 (not eps).
    """
    with torch.amp.autocast('cuda', enabled=False):
        c_dtype = torch.complex128
        rho_p = torch.complex(rho_pred_ri[:,0].double(), rho_pred_ri[:,1].double()).to(c_dtype)
        rho_t = torch.complex(rho_true_ri[:,0].double(), rho_true_ri[:,1].double()).to(c_dtype)

        d = rho_t.shape[-1]
        batch_size = rho_pred_ri.shape[0]

        # --- Pure-state fast path ---
        # A state is pure iff Tr(rho^2) = 1
        purity = torch.real(torch.diagonal(rho_t @ rho_t, dim1=-2, dim2=-1).sum(-1))
        is_pure = (purity > 1.0 - 1e-6)

        if is_pure.all():
            # F = Tr(rho_true @ rho_pred) — exact for pure target states
            fid = torch.real(torch.diagonal(rho_t @ rho_p, dim1=-2, dim2=-1).sum(-1))
            fid = torch.clamp(fid, min=0.0)
            if (fid > 1.0 + 1e-6).any():
                print(f"WARNING fidelity_batch: pure-state fidelity "
                      f"{fid.max().item():.6f} > 1 (d={d}), possible numerical issue")
            fid = torch.clamp(fid, max=1.0)
            return fid, fid.mean().item()

        # --- General Uhlmann fidelity (mixed states) ---
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
                print(f"WARNING fidelity_batch: sqrtm_psd returned None (d={d}), returning fid=0")
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


# === cell #7 ===
# =============================================================================
# 8-10. SPIKING NEURONS
# =============================================================================

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1.0 / (1.0 + torch.abs(x)))

def spike_fn(x): return SurrogateHeaviside.apply(x)

def _stdz(x, eps=1e-6):
    dims = tuple(range(1, x.dim()))
    mu, sd = x.mean(dim=dims, keepdim=True), x.std(dim=dims, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def _poisson_st(x, gamma=1.5, pmin=0.02, pmax=0.98):
    z = _stdz(x)
    p = torch.sigmoid(gamma * z).clamp(pmin, pmax)
    s = (torch.rand_like(p) < p).float()
    return s + (p - p.detach())

class LIFBase(nn.Module):
    def __init__(self, T=8, beta=0.95, v_th=0.3, return_rate=True,
                 enc_mode='poisson', enc_gamma=1.5, enc_pmin=0.02, enc_pmax=0.98):
        super().__init__()
        self.T, self.beta, self.v_th = T, beta, v_th
        self.return_rate, self.enc_mode = return_rate, enc_mode
        self.enc_gamma, self.enc_pmin, self.enc_pmax = enc_gamma, enc_pmin, enc_pmax

    def current(self, x): raise NotImplementedError
    def _encode_x(self, x):
        return _poisson_st(x, self.enc_gamma, self.enc_pmin, self.enc_pmax) if self.enc_mode == 'poisson' else x

    def forward(self, x):
        v = acc = None
        for _ in range(self.T):
            I = self.current(self._encode_x(x))
            if v is None: v = torch.zeros_like(I); acc = torch.zeros_like(I)
            v = self.beta * v + I
            s = spike_fn(v - self.v_th)
            v = v - s * self.v_th
            acc = acc + s
        return acc / float(self.T) if self.return_rate else v

class LIFLinear(LIFBase):
    def __init__(self, in_features, out_features, bias=True, **kw):
        super().__init__(**kw)
        self.fc = nn.Linear(in_features, out_features, bias=bias)
    def current(self, x): return self.fc(x)

class LIFConv2d(LIFBase):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False, **kw):
        super().__init__(**kw)
        self.cv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
    def current(self, x): return self.cv(x)

class LIFConvT2d(LIFBase):
    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False, **kw):
        super().__init__(**kw)
        self.cvt = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
    def current(self, x): return self.cvt(x)

def init_normal_002(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)

def warm_init_spiking(model, w_scale=3.0, bias=0.1):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.mul_(w_scale)
                if getattr(m, 'bias', None) is not None:
                    m.bias.add_(bias)


# === cell #8 ===
# =============================================================================
# 8-10. NORSE LIF NEURONS (Hardware-Realistic) - WITH LOIHI 8-BIT QUANTIZATION
# =============================================================================

# Keep surrogate and encoding functions
class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (1.0 / (1.0 + torch.abs(x)))

def spike_fn(x): return SurrogateHeaviside.apply(x)

def _stdz(x, eps=1e-6):
    dims = tuple(range(1, x.dim()))
    mu, sd = x.mean(dim=dims, keepdim=True), x.std(dim=dims, keepdim=True).clamp_min(eps)
    return (x - mu) / sd

def _poisson_st(x, gamma=1.5, pmin=0.02, pmax=0.98):
    z = _stdz(x)
    p = torch.sigmoid(gamma * z).clamp(pmin, pmax)
    s = (torch.rand_like(p) < p).float()
    return s + (p - p.detach())


# =============================================================================
# NORSE LIF BASE CLASS - WITH LOIHI 8-BIT QUANTIZATION
# =============================================================================

class NorseLIFBase(nn.Module):
    """
    Base class for Norse LIF neurons with hardware-realistic dynamics.

    NOW INCLUDES: Loihi-style 8-bit weight quantization for fair hardware comparison.
    Intel Loihi uses 8-bit signed weights (actually 9-bit: 1 sign + 8 magnitude).
    Reference: Davies et al., IEEE Micro 2018 - 'Loihi: A Neuromorphic Manycore Processor'

    IMPORTANT: Norse uses tau_inv as direct decay rates, NOT 1/tau_milliseconds!
    - tau_mem_inv: membrane voltage decay rate (default Norse: 100.0)
    - tau_syn_inv: synaptic current decay rate (default Norse: 200.0)
    - v_th: firing threshold (default Norse: 1.0)

    These are dimensionless parameters in Norse's internal dynamics.
    """

    def __init__(self, output_features, T=8,
                 tau_mem_inv=100.0,  # Norse-style: direct decay rate
                 tau_syn_inv=200.0,  # Norse-style: direct decay rate
                 v_th=1.0,           # Norse-style: threshold
                 weight_bits=8,      # Loihi uses 8-bit weights
                 return_rate=True, enc_mode='poisson',
                 enc_gamma=1.5, enc_pmin=0.02, enc_pmax=0.98):
        super().__init__()
        self.T = T
        self.return_rate = return_rate
        self.enc_mode = enc_mode
        self.enc_gamma = enc_gamma
        self.enc_pmin = enc_pmin
        self.enc_pmax = enc_pmax
        self.weight_bits = weight_bits  # Loihi quantization

        # Store hardware parameters for documentation
        self.tau_mem_inv = tau_mem_inv
        self.tau_syn_inv = tau_syn_inv
        self.v_th = v_th

        # Norse LIF parameters (use values directly, not 1/tau_ms!)
        self.lif_params = LIFParameters(
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

        Loihi uses 8-bit signed weights. We simulate this with symmetric quantization.
        Forward: Real quantization (non-differentiable)
        Backward: Straight-through gradient (dw_quant/dw = 1)

        Reference: Davies et al., IEEE Micro 2018
        """
        if self.weight_bits >= 32:
            return w  # No quantization for FP32

        n_levels = 2 ** self.weight_bits

        # Symmetric quantization (Loihi uses signed weights)
        w_max = w.abs().max()
        if w_max == 0:
            return w

        # Scale to [-1, 1], quantize, scale back
        w_normalized = w / w_max
        w_quant_norm = torch.round(w_normalized * (n_levels // 2 - 1)) / (n_levels // 2 - 1)
        w_quant = w_quant_norm * w_max

        # Straight-Through Estimator
        return w + (w_quant - w).detach()

    def current(self, x):
        raise NotImplementedError

    def _encode_x(self, x):
        if self.enc_mode == 'poisson':
            return _poisson_st(x, self.enc_gamma, self.enc_pmin, self.enc_pmax)
        return x

    def forward(self, x):
        # Get the shape for state initialization
        I_sample = self.current(x)

        # Initialize state to None (Norse handles it)
        state = None
        acc = torch.zeros_like(I_sample)

        # Simulate T timesteps
        for t in range(self.T):
            x_encoded = self._encode_x(x)
            I_t = self.current(x_encoded)

            # Norse LIF dynamics
            spikes, state = self.lif_cell(I_t, state)

            acc = acc + spikes

        return acc / float(self.T) if self.return_rate else acc

    def get_hardware_params(self):
        """Return hardware parameters for Loihi implementation"""
        return {
            'tau_mem_inv': self.tau_mem_inv,
            'tau_syn_inv': self.tau_syn_inv,
            'v_threshold': self.v_th,
            'n_timesteps': self.T,
            'weight_bits': self.weight_bits,
            'neuron_model': 'LIF',
            'framework': 'Norse',
            'target_hardware': 'Intel Loihi'
        }


# =============================================================================
# NORSE LIF LAYERS - WITH LOIHI WEIGHT QUANTIZATION
# =============================================================================

class NorseLIFLinear(NorseLIFBase):
    """Norse LIF with Linear layer and Loihi 8-bit weight quantization"""
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(output_features=out_features, **kwargs)
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def current(self, x):
        # Apply Loihi-style weight quantization
        w_quant = self.quantize_weights(self.fc.weight)
        return F.linear(x, w_quant, self.fc.bias)


class NorseLIFConv2d(NorseLIFBase):
    """Norse LIF with Conv2d layer and Loihi 8-bit weight quantization"""
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False, **kwargs):
        super().__init__(output_features=c_out, **kwargs)
        self.cv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.stride = s
        self.padding = p

    def current(self, x):
        # Apply Loihi-style weight quantization
        w_quant = self.quantize_weights(self.cv.weight)
        return F.conv2d(x, w_quant, self.cv.bias, stride=self.stride, padding=self.padding)


class NorseLIFConvT2d(NorseLIFBase):
    """Norse LIF with ConvTranspose2d layer and Loihi 8-bit weight quantization"""
    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False, **kwargs):
        super().__init__(output_features=c_out, **kwargs)
        self.cvt = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.stride = s
        self.padding = p

    def current(self, x):
        # Apply Loihi-style weight quantization
        w_quant = self.quantize_weights(self.cvt.weight)
        return F.conv_transpose2d(x, w_quant, self.cvt.bias, stride=self.stride, padding=self.padding)


# Keep init functions unchanged
def init_normal_002(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)

def warm_init_spiking(model, w_scale=3.0, bias=0.1):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.mul_(w_scale)
                if getattr(m, 'bias', None) is not None:
                    m.bias.add_(bias)


# === cell #9 ===
# =============================================================================
# MEMRISTOR CROSSBAR SIMULATION - Core Components
# =============================================================================

class QuantizedLinear(nn.Module):
    """
    Simulates a memristor crossbar for matrix-vector multiplication.

    Features:
    - Quantization-Aware Training (QAT) with Straight-Through Estimator
    - Configurable bit-width for weights, ADC, DAC
    - Device variation (manufacturing imperfections)
    - Read noise (cycle-to-cycle variation)
    - Wire resistance effects (IR drop)
    - Energy breakdown calculation

    Args:
        in_features: Input dimension
        out_features: Output dimension
        weight_bits: Bit-width for weight quantization (default: 8)
        adc_bits: ADC resolution (default: 8)
        dac_bits: DAC resolution (default: 8)
        noise_std: Read noise standard deviation (default: 0.01)
        wire_resistance: Wire resistance factor 0-1 (default: 0.0)
        device_variation: Manufacturing variation std (default: 0.02)
        bias: Include bias term (default: False, computed digitally)
    """

    def __init__(self, in_features, out_features,
                 weight_bits=8,
                 adc_bits=8,
                 dac_bits=8,
                 noise_std=0.01,
                 wire_resistance=0.0,
                 device_variation=0.02,
                 bias=False):
        super().__init__()

        # Expose in_features/out_features so the energy estimator's _is_linear_like
        # duck-typing matches QuantizedLinear (it isn't an nn.Linear subclass).
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weights for training (FP32)
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Crossbar parameters
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.wire_resistance = wire_resistance
        self.device_variation = device_variation

        # Conductance range (typical RRAM from literature)
        # Reference: Cai+ Nature Electronics 2019
        self.G_on = 1e-4   # 10 kΩ (Low Resistance State)
        self.G_off = 1e-6  # 1 MΩ (High Resistance State)

        # For device variation (fixed at init, represents manufacturing)
        self.register_buffer('device_variation_mask', None)

    def _init_device_variation(self, shape):
        """Initialize device-to-device variation (one-time, manufacturing)"""
        if self.device_variation_mask is None and self.device_variation > 0:
            self.device_variation_mask = torch.randn(shape, device=self.weight.device) * self.device_variation

    def quantize_weights(self, w):
        """
        Quantize weights to n-bit resolution with Straight-Through Estimator.

        Forward: Real quantization (non-differentiable)
        Backward: Straight-through gradient (∂w_quant/∂w ≈ 1)
        """
        n_levels = 2 ** self.weight_bits

        # Normalize to [0, 1]
        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8
        w_normalized = (w - w_min) / w_range

        # Quantize to n_levels
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)

        # Scale back to original range
        w_quant = w_quant_norm * w_range + w_min

        # Straight-Through Estimator: forward uses quantized, backward uses identity
        return w + (w_quant - w).detach()

    def add_device_variation(self, w):
        """
        Add device-to-device variation (manufacturing imperfections).
        This is FIXED after manufacturing, so we use a buffer.
        """
        if self.device_variation > 0:
            # Initialize variation mask if needed
            self._init_device_variation(w.shape)
            # Apply multiplicative variation
            w_varied = w * (1.0 + self.device_variation_mask)
        else:
            w_varied = w
        return w_varied

    def add_read_noise(self, w):
        """
        Add read noise (cycle-to-cycle variation).
        This changes EVERY read, so we sample fresh noise.
        """
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(w) * self.noise_std
            w_noisy = w * (1.0 + noise)
        else:
            w_noisy = w
        return w_noisy

    def dac_quantize(self, x):
        """
        Simulate DAC quantization (digital → analog conversion).

        DAC converts digital input to analog voltage/current.
        Limited by bit-width resolution.
        """
        if self.dac_bits >= 32:
            return x  # No quantization for high precision

        n_levels = 2 ** self.dac_bits

        # Per-sample quantization (more realistic)
        x_min = x.min()
        x_max = x.max()
        x_range = x_max - x_min + 1e-8

        # Quantize
        x_normalized = (x - x_min) / x_range
        x_quant_norm = torch.round(x_normalized * (n_levels - 1)) / (n_levels - 1)
        x_quant = x_quant_norm * x_range + x_min

        # Straight-through estimator
        return x + (x_quant - x).detach()

    def adc_quantize(self, y):
        """
        Simulate ADC quantization (analog → digital conversion).

        ADC converts analog current to digital value.
        This is typically the DOMINANT energy consumer in memristor systems!
        """
        if self.adc_bits >= 32:
            return y  # No quantization for high precision

        n_levels = 2 ** self.adc_bits

        # Per-sample quantization
        y_min = y.min()
        y_max = y.max()
        y_range = y_max - y_min + 1e-8

        # Quantize
        y_normalized = (y - y_min) / y_range
        y_quant_norm = torch.round(y_normalized * (n_levels - 1)) / (n_levels - 1)
        y_quant = y_quant_norm * y_range + y_min

        # Straight-through estimator
        return y + (y_quant - y).detach()

    def apply_wire_resistance(self, y):
        """
        Simulate IR drop due to wire resistance in crossbar.

        In large crossbars, current must flow through metal wires.
        Ohmic losses cause voltage drop, reducing effective current.
        Effect is position-dependent (worse at far end of array).
        """
        if self.wire_resistance > 0:
            # Linear drop from 100% to (100% - wire_resistance%)
            n_out = y.size(-1)
            position_factor = torch.linspace(
                1.0,
                1.0 - self.wire_resistance,
                n_out,
                device=y.device
            )
            y = y * position_factor
        return y

    def forward(self, x):
        """
        Forward pass through memristor crossbar.

        Pipeline:
        Input → DAC → Crossbar MVM → Wire Loss → ADC → Output
                      (quantized + noisy weights)
        """
        # 1. DAC: Quantize input (digital → analog)
        x_dac = self.dac_quantize(x)

        # 2. Weight quantization (simulates memristor programming)
        w_quant = self.quantize_weights(self.weight)

        # 3. Device variation (manufacturing imperfections, fixed)
        w_varied = self.add_device_variation(w_quant)

        # 4. Read noise (cycle-to-cycle variation, stochastic)
        w_noisy = self.add_read_noise(w_varied)

        # 5. Crossbar matrix-vector multiplication (analog compute)
        #    This is where the magic happens: V = I * R → I = V / R = V * G
        #    Currents sum at each column → analog MAC operation!
        y = F.linear(x_dac, w_noisy, None)

        # 6. Wire resistance effect (IR drop)
        y = self.apply_wire_resistance(y)

        # 7. ADC: Quantize output (analog → digital)
        y_adc = self.adc_quantize(y)

        # 8. Bias (computed digitally, outside crossbar)
        if self.bias is not None:
            y_adc = y_adc + self.bias

        return y_adc

    def get_energy_breakdown(self, batch_size=1, input_sparsity=0.1):
        """
        Calculate energy consumption breakdown.

        Based on parameters from:
        - Cai+ "Fully memristive neural networks" Nature Electronics 2019
        - Yao+ "Fully hardware-implemented memristor CNNs" Nature 2020

        Args:
            batch_size: Number of samples
            input_sparsity: Fraction of non-zero inputs (for SNNs)

        Returns:
            Dictionary with energy breakdown in Joules
        """
        # Energy parameters (from literature)
        E_DAC_per_sample = 10e-12   # 10 pJ per conversion (8-bit)
        E_ADC_per_sample = 20e-12   # 20 pJ per conversion (8-bit)
        E_MAC_analog = 2e-15        # 2 fJ per analog MAC operation

        # Scale with bit-width (energy ≈ 2^n for n-bit conversion)
        E_DAC = E_DAC_per_sample * (2 ** (self.dac_bits - 8))
        E_ADC = E_ADC_per_sample * (2 ** (self.adc_bits - 8))

        n_in = self.weight.size(1)
        n_out = self.weight.size(0)

        # DAC energy: one conversion per input
        E_dac_total = n_in * E_DAC * batch_size

        # Analog MVM energy: only for active (non-zero) inputs
        # In SNNs, sparsity is typically 5-20%
        n_active_ops = n_in * n_out * input_sparsity
        E_mvm_total = n_active_ops * E_MAC_analog * batch_size

        # ADC energy: one conversion per output
        E_adc_total = n_out * E_ADC * batch_size

        E_total = E_dac_total + E_mvm_total + E_adc_total

        return {
            'E_dac_J': E_dac_total,
            'E_mvm_J': E_mvm_total,
            'E_adc_J': E_adc_total,
            'E_total_J': E_total,
            'breakdown_percent': {
                'DAC': E_dac_total / E_total * 100 if E_total > 0 else 0,
                'MVM': E_mvm_total / E_total * 100 if E_total > 0 else 0,
                'ADC': E_adc_total / E_total * 100 if E_total > 0 else 0,
            },
            'params': {
                'weight_bits': self.weight_bits,
                'adc_bits': self.adc_bits,
                'dac_bits': self.dac_bits,
                'n_synapses': n_in * n_out,
            }
        }


# === cell #10 ===
# =============================================================================
# CROSSBAR + NORSE LIF INTEGRATION
# =============================================================================

class CrossbarLIFLinear(NorseLIFBase):
    """
    Norse LIF neuron with memristor crossbar for synaptic weights.

    Combines:
    - QuantizedLinear: Crossbar simulation (analog MVM)
    - Norse LIF: Hardware-realistic neuron dynamics

    This represents the complete memristor + LIF neuron system.

    Args:
        in_features: Input dimension
        out_features: Output dimension (number of neurons)
        bias: Include bias (default: True)
        weight_bits: Weight quantization (default: 8)
        adc_bits: ADC resolution (default: 8)
        dac_bits: DAC resolution (default: 8)
        noise_std: Read noise (default: 0.01)
        wire_resistance: Wire resistance factor (default: 0.0)
        device_variation: Device variation (default: 0.02)
        **kwargs: Norse LIF parameters (T, tau_mem_inv, tau_syn_inv, v_th, etc.)
    """

    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, wire_resistance=0.0,
                 device_variation=0.02, **kwargs):
        super().__init__(output_features=out_features, **kwargs)

        # Replace standard Linear with QuantizedLinear (crossbar)
        self.fc = QuantizedLinear(
            in_features, out_features, bias=bias,
            weight_bits=weight_bits,
            adc_bits=adc_bits,
            dac_bits=dac_bits,
            noise_std=noise_std,
            wire_resistance=wire_resistance,
            device_variation=device_variation
        )

    def current(self, x):
        """
        Compute input current through crossbar.

        Flow: Input → Crossbar MVM → Current → LIF
        """
        return self.fc(x)

    def get_layer_energy(self, batch_size=1, input_sparsity=0.1):
        """
        Calculate total layer energy (crossbar + LIF).

        Energy components:
        1. Crossbar MVM (DAC + analog MAC + ADC)
        2. LIF spike generation
        3. LIF leakage

        Returns energy per timestep and total over T timesteps.
        """
        # Crossbar energy (per timestep)
        crossbar_energy = self.fc.get_energy_breakdown(batch_size, input_sparsity)

        # LIF energy parameters (from Intel Loihi)
        E_spike = 23.6e-12       # 23.6 pJ per spike
        E_neuron_leak = 0.5e-12  # 0.5 pJ per neuron per timestep

        n_neurons = self.output_features
        expected_spikes = n_neurons * input_sparsity * batch_size

        # LIF energy per timestep
        E_lif_spikes = expected_spikes * E_spike
        E_lif_leak = n_neurons * E_neuron_leak * batch_size
        E_lif_per_timestep = E_lif_spikes + E_lif_leak

        # Total over T timesteps
        E_crossbar_total = crossbar_energy['E_total_J'] * self.T
        E_lif_total = E_lif_per_timestep * self.T
        E_total = E_crossbar_total + E_lif_total

        return {
            'E_crossbar_J': E_crossbar_total,
            'E_lif_J': E_lif_total,
            'E_total_J': E_total,
            'breakdown_percent': {
                'DAC': crossbar_energy['breakdown_percent']['DAC'] * (E_crossbar_total / E_total),
                'MVM': crossbar_energy['breakdown_percent']['MVM'] * (E_crossbar_total / E_total),
                'ADC': crossbar_energy['breakdown_percent']['ADC'] * (E_crossbar_total / E_total),
                'LIF_spikes': (E_lif_spikes * self.T) / E_total * 100,
                'LIF_leak': (E_lif_leak * self.T) / E_total * 100,
            },
            'params': {
                **crossbar_energy['params'],
                'n_neurons': n_neurons,
                'n_timesteps': self.T,
                'sparsity': input_sparsity,
            }
        }


# === cell #11 ===
# =============================================================================
# CROSSBAR CONVOLUTIONAL LAYERS - FIXED
# =============================================================================

class CrossbarLIFConv2d(NorseLIFBase):
    """
    Norse LIF with memristor crossbar for convolutional layers.

    Conv2d weights are quantized to simulate crossbar implementation.
    In hardware, conv is implemented as im2col + crossbar MVM.

    Full hardware pipeline (consistent with QuantizedLinear):
    Input -> DAC -> Weight quant -> Device variation -> Read noise -> Conv -> Wire resistance -> ADC

    Args:
        c_in: Input channels
        c_out: Output channels
        k: Kernel size (default: 3)
        s: Stride (default: 1)
        p: Padding (default: 1)
        bias: Include bias (default: False)
        weight_bits: Weight quantization (default: 8)
        adc_bits: ADC resolution (default: 8)
        dac_bits: DAC resolution (default: 8)
        noise_std: Read noise (default: 0.01)
        wire_resistance: Wire resistance factor 0-1 (default: 0.0)
        device_variation: Manufacturing variation std (default: 0.02)
        **kwargs: Norse LIF parameters (T, tau_mem_inv, tau_syn_inv, v_th, etc.)
    """

    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, wire_resistance=0.0,
                 device_variation=0.02, **kwargs):

        # Remove crossbar params from kwargs if present
        kwargs_norse = {k: v for k, v in kwargs.items()
                       if k not in ['weight_bits', 'adc_bits', 'dac_bits',
                                   'noise_std', 'device_variation', 'wire_resistance']}

        # Initialize NorseLIFBase with only Norse parameters
        super().__init__(output_features=c_out, **kwargs_norse)

        # Create Conv2d layer
        self.cv = nn.Conv2d(c_in, c_out, kernel_size=k,
                           stride=s, padding=p, bias=bias)

        # Store crossbar parameters
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.wire_resistance = wire_resistance
        self.device_variation = device_variation

        # Device variation mask (fixed at init, represents manufacturing)
        self.register_buffer('device_variation_mask', None)

    def quantize_conv_weights(self, w):
        """Quantize conv weights with STE"""
        n_levels = 2 ** self.weight_bits

        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8

        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min

        return w + (w_quant - w).detach()

    def dac_quantize(self, x):
        """Simulate DAC quantization (digital -> analog conversion)."""
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
        """ADC quantization for output"""
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

    def add_device_variation(self, w):
        """Add device-to-device variation (manufacturing imperfections, fixed)."""
        if self.device_variation > 0:
            if self.device_variation_mask is None:
                self.device_variation_mask = torch.randn(
                    w.shape, device=w.device) * self.device_variation
            w = w * (1.0 + self.device_variation_mask)
        return w

    def apply_wire_resistance(self, y):
        """Simulate IR drop due to wire resistance in crossbar."""
        if self.wire_resistance > 0:
            n_channels = y.size(1)
            position_factor = torch.linspace(
                1.0, 1.0 - self.wire_resistance,
                n_channels, device=y.device
            ).view(1, -1, 1, 1)
            y = y * position_factor
        return y

    def current(self, x):
        # 1. DAC: Quantize input (digital -> analog)
        x_dac = self.dac_quantize(x)

        # 2. Weight quantization (simulates memristor programming)
        w_quant = self.quantize_conv_weights(self.cv.weight)

        # 3. Device variation (manufacturing imperfections, fixed)
        w_varied = self.add_device_variation(w_quant)

        # 4. Read noise (cycle-to-cycle variation, multiplicative on weights)
        if self.noise_std > 0 and self.training:
            w_noisy = w_varied * (1.0 + torch.randn_like(w_varied) * self.noise_std)
        else:
            w_noisy = w_varied

        # 5. Convolution (analog MVM via im2col + crossbar)
        y = F.conv2d(x_dac, w_noisy, self.cv.bias,
                    stride=self.cv.stride, padding=self.cv.padding)

        # 6. Wire resistance (IR drop)
        y = self.apply_wire_resistance(y)

        # 7. ADC: Quantize output (analog -> digital)
        y = self.adc_quantize(y)

        return y


class CrossbarLIFConvT2d(NorseLIFBase):
    """
    Norse LIF with memristor crossbar for transposed convolution.
    Similar to CrossbarLIFConv2d but for upsampling.

    Full hardware pipeline (consistent with QuantizedLinear):
    Input -> DAC -> Weight quant -> Device variation -> Read noise -> ConvT -> Wire resistance -> ADC

    Args:
        c_in: Input channels
        c_out: Output channels
        k: Kernel size (default: 4)
        s: Stride (default: 2)
        p: Padding (default: 1)
        bias: Include bias (default: False)
        weight_bits: Weight quantization (default: 8)
        adc_bits: ADC resolution (default: 8)
        dac_bits: DAC resolution (default: 8)
        noise_std: Read noise (default: 0.01)
        wire_resistance: Wire resistance factor 0-1 (default: 0.0)
        device_variation: Manufacturing variation std (default: 0.02)
        **kwargs: Norse LIF parameters (T, tau_mem_inv, tau_syn_inv, v_th, etc.)
    """

    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, wire_resistance=0.0,
                 device_variation=0.02, **kwargs):

        # Remove crossbar params from kwargs
        kwargs_norse = {k: v for k, v in kwargs.items()
                       if k not in ['weight_bits', 'adc_bits', 'dac_bits',
                                   'noise_std', 'device_variation', 'wire_resistance']}

        # Initialize NorseLIFBase
        super().__init__(output_features=c_out, **kwargs_norse)

        # Create ConvTranspose2d layer
        self.cvt = nn.ConvTranspose2d(c_in, c_out, kernel_size=k,
                                     stride=s, padding=p, bias=bias)

        # Store crossbar parameters
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.wire_resistance = wire_resistance
        self.device_variation = device_variation

        # Device variation mask (fixed at init, represents manufacturing)
        self.register_buffer('device_variation_mask', None)

    def quantize_conv_weights(self, w):
        """Quantize transposed conv weights with STE"""
        n_levels = 2 ** self.weight_bits

        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8

        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min

        return w + (w_quant - w).detach()

    def dac_quantize(self, x):
        """Simulate DAC quantization (digital -> analog conversion)."""
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
        """ADC quantization"""
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

    def add_device_variation(self, w):
        """Add device-to-device variation (manufacturing imperfections, fixed)."""
        if self.device_variation > 0:
            if self.device_variation_mask is None:
                self.device_variation_mask = torch.randn(
                    w.shape, device=w.device) * self.device_variation
            w = w * (1.0 + self.device_variation_mask)
        return w

    def apply_wire_resistance(self, y):
        """Simulate IR drop due to wire resistance in crossbar."""
        if self.wire_resistance > 0:
            n_channels = y.size(1)
            position_factor = torch.linspace(
                1.0, 1.0 - self.wire_resistance,
                n_channels, device=y.device
            ).view(1, -1, 1, 1)
            y = y * position_factor
        return y

    def current(self, x):
        # 1. DAC: Quantize input (digital -> analog)
        x_dac = self.dac_quantize(x)

        # 2. Weight quantization (simulates memristor programming)
        w_quant = self.quantize_conv_weights(self.cvt.weight)

        # 3. Device variation (manufacturing imperfections, fixed)
        w_varied = self.add_device_variation(w_quant)

        # 4. Read noise (cycle-to-cycle variation, multiplicative on weights)
        if self.noise_std > 0 and self.training:
            w_noisy = w_varied * (1.0 + torch.randn_like(w_varied) * self.noise_std)
        else:
            w_noisy = w_varied

        # 5. Transposed convolution (analog MVM)
        y = F.conv_transpose2d(x_dac, w_noisy, self.cvt.bias,
                              stride=self.cvt.stride,
                              padding=self.cvt.padding)

        # 6. Wire resistance (IR drop)
        y = self.apply_wire_resistance(y)

        # 7. ADC: Quantize output (analog -> digital)
        y = self.adc_quantize(y)

        return y


# === cell #12 ===
# =============================================================================
# SCNN MODELS WITH MEMRISTOR CROSSBAR
# =============================================================================

class SCNNGen_Crossbar_Simple2D(nn.Module):
    """
    SCNN with memristor crossbar simulation for all synaptic layers.

    This model simulates the complete hardware stack:
    - Quantized weights (8-bit default)
    - ADC/DAC quantization
    - Device variation and read noise
    - Norse LIF neurons with hardware-realistic dynamics

    Args:
        cond_dim: Conditioning dimension (measurement data)
        d: Hilbert space dimension (2^N for N qubits)
        proj_hw: Projection spatial dimensions (H, W)
        ch: Channel sizes (C0, C1)
        T: Number of timesteps
        tau_mem_inv: Membrane decay rate (Norse)
        tau_syn_inv: Synaptic decay rate (Norse)
        v_th: Firing threshold (Norse)
        weight_bits: Weight quantization bits (default: 8)
        adc_bits: ADC resolution (default: 8)
        dac_bits: DAC resolution (default: 8)
        noise_std: Read noise std (default: 0.01)
        device_variation: Manufacturing variation (default: 0.02)
        enc_mode: Spike encoding ('poisson')
        enc_gamma: Poisson encoding sharpness
    """

    def __init__(self, cond_dim, d, proj_hw=(12,8), ch=(32,64),
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        H0,W0, C0,C1 = *proj_hw, *ch
        self.d, self.C0, self.HW = d, C0, (H0,W0)

        # Crossbar + Norse LIF parameters
        kw_spk = dict(
            T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
            v_th=v_th, return_rate=True, enc_mode=enc_mode, enc_gamma=enc_gamma,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )
        kw_ro = dict(
            T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
            v_th=v_th, return_rate=False, enc_mode=enc_mode, enc_gamma=enc_gamma,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )

        # Network layers with crossbar
        self.proj = CrossbarLIFLinear(cond_dim, C0*H0*W0, **kw_spk)
        self.c1 = CrossbarLIFConv2d(C0, C1, k=3, p=1, **kw_spk)
        self.c2 = CrossbarLIFConv2d(C1, C1, k=3, p=1, **kw_spk)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ro = CrossbarLIFLinear(C1, 2*d*d, **kw_ro)
        self.dm = DensityMap()

        self.apply(init_normal_002)
        self.kickstart = True

    def forward(self, cond):
        B, H0, W0 = cond.size(0), *self.HW
        x = self.proj(cond).view(B, self.C0, H0, W0)
        x = self.c1(x)
        x = self.c2(x)
        x = self.gap(x).flatten(1)
        x = self.ro(x).view(B, 2, self.d, self.d)
        if self.training and getattr(self, 'kickstart', True):
            x = x + 1e-3 * torch.randn_like(x)
        return self.dm(x)

    def get_model_energy(self, batch_size=1, input_sparsity=0.1):
        """
        Calculate total model energy consumption.

        Returns energy breakdown for all layers.
        """
        layers = [
            ('proj', self.proj),
            ('c1', self.c1),
            ('c2', self.c2),
            ('ro', self.ro)
        ]

        total_energy = 0
        layer_breakdown = {}

        for name, layer in layers:
            if hasattr(layer, 'get_layer_energy'):
                energy = layer.get_layer_energy(batch_size, input_sparsity)
                layer_breakdown[name] = energy
                total_energy += energy['E_total_J']

        return {
            'E_total_J': total_energy,
            'E_total_mJ': total_energy * 1e3,
            'layer_breakdown': layer_breakdown
        }


class SCNNGen_Crossbar_Up2D_Paper(nn.Module):
    """
    SCNN with upsampling architecture and memristor crossbar.
    Similar to Simple2D but with transposed convolutions for upsampling.
    """

    def __init__(self, cond_dim, d, T=8,
                 tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        self.d, h, w = d, d//2, d//2

        # Crossbar + Norse LIF parameters
        kw_spk = dict(
            T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
            v_th=v_th, return_rate=True, enc_mode=enc_mode, enc_gamma=enc_gamma,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )
        kw_ro = dict(
            T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
            v_th=v_th, return_rate=False, enc_mode=enc_mode, enc_gamma=enc_gamma,
            weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
            noise_std=noise_std, device_variation=device_variation
        )

        # Network with crossbar layers
        self.proj = CrossbarLIFLinear(cond_dim, 2*h*w, **kw_spk)
        self.de1 = CrossbarLIFConvT2d(2, 64, k=4, s=2, p=1, **kw_spk)
        self.de2 = CrossbarLIFConvT2d(64, 64, k=3, s=1, p=1, **kw_spk)
        self.de3 = CrossbarLIFConvT2d(64, 32, k=3, s=1, p=1, **kw_spk)
        self.de4 = CrossbarLIFConvT2d(32, 2, k=3, s=1, p=1, **kw_ro)
        self.dm = DensityMap()

        self.apply(init_normal_002)
        self.kickstart = True

    def forward(self, cond):
        B, h, w = cond.size(0), self.d//2, self.d//2
        x = self.proj(cond).view(B,2,h,w)
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = self.de4(x)
        if self.training and getattr(self, 'kickstart', True):
            x = x + 1e-3 * torch.randn_like(x)
        return self.dm(x)

    def get_model_energy(self, batch_size=1, input_sparsity=0.1):
        """Calculate total model energy"""
        layers = [
            ('proj', self.proj),
            ('de1', self.de1),
            ('de2', self.de2),
            ('de3', self.de3),
            ('de4', self.de4)
        ]

        total_energy = 0
        layer_breakdown = {}

        for name, layer in layers:
            if hasattr(layer, 'get_layer_energy'):
                energy = layer.get_layer_energy(batch_size, input_sparsity)
                layer_breakdown[name] = energy
                total_energy += energy['E_total_J']

        return {
            'E_total_J': total_energy,
            'E_total_mJ': total_energy * 1e3,
            'layer_breakdown': layer_breakdown
        }


# === cell #13 ===
# =============================================================================
# 11-12. CNN & SCNN MODELS
# =============================================================================

class CNNGen_Simple2D(nn.Module):
    def __init__(self, cond_dim, d, proj_hw=(12,8), ch=(32,64), dropout=0.0):
        super().__init__()
        H0,W0, C0,C1 = *proj_hw, *ch
        self.d, self.C0, self.HW = d, C0, (H0,W0)
        self.proj = nn.Linear(cond_dim, C0*H0*W0, bias=False)
        self.conv1 = nn.Conv2d(C0,C1,3,padding=1,bias=False)
        self.in1 = nn.InstanceNorm2d(C1, affine=True)
        self.conv2 = nn.Conv2d(C1,C1,3,padding=1,bias=False)
        self.in2 = nn.InstanceNorm2d(C1, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout>0 else nn.Identity()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(C1, 2*d*d, bias=False)
        self.apply(init_normal_002)
        self.dm = DensityMap()

    def forward(self, cond):
        B, H0, W0 = cond.size(0), *self.HW
        x = self.proj(cond).view(B,self.C0,H0,W0)
        x = self.act(self.in1(self.conv1(x)))
        x = self.act(self.in2(self.conv2(x)))
        x = self.drop(x)
        x = self.gap(x).flatten(1)
        x = self.head(x).view(B,2,self.d,self.d)
        return self.dm(x)

class CNNGen_Up2D_Paper(nn.Module):
    def __init__(self, cond_dim, d):
        super().__init__()
        self.d = d
        h = w = d//2
        self.proj = nn.Linear(cond_dim, 2*h*w, bias=False)
        self.de1 = nn.ConvTranspose2d(2, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(64, affine=True)
        self.de2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.de3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.de4 = nn.ConvTranspose2d(32, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.apply(init_normal_002)
        self.dm = DensityMap()

    def forward(self, cond):
        B, h, w = cond.size(0), self.d//2, self.d//2
        x = self.proj(cond).view(B,2,h,w)
        x = self.act(self.in1(self.de1(x)))
        x = self.act(self.in2(self.de2(x)))
        x = self.de3(x)
        x = self.de4(x)
        return self.dm(x)

class SCNNGen_LIF_Simple2D(nn.Module):
    def __init__(self, cond_dim, d, proj_hw=(12,8), ch=(32,64),
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        H0,W0, C0,C1 = *proj_hw, *ch
        self.d, self.C0, self.HW = d, C0, (H0,W0)

        # Norse-style parameters (direct decay rates, not milliseconds!)
        kw_spk = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=True, enc_mode=enc_mode,
                      enc_gamma=enc_gamma)
        kw_ro  = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=False, enc_mode=enc_mode,
                      enc_gamma=enc_gamma)

        self.proj = NorseLIFLinear(cond_dim, C0*H0*W0, **kw_spk)
        self.c1 = NorseLIFConv2d(C0, C1, k=3, p=1, **kw_spk)
        self.c2 = NorseLIFConv2d(C1, C1, k=3, p=1, **kw_spk)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ro = NorseLIFLinear(C1, 2*d*d, **kw_ro)
        self.dm = DensityMap()
        self.apply(init_normal_002)
        self.kickstart = True

    def forward(self, cond):
        B, H0, W0 = cond.size(0), *self.HW
        x = self.proj(cond).view(B, self.C0, H0, W0)
        x = self.c1(x)
        x = self.c2(x)
        x = self.gap(x).flatten(1)
        x = self.ro(x).view(B, 2, self.d, self.d)
        if self.training and getattr(self, 'kickstart', True):
            x = x + 1e-3 * torch.randn_like(x)
        return self.dm(x)


class SCNNGen_LIF_Up2D_Paper(nn.Module):
    def __init__(self, cond_dim, d, T=8, tau_mem_inv=100.0, tau_syn_inv=200.0,
                 v_th=1.0, enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        self.d, h, w = d, d//2, d//2

        # Norse-style parameters (direct decay rates, not milliseconds!)
        kw_spk = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=True, enc_mode=enc_mode,
                      enc_gamma=enc_gamma)
        kw_ro  = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=False, enc_mode=enc_mode,
                      enc_gamma=enc_gamma)

        self.proj = NorseLIFLinear(cond_dim, 2*h*w, **kw_spk)
        self.de1 = NorseLIFConvT2d(2, 64, k=4, s=2, p=1, **kw_spk)
        self.de2 = NorseLIFConvT2d(64, 64, k=3, s=1, p=1, **kw_spk)
        self.de3 = NorseLIFConvT2d(64, 32, k=3, s=1, p=1, **kw_spk)
        self.de4 = NorseLIFConvT2d(32, 2, k=3, s=1, p=1, **kw_ro)
        self.dm = DensityMap()
        self.apply(init_normal_002)
        self.kickstart = True

    def forward(self, cond):
        B, h, w = cond.size(0), self.d//2, self.d//2
        x = self.proj(cond).view(B,2,h,w)
        x = self.de1(x)
        x = self.de2(x)
        x = self.de3(x)
        x = self.de4(x)
        if self.training and getattr(self, 'kickstart', True):
            x = x + 1e-3 * torch.randn_like(x)
        return self.dm(x)


# === cell #14 ===
# =============================================================================
# TRAINING LOOP CON CROSSBAR - Energy-Aware
# =============================================================================

def optimize_single_state_crossbar(model_class, N=3, method='M1', M=128, steps=500, lr=1e-3,
                                   noise_std=0.0, normalize_cond=True, warm_spiking=False,
                                   weight_bits=8, adc_bits=8, dac_bits=8,
                                   device_variation=0.02, read_noise=0.01,
                                   use_amp=True, log_energy=True):
    """
    Training con crossbar memristor simulation (mixed GHZ target).

    NOTA: model_class deve essere un callable che accetta (cond_dim, d) come argomenti.
          I parametri crossbar (weight_bits, etc) devono essere gia inclusi nel lambda
          oppure passati qui per essere usati nel print.
    """
    d = 2**N
    rho_t = mixed_ghz_state(N, p=0.5)
    rho_t_ri = to_ri(rho_t).unsqueeze(0).to(DEVICE)

    # Setup input
    if method == 'M1':
        ops = select_ops_nonzero_M1(rho_t, M=M, N=N)
        M = len(ops)
        y = compute_expectations(rho_t, ops)
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
        x_in = y_t
        if normalize_cond:
            mu, sd = x_in.mean(dim=1, keepdim=True), x_in.std(dim=1, keepdim=True).clamp_min(1e-8)
            x_in = (x_in - mu) / sd
        cond_vec = x_in
        ops_c = np.stack(ops, axis=0)
        ops_ri = np.stack([ops_c.real, ops_c.imag], axis=1)
        ops_ri_t = torch.from_numpy(ops_ri).to(DEVICE)
        exp = ExpectationLayer().to(DEVICE)
        exp.set_ops(ops_ri_t)
        target = y_t
    elif method == 'M2':
        bases = select_bases_nonzero_M2(rho_t_ri, N=N, M=M, seed=0)
        target = probs_from_bases_torch(rho_t_ri, bases)
        cond_vec = target.clone().detach()
    else:
        raise ValueError("method must be 'M1' or 'M2'")

    cond_dim = cond_vec.shape[-1]

    # Create crossbar model - model_class gia contiene i parametri crossbar
    G = model_class(cond_dim, d).to(DEVICE)

    if warm_spiking:
        warm_init_spiking(G, w_scale=3.0, bias=0.1)
        if hasattr(G, 'kickstart'):
            G.kickstart = True

    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda t: 1.0 / (1.0 + 0.96 * t / max(steps, 1))
    )

    # Mixed precision setup
    use_amp_actual = use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp_actual)

    f_hist, l_hist, e_hist = [], [], []

    print(f"    Training with {weight_bits}-bit weights, {adc_bits}-bit ADC/DAC")
    print(f"    Device variation: {device_variation*100:.1f}%, Read noise: {read_noise*100:.1f}%")
    print(f"    Target: Mixed GHZ state (Werner p=0.5)")

    t0 = time.time()

    for it in range(1, steps+1):
        G.train()

        # Forward pass with autocast
        with autocast(enabled=use_amp_actual):
            rho_hat = G(cond_vec)

            if method == 'M1':
                y_hat = exp(rho_hat, None)
            else:
                y_hat = probs_from_bases_torch(rho_hat, bases)

            if noise_std > 0:
                y_hat = y_hat + noise_std * torch.randn_like(y_hat)

            loss = F.mse_loss(y_hat, target)

        # Backward pass
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it == 20 and hasattr(G, 'kickstart'):
            G.kickstart = False

        # Evaluation
        with torch.no_grad():
            F_vec, F_mean = fidelity_batch(rho_hat, rho_t_ri)

        f_hist.append(F_mean)
        l_hist.append(float(loss.item()))

        # Energy logging (ogni 50 iter per non rallentare)
        if log_energy and it % 50 == 0:
            with torch.no_grad():
                energy = G.get_model_energy(batch_size=1, input_sparsity=0.1)
                e_hist.append(energy['E_total_mJ'])

        # Progress
        if it % 100 == 0 or it == steps:
            energy_str = f", E={e_hist[-1]:.4f}mJ" if e_hist else ""
            print(f"      Iter {it:3d}: Loss={loss.item():.6f}, F={F_mean:.4f}{energy_str}")

    elapsed = time.time() - t0

    # Final evaluation
    G.eval()
    with torch.no_grad():
        rho_final = G(cond_vec)
        F_vec, F_final = fidelity_batch(rho_final, rho_t_ri)

        # Final energy calculation
        energy_final = G.get_model_energy(batch_size=1, input_sparsity=0.1)

    print(f"    Final: F={F_final:.4f}, E={energy_final['E_total_mJ']:.4f}mJ, Time={elapsed:.1f}s")

    # Free optimizer states and intermediate GPU tensors before returning
    del opt, scaler, scheduler, cond_vec, target, rho_t_ri
    if method == 'M1':
        del exp, ops_ri_t
    else:
        del bases
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Return results
    return {
        'model': G,
        'F_best': F_final,
        'F_hist': f_hist,
        'loss_hist': l_hist,
        'energy_hist': e_hist,
        'energy_final': energy_final,
        'time_sec': elapsed,
        'config': {
            'N': N,
            'method': method,
            'weight_bits': weight_bits,
            'adc_bits': adc_bits,
            'dac_bits': dac_bits,
            'device_variation': device_variation,
            'read_noise': read_noise,
        }
    }


# =============================================================================
# BENCHMARK FUNCTION - Crossbar Models
# =============================================================================

def run_crossbar_benchmark(N_list=[3,4,5], methods=['M1','M2'],
                          M_M1=256, M_M2=4, steps=500, lr=1e-3,
                          weight_bits=8, adc_bits=8, dac_bits=8,
                          device_variation=0.02, read_noise=0.01,
                          use_amp=True):
    """
    Benchmark completo con crossbar models (mixed GHZ target).

    Args:
        N_list: Lista di N qubit
        methods: Lista metodi ('M1', 'M2')
        weight_bits: Quantization bits (8 consigliato)
        adc_bits: ADC resolution (8 consigliato)
        dac_bits: DAC resolution (8 consigliato)
        device_variation: Manufacturing variation (0.02 = 2%)
        read_noise: Read noise std (0.01 = 1%)

    Returns:
        Dictionary con risultati per ogni configurazione
    """
    results = defaultdict(list)

    for method in methods:
        print(f"\n{'='*70}")
        print(f"CROSSBAR BENCHMARK (Mixed GHZ p=0.5): {method}")
        print(f"   Crossbar config: {weight_bits}-bit weights, {adc_bits}-bit ADC/DAC")
        print(f"={'='*70}")

        for N in N_list:
            # Fixed M from sweep studies
            if method == 'M1':
                M = min(4**N - 1, M_M1)  # Cap at available Paulis
            else:
                M = M_M2
            d = 2**N
            print(f"\n--- N={N} qubit (d={d}) ---")

            # Adapt parameters for larger N
            current_steps = steps  # Fair benchmark: same steps for all N

            # Model configurations (crossbar versions)
            model_configs = [
                ('SCNN-Crossbar-Simple2D',
                 lambda cond_dim, d, N=N: SCNNGen_Crossbar_Simple2D(
                     cond_dim=cond_dim, d=d, proj_hw=(N*2,N*2), ch=(32,64),
                     T=(32 if N>=7 else 16 if N>=6 else 8),
                     v_th=(0.8 if N>=7 else 0.9 if N>=6 else 1.0),
                     enc_gamma=(2.5 if N>=7 else 2.0 if N>=6 else 1.5),
                     weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                     device_variation=device_variation, noise_std=read_noise
                 ),
                 True),
                ('SCNN-Crossbar-Up2D-Paper',
                 lambda cond_dim, d, N=N: SCNNGen_Crossbar_Up2D_Paper(
                     cond_dim=cond_dim, d=d,
                     T=(32 if N>=7 else 16 if N>=6 else 8),
                     v_th=(0.8 if N>=7 else 0.9 if N>=6 else 1.0),
                     enc_gamma=(2.5 if N>=7 else 2.0 if N>=6 else 1.5),
                     weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                     device_variation=device_variation, noise_std=read_noise
                 ),
                 True),
            ]

            for model_name, model_fn, is_spiking in model_configs:
                print(f"\n  {model_name}")

                t0 = time.time()

                res = optimize_single_state_crossbar(
                    model_class=model_fn,
                    N=N, method=method, M=M,
                    steps=current_steps, lr=lr,
                    warm_spiking=is_spiking,
                    weight_bits=weight_bits,
                    adc_bits=adc_bits,
                    dac_bits=dac_bits,
                    device_variation=device_variation,
                    read_noise=read_noise,
                    use_amp=use_amp,
                    log_energy=True
                )

                elapsed = time.time() - t0

                # Store results
                results[method].append({
                    'Model': model_name,
                    'N': N,
                    'F_best': res['F_best'],
                    'Energy_Total_J': res['energy_final']['E_total_J'],
                    'Energy_Total_mJ': res['energy_final']['E_total_mJ'],
                    'Energy_Details': res['energy_final'],
                    'Time_sec': elapsed,
                    'f_hist': res['F_hist'],
                    'Config': res['config'],
                    'HW_Type': 'Memristor Crossbar'
                })

                print(f"    Time: {elapsed:.1f}s")

    return results


# =============================================================================
# COMPARISON: Crossbar vs Standard (per vedere impatto quantization)
# =============================================================================

def compare_crossbar_vs_standard(N=3, method='M1', M=128, steps=500,
                                weight_bits_list=[32, 8, 4, 2]):
    """
    Confronta diverse quantization levels per vedere impatto su accuracy.

    Args:
        weight_bits_list: Lista di bit-widths da testare (32 = FP32 baseline)

    Returns:
        DataFrame con risultati comparativi
    """
    print(f"\n{'='*70}")
    print(f"QUANTIZATION IMPACT ANALYSIS (Mixed GHZ p=0.5): N={N}, {method}")
    print(f"{'='*70}\n")

    results = []

    for bits in weight_bits_list:
        print(f"Testing {bits}-bit weights...")

        if bits == 32:
            # Baseline: standard model (no quantization)
            res = optimize_single_state(
                model_class=lambda cond_dim, d: SCNNGen_LIF_Simple2D(cond_dim=cond_dim, d=d),
                N=N, method=method, M=M, steps=steps, warm_spiking=True
            )
            hw_type = 'Standard (FP32)'
            energy = None  # No crossbar energy
        else:
            # Crossbar model with quantization
            res = optimize_single_state_crossbar(
                model_class=lambda cond_dim, d: SCNNGen_Crossbar_Simple2D(
                    cond_dim=cond_dim, d=d,
                    weight_bits=bits, adc_bits=bits, dac_bits=bits
                ),
                N=N, method=method, M=M, steps=steps,
                weight_bits=bits, adc_bits=bits, dac_bits=bits,
                warm_spiking=True, log_energy=False
            )
            hw_type = f'Crossbar ({bits}-bit)'
            energy = res['energy_final']['E_total_mJ']

        results.append({
            'seed': SEED,
            'Bits': bits if bits < 32 else 'FP32',
            'Fidelity': res['F_best'] if bits == 32 else res['F_best'],
            'Energy_mJ': energy,
            'HW_Type': hw_type
        })

        print(f"  {results[-1]['Bits']:5s}: F={results[-1]['Fidelity']:.4f}")
        if energy:
            print(f"           E={energy:.4f}mJ\n")

    df = pd.DataFrame(results)

    # Calculate accuracy drop
    baseline_F = df[df['Bits'] == 'FP32']['Fidelity'].values[0]
    df['Accuracy_Drop_%'] = (baseline_F - df['Fidelity']) / baseline_F * 100

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(df.to_string(index=False))

    return df


# === cell #15 ===
# =============================================================================
# 13. TRAINING FUNCTION - GPU OPTIMIZED (Mixed GHZ Target)
# =============================================================================

def to_ri(x_np):
    return torch.from_numpy(np.stack([x_np.real, x_np.imag], axis=0).astype(np.float32))

def optimize_single_state(model_class, N=3, method='M1', M=128, steps=500, lr=1e-3,
                          noise_std=0.0, normalize_cond=True, warm_spiking=False,
                          use_amp=True, use_compile=False):
    """
    Ottimizzazione con Mixed Precision Training e opzionalmente torch.compile.
    Target: Mixed GHZ state (Werner state, p=0.5 pure-state weight).

    Args:
        use_amp: Usa Automatic Mixed Precision (default: True)
        use_compile: Usa torch.compile() per ottimizzazione (default: False)
    """
    d = 2**N
    rho_t = mixed_ghz_state(N, p=0.5)
    rho_t_ri = to_ri(rho_t).unsqueeze(0).to(DEVICE)

    if method == 'M1':
        ops = select_ops_nonzero_M1(rho_t, M=M, N=N)
        M = len(ops)
        y = compute_expectations(rho_t, ops)
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(DEVICE)
        x_in = y_t
        if normalize_cond:
            mu, sd = x_in.mean(dim=1, keepdim=True), x_in.std(dim=1, keepdim=True).clamp_min(1e-8)
            x_in = (x_in - mu) / sd
        cond_vec = x_in
        ops_c = np.stack(ops, axis=0)
        ops_ri = np.stack([ops_c.real, ops_c.imag], axis=1)
        ops_ri_t = torch.from_numpy(ops_ri).to(DEVICE)
        exp = ExpectationLayer().to(DEVICE)
        exp.set_ops(ops_ri_t)
        target = y_t
    elif method == 'M2':
        bases = select_bases_nonzero_M2(rho_t_ri, N=N, M=M, seed=0)
        target = probs_from_bases_torch(rho_t_ri, bases)
        cond_vec = target.clone().detach()
    else:
        raise ValueError("method must be 'M1' or 'M2'")

    cond_dim = cond_vec.shape[-1]
    G = model_class(cond_dim, d).to(DEVICE)  # Argomenti posizionali

    if warm_spiking and 'SCNN' in G.__class__.__name__:
        warm_init_spiking(G, w_scale=3.0, bias=0.1)
        if hasattr(G, 'kickstart'): G.kickstart = True

    # Opzionale: compila il modello per velocita (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            G = torch.compile(G, mode="reduce-overhead")
            print("      Model compiled")
        except Exception as e:
            print(f"      Compile failed: {e}")

    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
    # InverseTimeDecay: lr(t) = lr_0 / (1 + decay_rate * t / decay_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda t: 1.0 / (1.0 + 0.96 * t / max(steps, 1))
    )

    # Setup Mixed Precision
    use_amp = use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    f_hist, l_hist = [], []

    for it in range(1, steps+1):
        G.train()

        # Forward pass con autocast
        with autocast(enabled=use_amp):
            rho_hat = G(cond_vec)

            if method == 'M1':
                y_hat = exp(rho_hat, None)
            else:
                y_hat = probs_from_bases_torch(rho_hat, bases)

            if noise_std > 0:
                y_hat = y_hat + noise_std * torch.randn_like(y_hat)

            loss = F.mse_loss(y_hat, target)

        # Backward pass con gradient scaling
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it == 20 and hasattr(G, 'kickstart'): G.kickstart = False

        with torch.no_grad():
            F_vec, F_mean = fidelity_batch(rho_hat, rho_t_ri)

        f_hist.append(F_mean)
        l_hist.append(float(loss.item()))

    with torch.no_grad():
        rho_final = G(cond_vec)

    # Free optimizer states and intermediate GPU tensors before returning
    del opt, scaler, scheduler, cond_vec, target
    if method == 'M1':
        del exp, ops_ri_t
    else:
        del bases
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'model': G, 'f_hist': np.array(f_hist), 'l_hist': np.array(l_hist),
        'rho_t': rho_t_ri, 'rho_final': rho_final, 'M': M, 'N': N, 'method': method
    }


# === cell #16 ===
# =============================================================================
# COMPLETE BENCHMARK: CNN vs SCNN vs Crossbar - FIXED
# =============================================================================

# =============================================================================
# UNIFIED WRAPPER - FIXED con chiavi corrette
# =============================================================================

def optimize_with_unified_output(model_class, N, method, M, steps, lr,
                                 warm_spiking=False, use_amp=True,
                                 is_crossbar=False, **crossbar_kwargs):
    """
    Wrapper che uniforma output di optimize_single_state e optimize_single_state_crossbar.

    optimize_single_state ritorna:
        {'model': G, 'f_hist': array, 'l_hist': array, 'rho_t': ..., 'rho_final': ..., 'M': ..., 'N': ..., 'method': ...}

    optimize_single_state_crossbar ritorna:
        {'model': G, 'F_best': float, 'F_hist': list, 'loss_hist': list, 'energy_final': dict, 'config': dict}

    Questo wrapper uniforma a:
        {'model': model, 'F_best': float, 'F_hist': list, 'loss_hist': list, 'time_sec': float, ...}
    """
    import time

    t0 = time.time()

    if is_crossbar:
        # Use crossbar optimizer
        res = optimize_single_state_crossbar(
            model_class=model_class,
            N=N, method=method, M=M,
            steps=steps, lr=lr,
            warm_spiking=warm_spiking,
            use_amp=use_amp,
            log_energy=False,
            **crossbar_kwargs
        )
        elapsed = time.time() - t0

        return {
            'model': res['model'],
            'F_best': res['F_best'],
            'F_hist': res['F_hist'],
            'loss_hist': res.get('loss_hist', []),
            'time_sec': elapsed,
            'energy_final': res['energy_final']
        }
    else:
        # Use standard optimizer
        res = optimize_single_state(
            model_class=model_class,
            N=N, method=method, M=M,
            steps=steps, lr=lr,
            warm_spiking=warm_spiking,
            use_amp=use_amp
        )
        elapsed = time.time() - t0

        # Extract from optimize_single_state format
        # Keys: 'model', 'f_hist', 'l_hist', 'rho_t', 'rho_final', 'M', 'N', 'method'
        f_hist = res['f_hist']

        # F_best is the last (best) fidelity
        if isinstance(f_hist, np.ndarray):
            F_best = float(f_hist[-1]) if len(f_hist) > 0 else 0.0
            F_hist = f_hist.tolist()
        elif isinstance(f_hist, list):
            F_best = float(f_hist[-1]) if len(f_hist) > 0 else 0.0
            F_hist = f_hist
        else:
            F_best = float(f_hist)
            F_hist = [F_best]

        l_hist = res.get('l_hist', [])
        if isinstance(l_hist, np.ndarray):
            l_hist = l_hist.tolist()

        return {
            'model': res['model'],
            'F_best': F_best,
            'F_hist': F_hist,
            'loss_hist': l_hist,
            'time_sec': elapsed
        }

# =============================================================================
# CELLA FIXATA: run_benchmark_with_energy_breakdown
# =============================================================================
# SOSTITUISCI la cella originale con questa versione che include:
# - FIX 1: Parametri specifici per SCNN-Norse-Up2D (v_th più basso)
# - FIX 2: Parametri adattivi per SCNN-Crossbar-8b (meno noise per N problematici)
# =============================================================================

def gpu_cleanup():
    """Aggressive GPU memory cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_benchmark_with_energy_breakdown(N_list=[3,4,5], methods=['M1'],
                                        M_M1=256, M_M2=4, steps=500, lr=1e-3,
                                        crossbar_bits=[8, 4], T=8, sparsity=0.1,
                                        use_amp=True):
    """
    Benchmark con breakdown energetico completo.

    Hyperparameters validated by sweep studies:
    - T=8 (T-sweep), M1=256 Paulis / M2=4 bases (M-sweep)
    - Norse: v_th=0.9, gamma=1.5 | Xbar-8b: v_th=0.5, gamma=2.25
    - Xbar-4b: v_th=0.3, gamma=3.0 | ch=(32,64), depth=2

    Calcola separatamente:
    - E_inference: energia per singola inference sul target hardware
    - E_training: energia per training (sempre su GPU)
    - E_deployment: energia per weight programming (solo crossbar)
    """
    all_results = {
        'CNN': defaultdict(list),
        'SCNN-Norse': defaultdict(list),
        'SCNN-Crossbar': defaultdict(lambda: defaultdict(list))
    }

    print("\n" + "="*80)
    print("BENCHMARK CON BREAKDOWN ENERGETICO (Training vs Inference)")
    print("   Sweep-validated params: T=8, M1=256, M2=4")
    print("="*80)
    print(f"\nConfigurazione:")
    print(f"  • N qubits: {N_list}")
    print(f"  • Methods: {methods}")
    print(f"  • M1={M_M1} Paulis, M2={M_M2} bases (from M-sweep)")
    print(f"  • Training steps: {steps}")
    print(f"  • SNN timesteps: T={T} (from T-sweep), sparsity={sparsity*100:.0f}%")
    print(f"  • Crossbar bits: {crossbar_bits}")

    print(f"\n📊 METRICHE CALCOLATE:")
    print(f"  • E_inference: energia per SINGOLA inference su target HW")
    print(f"  • E_training: energia per training completo (su GPU per tutti)")
    print(f"  • E_programming: energia per weight programming (solo crossbar)")
    print(f"  • Speedup: calcolato su E_inference (confronto fair)")

    for method in methods:
        print(f"\n{'='*80}")
        print(f"📊 METHOD: {method}")
        print(f"{'='*80}")

        for N in N_list:
            d = 2**N

            # Fixed M from sweep studies (M-sweep validated at N=8)
            if method == 'M1':
                M = min(4**N - 1, M_M1)  # Cap at available Paulis for small N
            else:
                M = M_M2  # 4 bases sufficient for all N

            print(f"\n{'─'*80}")
            print(f"🔬 N={N} qubits (d={d}, M={M})")
            print(f"{'─'*80}")

            # Fixed hyperparameters from sweep studies
            current_steps = steps
            current_T = T       # T=8 optimal (T-sweep: beyond T=16 hurts fidelity)
            # Phase 2 multi-seed sweep: at N=8 V_th=0.3 (M1) / 0.5 (M2) lifts F by 1.3-3.8%
            current_vth = (0.3 if N == 8 and method == 'M1' else
                           0.5 if N == 8 and method == 'M2' else
                           0.9)
            current_gamma = 1.5 # Norse Simple2D (T-sweep)

            # Norse Up2D: fixed params (lower v_th for ConvTranspose2d architecture)
            up2d_T = current_T       # T=8 from T-sweep
            up2d_vth = 0.4           # Lower threshold for Up2D (ConvTranspose needs this)
            up2d_gamma = 2.5         # Stronger encoding for Up2D
            up2d_tau_mem = 100.0     # Same as Simple2D
            up2d_tau_syn = 200.0     # Same as Simple2D

            # Crossbar noise parameters (from M-sweep / T-sweep)
            cb8_variation = 0.02
            cb8_noise = 0.01
            cb8_steps = current_steps

            # ================================================================
            # 1. CNN BASELINE
            # ================================================================
            print(f"\n1️⃣  CNN (GPU)")

            # CNN-Simple2D
            print(f"\n  🔧 CNN-Simple2D")
            model_fn_cnn1 = lambda cd, d_: CNNGen_Simple2D(
                cond_dim=cd, d=d_, proj_hw=(N*2, N*2), ch=(32,64))

            res = optimize_with_unified_output(
                model_class=model_fn_cnn1, N=N, method=method, M=M,
                steps=current_steps, lr=lr, warm_spiking=False,
                use_amp=use_amp, is_crossbar=False
            )
            model = res['model']
            E_inf = estimate_gpu_inference_energy(model)
            E_train = estimate_gpu_training_energy(model, current_steps, measured_time_s=res['time_sec'])

            all_results['CNN'][method].append({
                'Model': 'CNN-Simple2D', 'N': N, 'method': method,
                'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'n_params': E_inf['n_params'],
                'Time_sec': res['time_sec'],
                'HW_inference': 'GPU',
                'HW_training': 'GPU',
            })
            print(f"    F={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.2f}µJ, E_train={E_train['E_total_mJ']:.1f}mJ")

            # CNN-Up2D-Paper
            print(f"\n  🔧 CNN-Up2D-Paper")
            model_fn_cnn2 = lambda cd, d_: CNNGen_Up2D_Paper(cond_dim=cd, d=d_)

            res = optimize_with_unified_output(
                model_class=model_fn_cnn2, N=N, method=method, M=M,
                steps=current_steps, lr=lr, warm_spiking=False,
                use_amp=use_amp, is_crossbar=False
            )
            model = res['model']
            E_inf = estimate_gpu_inference_energy(model)
            E_train = estimate_gpu_training_energy(model, current_steps, measured_time_s=res['time_sec'])

            all_results['CNN'][method].append({
                'Model': 'CNN-Up2D-Paper', 'N': N, 'method': method,
                'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'n_params': E_inf['n_params'],
                'Time_sec': res['time_sec'],
                'HW_inference': 'GPU',
                'HW_training': 'GPU',
            })
            print(f"    F={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.2f}µJ, E_train={E_train['E_total_mJ']:.1f}mJ")

            # ================================================================
            # 2. SCNN-Norse (Loihi target)
            # ================================================================
            print(f"\n2️⃣  SCNN-Norse (Loihi)")

            # SCNN-Norse-Simple2D (usa parametri standard)
            print(f"\n  🔧 SCNN-Norse-Simple2D")
            model_fn_norse1 = lambda cd, d_: SCNNGen_LIF_Simple2D(
                cond_dim=cd, d=d_, proj_hw=(N*2, N*2), ch=(32,64),
                T=current_T, v_th=current_vth, enc_gamma=current_gamma)

            res = optimize_with_unified_output(
                model_class=model_fn_norse1, N=N, method=method, M=M,
                steps=current_steps, lr=lr, warm_spiking=True,
                use_amp=use_amp, is_crossbar=False
            )
            model = res['model']
            E_inf = estimate_loihi_inference_energy(model, T=current_T, sparsity=sparsity)
            E_train = estimate_gpu_training_energy(model, current_steps, measured_time_s=res['time_sec'])

            all_results['SCNN-Norse'][method].append({
                'Model': 'SCNN-Norse-Simple2D', 'N': N, 'method': method,
                'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'n_params': E_inf['n_params'],
                'n_neurons': E_inf['n_neurons'],
                'n_spikes': E_inf['n_spikes'],
                'Time_sec': res['time_sec'],
                'HW_inference': 'Loihi',
                'HW_training': 'GPU',
                'T': current_T,
                'sparsity': sparsity,
                'breakdown': E_inf['breakdown'],
            })
            print(f"    F={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ (Loihi), E_train={E_train['E_total_mJ']:.1f}mJ")

            # ═══════════════════════════════════════════════════════════════
            # SCNN-Norse-Up2D (FIX: usa parametri up2d_*)
            # ═══════════════════════════════════════════════════════════════
            print(f"\n  🔧 SCNN-Norse-Up2D")
            print(f"    Params: T={up2d_T}, v_th={up2d_vth}, gamma={up2d_gamma}")

            # Cattura i parametri in variabili locali per la closure lambda
            _up2d_T = up2d_T
            _up2d_vth = up2d_vth
            _up2d_gamma = up2d_gamma
            _up2d_tau_mem = up2d_tau_mem
            _up2d_tau_syn = up2d_tau_syn

            model_fn_norse2 = lambda cd, d_: SCNNGen_LIF_Up2D_Paper(
                cond_dim=cd, d=d_,
                T=_up2d_T,
                v_th=_up2d_vth,
                tau_mem_inv=_up2d_tau_mem,
                tau_syn_inv=_up2d_tau_syn,
                enc_gamma=_up2d_gamma)

            res = optimize_with_unified_output(
                model_class=model_fn_norse2, N=N, method=method, M=M,
                steps=current_steps, lr=lr, warm_spiking=True,
                use_amp=use_amp, is_crossbar=False
            )
            model = res['model']
            # FIX: usa up2d_T per la stima energia
            E_inf = estimate_loihi_inference_energy(model, T=up2d_T, sparsity=sparsity)
            E_train = estimate_gpu_training_energy(model, current_steps, measured_time_s=res['time_sec'])

            all_results['SCNN-Norse'][method].append({
                'Model': 'SCNN-Norse-Up2D', 'N': N, 'method': method,
                'F_best': res['F_best'],
                'E_inference_uJ': E_inf['E_total_uJ'],
                'E_training_mJ': E_train['E_total_mJ'],
                'n_params': E_inf['n_params'],
                'n_neurons': E_inf['n_neurons'],
                'n_spikes': E_inf['n_spikes'],
                'Time_sec': res['time_sec'],
                'HW_inference': 'Loihi',
                'HW_training': 'GPU',
                'T': up2d_T,  # FIX: salva il T corretto
                'sparsity': sparsity,
                'breakdown': E_inf['breakdown'],
            })
            print(f"    F={res['F_best']:.4f}, E_inf={E_inf['E_total_uJ']:.4f}µJ (Loihi), E_train={E_train['E_total_mJ']:.1f}mJ")

            # --- Free GPU memory before crossbar section ---
            gpu_cleanup()

            # ================================================================
            # 3. SCNN-Crossbar
            # ================================================================
            print(f"\n3️⃣  SCNN-Crossbar (Memristor)")

            for bits in crossbar_bits:
                print(f"\n  📊 {bits}-bit quantization")

                # Crea model_fn con bits catturato correttamente
                def make_crossbar_model_fn(bits_val, N_val, T_val):
                    # Sweep-validated crossbar params
                    if bits_val <= 4:
                        actual_vth = 0.3    # Crossbar-4b (T-sweep / M-sweep)
                        actual_gamma = 3.0  # Crossbar-4b (T-sweep / M-sweep)
                    else:
                        actual_vth = 0.5    # Crossbar-8b (T-sweep / M-sweep)
                        actual_gamma = 2.25 # Crossbar-8b (T-sweep / M-sweep)

                    def model_fn(cd, d_):
                        return SCNNGen_Crossbar_Simple2D(
                            cond_dim=cd, d=d_,
                            proj_hw=(N_val*2, N_val*2), ch=(32,64),
                            T=T_val, v_th=actual_vth, enc_gamma=actual_gamma,
                            weight_bits=bits_val, adc_bits=bits_val, dac_bits=bits_val
                        )
                    return model_fn

                model_fn_crossbar = make_crossbar_model_fn(bits, N, current_T)

                print(f"\n    🔧 SCNN-Crossbar-Simple2D-{bits}b")

                # ═══════════════════════════════════════════════════════════
                # FIX: usa parametri adattivi per 8-bit
                # ═══════════════════════════════════════════════════════════
                if bits == 8:
                    _variation = cb8_variation
                    _noise = cb8_noise
                    _steps = cb8_steps
                else:
                    _variation = 0.02
                    _noise = 0.01
                    _steps = current_steps

                res = optimize_single_state_crossbar(
                    model_class=model_fn_crossbar,
                    N=N, method=method, M=M,
                    steps=_steps, lr=lr,
                    warm_spiking=True,
                    weight_bits=bits, adc_bits=bits, dac_bits=bits,
                    device_variation=_variation, read_noise=_noise,
                    use_amp=use_amp, log_energy=False
                )

                model = res['model']

                # Energie - usa il metodo del modello crossbar
                E_inf_crossbar = model.get_model_energy(batch_size=1, input_sparsity=sparsity)
                E_inf = estimate_crossbar_inference_energy(model, T=current_T, sparsity=sparsity, bits=bits)
                E_train = estimate_gpu_training_energy(model, _steps, measured_time_s=res['time_sec'])
                E_prog = estimate_weight_programming_energy(model)

                all_results['SCNN-Crossbar'][f'{bits}bit'][method].append({
                    'Model': f'SCNN-Crossbar-Simple2D-{bits}b', 'N': N, 'method': method,
                    'F_best': res['F_best'],
                    'E_inference_uJ': E_inf['E_total_uJ'],
                    'E_inference_model_uJ': E_inf_crossbar['E_total_mJ'] * 1e3,  # dal modello
                    'E_training_mJ': E_train['E_total_mJ'],
                    'E_programming_mJ': E_prog['E_total_mJ'],
                    'n_params': E_inf['n_params'],
                    'bits': bits,
                    'Time_sec': res.get('time_sec', 0),
                    'HW_inference': f'Crossbar-{bits}b',
                    'HW_training': 'GPU',
                    'T': current_T,
                    'sparsity': sparsity,
                    'breakdown': E_inf['breakdown'],
                })

                print(f"      F={res['F_best']:.4f}")
                print(f"      E_inference={E_inf['E_total_uJ']:.4f}µJ (formula), {E_inf_crossbar['E_total_mJ']*1e3:.4f}µJ (model)")
                print(f"      E_training={E_train['E_total_mJ']:.1f}mJ (GPU)")
                print(f"      E_programming={E_prog['E_total_mJ']:.3f}mJ (one-time)")
                print(f"      ADC dominance: {E_inf['breakdown']['ADC_%']:.1f}%")

                # --- Free crossbar model GPU memory ---
                del model, res
                gpu_cleanup()

    return all_results

# =============================================================================
# CELLA 4: ANALISI E VISUALIZZAZIONE (sostituisce analyze_complete_results)
# =============================================================================

def analyze_energy_breakdown(results):
    """
    Analizza risultati con focus su inference vs training.
    """
    print("\n" + "="*80)
    print("📊 ANALISI ENERGETICA: INFERENCE vs TRAINING")
    print("="*80)

    for method in results['CNN'].keys():
        print(f"\n{'='*80}")
        print(f"METHOD: {method}")
        print(f"{'='*80}")

        # Raccogli tutti i dati
        all_data = []

        # CNN
        for r in results['CNN'][method]:
            all_data.append({
                'seed': SEED,
                'Category': 'CNN',
                'Model': r['Model'],
                'N': r['N'],
                'F': r['F_best'],
                'E_inf_uJ': r['E_inference_uJ'],
                'E_train_mJ': r['E_training_mJ'],
                'E_prog_mJ': 0,
                'HW_inf': 'GPU',
            })

        # SCNN-Norse
        for r in results['SCNN-Norse'][method]:
            all_data.append({
                'seed': SEED,
                'Category': 'SCNN-Norse',
                'Model': r['Model'],
                'N': r['N'],
                'F': r['F_best'],
                'E_inf_uJ': r['E_inference_uJ'],
                'E_train_mJ': r['E_training_mJ'],
                'E_prog_mJ': 0,
                'HW_inf': 'Loihi',
            })

        # SCNN-Crossbar
        for bits_key in results['SCNN-Crossbar'].keys():
            if method in results['SCNN-Crossbar'][bits_key]:
                for r in results['SCNN-Crossbar'][bits_key][method]:
                    all_data.append({
                        'seed': SEED,
                        'Category': f'Crossbar-{r["bits"]}b',
                        'Model': r['Model'],
                        'N': r['N'],
                        'F': r['F_best'],
                        'E_inf_uJ': r['E_inference_uJ'],
                        'E_train_mJ': r['E_training_mJ'],
                        'E_prog_mJ': r['E_programming_mJ'],
                        'HW_inf': f'Crossbar-{r["bits"]}b',
                    })

        df = pd.DataFrame(all_data)

        # Per ogni N, mostra confronto
        for N in sorted(df['N'].unique()):
            df_N = df[df['N'] == N].copy()

            # Baseline: CNN GPU
            E_inf_baseline = df_N[df_N['Category'] == 'CNN']['E_inf_uJ'].min()

            df_N['Speedup_inf'] = E_inf_baseline / df_N['E_inf_uJ']
            df_N['F_drop_%'] = (df_N[df_N['Category']=='CNN']['F'].max() - df_N['F']) * 100

            print(f"\n{'─'*80}")
            print(f"N = {N} qubits")
            print(f"{'─'*80}")
            print(f"\n{'Category':<18} {'Model':<28} {'F':<8} {'E_inf(µJ)':<12} {'Speedup':<10} {'E_train(mJ)':<12} {'HW_inf'}")
            print("-"*110)

            for _, row in df_N.iterrows():
                print(f"{row['Category']:<18} {row['Model']:<28} {row['F']:.4f}   "
                      f"{row['E_inf_uJ']:<12.4f} {row['Speedup_inf']:<10.0f}× "
                      f"{row['E_train_mJ']:<12.1f} {row['HW_inf']}")

        # Summary
        print(f"\n{'='*80}")
        print("📈 SUMMARY")
        print(f"{'='*80}")
        print("\n⚡ INFERENCE SPEEDUP (rispetto a CNN su GPU):")

        for cat in ['SCNN-Norse', 'Crossbar-8b', 'Crossbar-4b']:
            cat_data = df[df['Category'] == cat]
            if len(cat_data) > 0:
                avg_speedup = E_inf_baseline / cat_data['E_inf_uJ'].mean()
                print(f"   {cat}: {avg_speedup:.0f}× media")

        print("\n📋 NOTE:")
        print("   • E_inference: energia per SINGOLA inference sul target HW")
        print("   • E_training: energia per training completo (sempre su GPU)")
        print("   • Speedup: calcolato su E_inference (confronto fair)")
        print("   • Il training domina il costo totale, ma è one-time!")

    return df


def plot_energy_comparison(results, N=3):
    """
    Visualizza confronto energetico per un dato N.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raccogli dati per N specifico
    categories = []
    E_inference = []
    E_training = []
    fidelities = []

    for method in list(results['CNN'].keys())[:1]:  # Solo primo metodo
        for r in results['CNN'][method]:
            if r['N'] == N:
                categories.append('CNN-GPU')
                E_inference.append(r['E_inference_uJ'])
                E_training.append(r['E_training_mJ'])
                fidelities.append(r['F_best'])
                break

        for r in results['SCNN-Norse'][method]:
            if r['N'] == N:
                categories.append('SCNN-Loihi')
                E_inference.append(r['E_inference_uJ'])
                E_training.append(r['E_training_mJ'])
                fidelities.append(r['F_best'])
                break

        for bits_key in ['8bit', '4bit']:
            if bits_key in results['SCNN-Crossbar']:
                for r in results['SCNN-Crossbar'][bits_key][method]:
                    if r['N'] == N:
                        categories.append(f'Crossbar-{bits_key}')
                        E_inference.append(r['E_inference_uJ'])
                        E_training.append(r['E_training_mJ'])
                        fidelities.append(r['F_best'])
                        break

    # Plot 1: Inference Energy (log scale)
    ax1 = axes[0]
    colors = ['blue', 'green', 'orange', 'red']
    bars1 = ax1.bar(categories, E_inference, color=colors[:len(categories)])
    ax1.set_ylabel('Energy (µJ)')
    ax1.set_title(f'Inference Energy (N={N})')
    ax1.set_yscale('log')
    for bar, val in zip(bars1, E_inference):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Training Energy
    ax2 = axes[1]
    bars2 = ax2.bar(categories, E_training, color=colors[:len(categories)])
    ax2.set_ylabel('Energy (mJ)')
    ax2.set_title(f'Training Energy (N={N}) - Always on GPU')
    for bar, val in zip(bars2, E_training):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    # Plot 3: Fidelity
    ax3 = axes[2]
    bars3 = ax3.bar(categories, fidelities, color=colors[:len(categories)])
    ax3.set_ylabel('Fidelity')
    ax3.set_title(f'Reconstruction Fidelity (N={N})')
    ax3.set_ylim([0.9, 1.01])
    for bar, val in zip(bars3, fidelities):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('energy_comparison.png', dpi=150, bbox_inches='tight')

    # Speedup summary
    baseline = E_inference[0]
    print(f"\n📊 INFERENCE SPEEDUP vs CNN-GPU (N={N}):")
    for cat, E in zip(categories, E_inference):
        print(f"   {cat}: {baseline/E:.0f}×")


print("✓ Funzioni analisi definite")
print("\n" + "="*70)
print("📋 CELLE PRONTE PER IL NOTEBOOK")
print("="*70)



# =============================================================================
# CELLA DA AGGIUNGERE: estimate_weight_programming_energy
# =============================================================================

def estimate_weight_programming_energy(model):
    """
    Energia one-time per programmare pesi nel crossbar (deployment).
    """
    n_params = sum(p.numel() for p in model.parameters())
    E_total = n_params * MEMRISTOR_PARAMS['E_write']
    return {
        'E_total_J': E_total,
        'E_total_mJ': E_total * 1e3,
        'n_params': n_params,
    }


# === cell #17 ===
# =============================================================================
# RUN COMPLETE BENCHMARK M1
# =============================================================================

print("\n" + "="*80)
print("🚀 STARTING COMPLETE BENCHMARK")
print("="*80)
print("\nThis will compare:")
print("  1. CNN (standard, non-spiking)")
print("  2. SCNN-Norse (spiking, no crossbar)")
print("  3. SCNN-Crossbar (spiking + memristor, 8-bit and 4-bit)")
print("\n" + "="*80)

# Run benchmark
if _seed_done('M1'):
    print(f"[multiseed] seed={SEED} method=M1 already done — skipping", flush=True)
    results_complete_M1 = None
else:
    if _ARGS.quick:
        results_complete_M1 = run_benchmark_with_energy_breakdown(
            N_list=[3],
            methods=['M1'],
            M_M1=256,           # M-sweep: saturates at F~0.83 for N=8
            M_M2=4,             # M-sweep: sufficient for Norse/Crossbar-4b
            steps=3,
            lr=1e-3,
            crossbar_bits=[8, 4],
            T=8,                # T-sweep: optimal sweet spot
            use_amp=True
        )

    else:
        results_complete_M1 = run_benchmark_with_energy_breakdown(
            N_list=[3, 4, 5, 6, 7, 8],
            methods=['M1'],
            M_M1=256,           # M-sweep: saturates at F~0.83 for N=8
            M_M2=4,             # M-sweep: sufficient for Norse/Crossbar-4b
            steps=500,
            lr=1e-3,
            crossbar_bits=[8, 4],
            T=8,                # T-sweep: optimal sweet spot
            use_amp=True
        )

    _save_with_seed(results_complete_M1, 'M1', analyze_energy_breakdown)
# Analyze results
df_analysis = analyze_energy_breakdown(results_complete_M1)

# Plot comparisons
plot_energy_comparison(results_complete_M1)

print("\n" + "="*80)
print("✅ COMPLETE BENCHMARK FINISHED!")
print("="*80)


# === cell #18 ===
# =============================================================================
# RUN COMPLETE BENCHMARK M2
# =============================================================================

print("\n" + "="*80)
print("🚀 STARTING COMPLETE BENCHMARK")
print("="*80)
print("\nThis will compare:")
print("  1. CNN (standard, non-spiking)")
print("  2. SCNN-Norse (spiking, no crossbar)")
print("  3. SCNN-Crossbar (spiking + memristor, 8-bit and 4-bit)")
print("\n" + "="*80)

# Run benchmark
if _seed_done('M2'):
    print(f"[multiseed] seed={SEED} method=M2 already done — skipping", flush=True)
    results_complete_M2 = None
else:
    if _ARGS.quick:
        results_complete_M2 = run_benchmark_with_energy_breakdown(
            N_list=[3],
            methods=['M2'],
            M_M1=256,           # M-sweep: saturates at F~0.83 for N=8
            M_M2=4,             # M-sweep: sufficient for Norse/Crossbar-4b
            steps=3,
            lr=1e-3,
            crossbar_bits=[8, 4],
            T=8,                # T-sweep: optimal sweet spot
            use_amp=True
        )

    else:
        results_complete_M2 = run_benchmark_with_energy_breakdown(
            N_list=[3, 4, 5, 6, 7, 8],
            methods=['M2'],
            M_M1=256,           # M-sweep: saturates at F~0.83 for N=8
            M_M2=4,             # M-sweep: sufficient for Norse/Crossbar-4b
            steps=500,
            lr=1e-3,
            crossbar_bits=[8, 4],
            T=8,                # T-sweep: optimal sweet spot
            use_amp=True
        )

    _save_with_seed(results_complete_M2, 'M2', analyze_energy_breakdown)
# Analyze results
df_analysis = analyze_energy_breakdown(results_complete_M2)

# Plot comparisons
plot_energy_comparison(results_complete_M2)

print("\n" + "="*80)
print("✅ COMPLETE BENCHMARK FINISHED!")
print("="*80)


# === cell #19 ===
# =============================================================================
# CELLA PLOTTING FIXATA - Corretto bug Speedup + Diagnostica migliorata
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_complete_benchmark(results, save_prefix='benchmark'):
    """
    Genera plot completi dei risultati del benchmark.

    FIXES:
    - Corretto bug grafico Speedup (ora funziona correttamente)
    - Aggiunta diagnostica per problemi di convergenza
    - Migliorata gestione casi edge (F=0, energie anomale)
    """

    # Colori consistenti per ogni categoria
    COLORS = {
        'CNN-Simple2D': '#1f77b4',
        'CNN-Up2D-Paper': '#aec7e8',
        'SCNN-Norse-Simple2D': '#2ca02c',
        'SCNN-Norse-Up2D': '#98df8a',
        'SCNN-Crossbar-Simple2D-8b': '#ff7f0e',
        'SCNN-Crossbar-Simple2D-4b': '#d62728',
    }

    MARKERS = {
        'CNN-Simple2D': 'o',
        'CNN-Up2D-Paper': 's',
        'SCNN-Norse-Simple2D': '^',
        'SCNN-Norse-Up2D': 'v',
        'SCNN-Crossbar-Simple2D-8b': 'D',
        'SCNN-Crossbar-Simple2D-4b': 'p',
    }

    # Raccogli dati per ogni metodo
    for method in results['CNN'].keys():
        print(f"\n{'='*60}")
        print(f"📊 Generating plots for METHOD: {method}")
        print(f"{'='*60}")

        # Organizza dati per modello
        data_by_model = defaultdict(lambda: {'N': [], 'F': [], 'E_inf': [], 'E_train': []})

        # CNN
        for r in results['CNN'][method]:
            model = r['Model']
            data_by_model[model]['N'].append(int(r['N']))  # FIX: forza int
            data_by_model[model]['F'].append(float(r['F_best']))
            data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
            data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

        # SCNN-Norse
        for r in results['SCNN-Norse'][method]:
            model = r['Model']
            data_by_model[model]['N'].append(int(r['N']))
            data_by_model[model]['F'].append(float(r['F_best']))
            data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
            data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

        # Crossbar
        for bits_key in results['SCNN-Crossbar'].keys():
            if method in results['SCNN-Crossbar'][bits_key]:
                for r in results['SCNN-Crossbar'][bits_key][method]:
                    model = r['Model']
                    data_by_model[model]['N'].append(int(r['N']))
                    data_by_model[model]['F'].append(float(r['F_best']))
                    data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
                    data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

        # Ordina per N
        for model in data_by_model:
            idx = np.argsort(data_by_model[model]['N'])
            for key in ['N', 'F', 'E_inf', 'E_train']:
                data_by_model[model][key] = [data_by_model[model][key][i] for i in idx]

        # =====================================================================
        # Prepara baseline per Speedup (FIX: costruisci correttamente)
        # =====================================================================
        baseline_E = {}
        if 'CNN-Simple2D' in data_by_model:
            for n, e in zip(data_by_model['CNN-Simple2D']['N'],
                          data_by_model['CNN-Simple2D']['E_inf']):
                baseline_E[int(n)] = float(e)
            print(f"   Baseline energies (CNN-Simple2D): {baseline_E}")
        print(f"   Available models: {list(data_by_model.keys())}")
        print(f"   data_by_model N values: {[(m, data_by_model[m]['N']) for m in data_by_model]}")

        # =====================================================================
        # FIGURA 1: Dashboard completa (2x3 con Infidelity)
        # =====================================================================
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Quantum State Tomography Benchmark - Method {method}',
                    fontsize=14, fontweight='bold')

        # 1. Fidelity vs N
        ax1 = axes[0, 0]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            ax1.plot(data['N'], data['F'], marker=marker, color=color,
                    label=model, linewidth=2, markersize=8)
        ax1.set_xlabel('N qubits', fontsize=11)
        ax1.set_ylabel('Fidelity', fontsize=11)
        ax1.set_title('Fidelity vs System Size', fontsize=12)
        ax1.set_ylim([0, 1.05])
        ax1.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='F=0.99')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left', fontsize=7)

        # 2. Infidelity (1-F) vs N in scala log
        ax2 = axes[0, 1]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            infidelity = [max(1 - f, 1e-6) for f in data['F']]
            ax2.semilogy(data['N'], infidelity, marker=marker, color=color,
                        label=model, linewidth=2, markersize=8)
        ax2.set_xlabel('N qubits', fontsize=11)
        ax2.set_ylabel('Infidelity (1 - F)', fontsize=11)
        ax2.set_title('Infidelity vs System Size (log scale)', fontsize=12)
        ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='1%')
        ax2.axhline(y=0.001, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='0.1%')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(loc='upper left', fontsize=7)
        ax2.set_ylim([1e-6, 1.5])

        # 3. Inference Energy vs N
        ax3 = axes[0, 2]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            ax3.semilogy(data['N'], data['E_inf'], marker=marker, color=color,
                        label=model, linewidth=2, markersize=8)
        ax3.set_xlabel('N qubits', fontsize=11)
        ax3.set_ylabel('Inference Energy (µJ)', fontsize=11)
        ax3.set_title('Inference Energy vs System Size', fontsize=12)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.legend(loc='upper left', fontsize=7)

        # 4. Speedup vs N (FIXED!)
        ax4 = axes[1, 0]
        has_speedup_data = False

        for model, data in data_by_model.items():
            if model.startswith('SCNN'):  # Solo modelli neuromorphic (SCNN)
                print(f"     N values: {data['N']}, E_inf values: {data['E_inf']}")
                color = COLORS.get(model, 'gray')
                marker = MARKERS.get(model, 'o')
                speedups = []
                N_valid = []

                for n, e in zip(data['N'], data['E_inf']):
                    n_int = int(n)
                    print(f"     Checking n={n}, e={e}: n_int={n_int}, in_baseline={n_int in baseline_E}, e>0={e>0}")
                    if n_int in baseline_E and e > 0 and baseline_E[n_int] > 0:
                        speedup = baseline_E[n_int] / e
                        speedups.append(speedup)
                        N_valid.append(n_int)

                if speedups:
                    has_speedup_data = True
                    ax4.semilogy(N_valid, speedups, marker=marker, color=color,
                                label=model, linewidth=2, markersize=8)

        ax4.set_xlabel('N qubits', fontsize=11)
        ax4.set_ylabel('Speedup vs CNN-GPU', fontsize=11)
        ax4.set_title('Energy Efficiency Gain', fontsize=12)
        ax4.grid(True, alpha=0.3, which='both')

        if has_speedup_data:
            ax4.legend(loc='upper left', fontsize=7)
            # Imposta limiti ragionevoli
            ax4.set_xlim([min(baseline_E.keys())-0.5, max(baseline_E.keys())+0.5])
        else:
            ax4.text(0.5, 0.5, 'No speedup data available',
                    transform=ax4.transAxes, ha='center', va='center',
                    fontsize=12, color='red')

        # 5. Training Energy vs N
        ax5 = axes[1, 1]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            E_train_J = [e / 1000 for e in data['E_train']]  # mJ → J
            ax5.semilogy(data['N'], E_train_J, marker=marker, color=color,
                        label=model, linewidth=2, markersize=8)
        ax5.set_xlabel('N qubits', fontsize=11)
        ax5.set_ylabel('Training Energy (J)', fontsize=11)
        ax5.set_title('Training Energy vs System Size (GPU)', fontsize=12)
        ax5.grid(True, alpha=0.3, which='both')
        ax5.legend(loc='upper left', fontsize=7)

        # 6. Efficiency (F/E) vs N - solo per F > 0.5
        ax6 = axes[1, 2]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            efficiency = []
            N_valid = []
            for n, f, e in zip(data['N'], data['F'], data['E_inf']):
                if e > 0 and f > 0.5:
                    efficiency.append(f / e)
                    N_valid.append(n)
            if efficiency:
                ax6.semilogy(N_valid, efficiency, marker=marker, color=color,
                            label=model, linewidth=2, markersize=8)
        ax6.set_xlabel('N qubits', fontsize=11)
        ax6.set_ylabel('Efficiency (F / µJ)', fontsize=11)
        ax6.set_title('Energy Efficiency (only F > 0.5)', fontsize=12)
        ax6.grid(True, alpha=0.3, which='both')
        ax6.legend(loc='upper right', fontsize=7)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{method}_dashboard.png', dpi=150, bbox_inches='tight')

        # =====================================================================
        # FIGURA 2: Infidelity Analysis Dettagliata
        # =====================================================================
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Infidelity Analysis - Method {method}', fontsize=13, fontweight='bold')

        # Left: Infidelity vs N con zone colorate
        ax1 = axes[0]
        for model, data in data_by_model.items():
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            infidelity = [max(1 - f, 1e-6) for f in data['F']]
            ax1.semilogy(data['N'], infidelity, marker=marker, color=color,
                        label=model, linewidth=2, markersize=8)

        ax1.set_xlabel('N qubits', fontsize=11)
        ax1.set_ylabel('Infidelity (1 - F)', fontsize=11)
        ax1.set_title('Infidelity vs System Size', fontsize=12)

        # Zone colorate per indicare quality
        N_min = min(min(d['N']) for d in data_by_model.values())
        N_max = max(max(d['N']) for d in data_by_model.values())
        N_range = [N_min - 0.5, N_max + 0.5]

        ax1.fill_between(N_range, [1, 1], [1.5, 1.5], alpha=0.15, color='darkred')
        ax1.fill_between(N_range, [0.1, 0.1], [1, 1], alpha=0.1, color='red')
        ax1.fill_between(N_range, [0.01, 0.01], [0.1, 0.1], alpha=0.1, color='orange')
        ax1.fill_between(N_range, [0.001, 0.001], [0.01, 0.01], alpha=0.1, color='yellow')
        ax1.fill_between(N_range, [1e-6, 1e-6], [0.001, 0.001], alpha=0.1, color='green')

        ax1.axhline(y=0.01, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(loc='upper left', fontsize=7, ncol=2)
        ax1.set_ylim([1e-6, 1.5])
        ax1.set_xlim(N_range)

        # Annotazioni zone
        ax1.text(N_max + 0.6, 0.5, 'F<0.9', fontsize=8, color='red', va='center')
        ax1.text(N_max + 0.6, 0.03, 'F<0.99', fontsize=8, color='orange', va='center')
        ax1.text(N_max + 0.6, 0.003, 'F<0.999', fontsize=8, color='olive', va='center')
        ax1.text(N_max + 0.6, 0.0001, 'F≥0.999', fontsize=8, color='green', va='center')

        # Right: Bar chart per N selezionati
        ax2 = axes[1]
        N_values = sorted(set(data_by_model['CNN-Simple2D']['N']))
        N_selected = [n for n in N_values if n in [3, 4, 5, 6, 7, 8]][:4]

        x = np.arange(len(N_selected))
        width = 0.13
        models_to_plot = ['CNN-Simple2D', 'SCNN-Norse-Simple2D', 'SCNN-Norse-Up2D',
                         'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']

        for i, model in enumerate(models_to_plot):
            if model in data_by_model:
                infidelities = []
                for N in N_selected:
                    if N in data_by_model[model]['N']:
                        idx = data_by_model[model]['N'].index(N)
                        f = data_by_model[model]['F'][idx]
                        infidelities.append(max(1 - f, 1e-6))
                    else:
                        infidelities.append(np.nan)

                color = COLORS.get(model, 'gray')
                offset = (i - len(models_to_plot)/2 + 0.5) * width
                short_name = model.replace('SCNN-', '').replace('-Simple2D', '')
                ax2.bar(x + offset, infidelities, width, label=short_name,
                       color=color, alpha=0.8)

        ax2.set_yscale('log')
        ax2.set_xlabel('N qubits', fontsize=11)
        ax2.set_ylabel('Infidelity (1 - F)', fontsize=11)
        ax2.set_title('Infidelity by Model at Selected N', fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'N={n}' for n in N_selected])
        ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='1% threshold')
        ax2.legend(loc='upper left', fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3, which='both', axis='y')
        ax2.set_ylim([1e-6, 1.5])

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{method}_infidelity.png', dpi=150, bbox_inches='tight')

        # =====================================================================
        # FIGURA 3: Confronto Hardware
        # =====================================================================
        N_selected_hw = [n for n in N_values if n in [3, 5, 7, 8]]

        if len(N_selected_hw) >= 2:
            fig, axes_hw = plt.subplots(1, len(N_selected_hw), figsize=(4*len(N_selected_hw), 5))
            if len(N_selected_hw) == 1:
                axes_hw = [axes_hw]

            fig.suptitle(f'Inference Energy by Hardware Platform - Method {method}',
                        fontsize=13, fontweight='bold')

            categories = ['CNN\n(GPU)', 'Norse\n(Loihi)', 'Crossbar\n8-bit', 'Crossbar\n4-bit']
            cat_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
            model_keys = ['CNN-Simple2D', 'SCNN-Norse-Simple2D',
                         'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']

            for ax, N in zip(axes_hw, N_selected_hw):
                energies = []
                fidelities = []

                for mk in model_keys:
                    if mk in data_by_model and N in data_by_model[mk]['N']:
                        idx = data_by_model[mk]['N'].index(N)
                        energies.append(data_by_model[mk]['E_inf'][idx])
                        fidelities.append(data_by_model[mk]['F'][idx])
                    else:
                        energies.append(0.001)  # placeholder
                        fidelities.append(0)

                bars = ax.bar(categories, energies, color=cat_colors)
                ax.set_ylabel('Energy (µJ)' if N == N_selected_hw[0] else '')
                ax.set_title(f'N = {N} qubits\n(d = {2**N})', fontsize=11)
                ax.set_yscale('log')

                # Annota con fidelity
                for bar, f in zip(bars, fidelities):
                    height = bar.get_height()
                    color = 'red' if f < 0.9 else ('orange' if f < 0.99 else 'black')
                    fontweight = 'bold' if f < 0.9 else 'normal'
                    ax.annotate(f'F={f:.2f}',
                               xy=(bar.get_x() + bar.get_width()/2, height),
                               xytext=(0, 3), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8,
                               rotation=0, color=color, fontweight=fontweight)

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
            plt.savefig(f'{save_prefix}_{method}_hardware_comparison.png', dpi=150, bbox_inches='tight')

        # =====================================================================
        # FIGURA 4: Scaling Analysis
        # =====================================================================
        fig, axes_sc = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Scaling Analysis - Method {method}', fontsize=13, fontweight='bold')

        # Left: Energy scaling (log-log)
        ax1 = axes_sc[0]
        for model, data in data_by_model.items():
            if 'Simple2D' in model:
                color = COLORS.get(model, 'gray')
                marker = MARKERS.get(model, 'o')
                d_values = [2**n for n in data['N']]
                ax1.loglog(d_values, data['E_inf'], marker=marker, color=color,
                          label=model, linewidth=2, markersize=8)

        d_ref = np.array([8, 256])
        ax1.loglog(d_ref, 0.1 * (d_ref/8)**2, 'k--', alpha=0.3, label='O(d²)')
        ax1.loglog(d_ref, 0.1 * (d_ref/8)**3, 'k:', alpha=0.3, label='O(d³)')

        ax1.set_xlabel('Hilbert space dimension d', fontsize=11)
        ax1.set_ylabel('Inference Energy (µJ)', fontsize=11)
        ax1.set_title('Energy Scaling with System Size', fontsize=12)
        ax1.grid(True, alpha=0.3, which='both')
        ax1.legend(loc='upper left', fontsize=8)

        # Right: Efficiency (Fidelity per µJ)
        ax2 = axes_sc[1]
        for model, data in data_by_model.items():
            if 'Simple2D' in model:
                color = COLORS.get(model, 'gray')
                marker = MARKERS.get(model, 'o')
                efficiency = [f / e if e > 0 and f > 0.5 else np.nan
                             for f, e in zip(data['F'], data['E_inf'])]
                ax2.semilogy(data['N'], efficiency, marker=marker, color=color,
                            label=model, linewidth=2, markersize=8)

        ax2.set_xlabel('N qubits', fontsize=11)
        ax2.set_ylabel('Efficiency (Fidelity / µJ)', fontsize=11)
        ax2.set_title('Energy Efficiency vs System Size', fontsize=12)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{method}_scaling.png', dpi=150, bbox_inches='tight')

        # =====================================================================
        # Print Summary Statistics
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"📈 SUMMARY STATISTICS - {method}")
        print(f"{'='*60}")

        # Problem analysis
        print(f"\n⚠️  PROBLEM ANALYSIS:")
        print(f"   Models with F = 0 (completely failed):")
        has_problems = False
        for model, data in data_by_model.items():
            zeros = [(n, f) for n, f in zip(data['N'], data['F']) if f < 0.01]
            if zeros:
                has_problems = True
                for n, f in zeros:
                    print(f"      ❌ {model} @ N={n}: F={f:.4f}")
        if not has_problems:
            print(f"      ✅ None!")

        print(f"\n   Models with F < 0.9 (poor performance):")
        has_problems = False
        for model, data in data_by_model.items():
            poor = [(n, f) for n, f in zip(data['N'], data['F']) if 0.01 <= f < 0.9]
            if poor:
                has_problems = True
                for n, f in poor:
                    print(f"      ⚠️  {model} @ N={n}: F={f:.4f}")
        if not has_problems:
            print(f"      ✅ None!")

        print(f"\n   Models with F < 0.99 (below target):")
        for model, data in data_by_model.items():
            below = [(n, f) for n, f in zip(data['N'], data['F']) if 0.9 <= f < 0.99]
            if below:
                for n, f in below:
                    print(f"      ⚡ {model} @ N={n}: F={f:.4f}")

        # Best performers at max N
        max_N = max(N_values)
        print(f"\n🏆 Best performers at N={max_N} (d={2**max_N}):")

        best_fidelity = []
        for model, data in data_by_model.items():
            if max_N in data['N']:
                idx = data['N'].index(max_N)
                F = data['F'][idx]
                E = data['E_inf'][idx]
                best_fidelity.append((model, F, E))

        print("\n  By Fidelity:")
        for model, F, E in sorted(best_fidelity, key=lambda x: -x[1])[:4]:
            print(f"    {model}: F={F:.4f}, E={E:.2f}µJ")

        print("\n  By Efficiency (F/E) with F≥0.99:")
        high_fid = [(m, f/e, f, e) for m, f, e in best_fidelity if f >= 0.99 and e > 0]
        for model, eff, F, E in sorted(high_fid, key=lambda x: -x[1])[:3]:
            print(f"    {model}: {eff:.2f} F/µJ (F={F:.4f}, E={E:.4f}µJ)")

        # Average speedups
        print("\n  Average Speedup vs CNN-GPU (all N, only F≥0.9):")
        for model in ['SCNN-Norse-Simple2D', 'SCNN-Norse-Up2D',
                     'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']:
            if model in data_by_model:
                speedups = []
                for n, e, f in zip(data_by_model[model]['N'],
                                  data_by_model[model]['E_inf'],
                                  data_by_model[model]['F']):
                    n_int = int(n)
                    if n_int in baseline_E and e > 0 and f >= 0.9:
                        speedups.append(baseline_E[n_int] / e)
                if speedups:
                    print(f"    {model}: {np.mean(speedups):.0f}× (range: {min(speedups):.0f}-{max(speedups):.0f}×)")
                else:
                    print(f"    {model}: No valid speedup data (F<0.9 for all N)")

    print("\n" + "="*60)
    print("✅ All plots saved!")
    print("="*60)

    return data_by_model


# === cell #20 ===
# =============================================================================
# VISUALIZATION M1
# =============================================================================
# Uncomment the line below to generate plots:
data = plot_complete_benchmark(results_complete_M1, save_prefix='qst_benchmark')


# === cell #21 ===
# =============================================================================
# VISUALIZATION M2
# =============================================================================
# Uncomment the line below to generate plots:
data = plot_complete_benchmark(results_complete_M2, save_prefix='qst_benchmark')


# === cell #22 ===
# =============================================================================
# CELLA PLOTTING AGGIORNATA - Labels corrette + Nuovi grafici
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# =============================================================================
# MAPPING LABELS
# =============================================================================

# Labels per grafici con tutti i modelli
LABELS_ALL_MODELS = {
    'CNN-Simple2D': 'CNN Simple',
    'CNN-Up2D-Paper': 'CNN Up2D',
    'SCNN-Norse-Simple2D': 'SCNN Simple (Loihi)',
    'SCNN-Norse-Up2D': 'SCNN Up2D (Loihi)',
    'SCNN-Crossbar-Simple2D-4b': 'SCNN Simple (crossbar 4-bit)',
    'SCNN-Crossbar-Simple2D-8b': 'SCNN Simple (crossbar 8-bit)',
}

# Labels per grafici confronto hardware (4 modelli)
LABELS_HARDWARE = {
    'CNN-Simple2D': 'CNN (GPU)',
    'SCNN-Norse-Simple2D': 'SCNN (Loihi)',
    'SCNN-Crossbar-Simple2D-4b': 'SCNN (4-bit)',
    'SCNN-Crossbar-Simple2D-8b': 'SCNN (8-bit)',
}

# Colori consistenti
COLORS = {
    'CNN-Simple2D': '#1f77b4',
    'CNN-Up2D-Paper': '#aec7e8',
    'SCNN-Norse-Simple2D': '#2ca02c',
    'SCNN-Norse-Up2D': '#98df8a',
    'SCNN-Crossbar-Simple2D-8b': '#ff7f0e',
    'SCNN-Crossbar-Simple2D-4b': '#d62728',
}

MARKERS = {
    'CNN-Simple2D': 'o',
    'CNN-Up2D-Paper': 's',
    'SCNN-Norse-Simple2D': '^',
    'SCNN-Norse-Up2D': 'v',
    'SCNN-Crossbar-Simple2D-8b': 'D',
    'SCNN-Crossbar-Simple2D-4b': 'p',
}


def get_label(model_key, use_hardware_labels=False):
    """Restituisce la label corretta per il modello."""
    if use_hardware_labels:
        return LABELS_HARDWARE.get(model_key, model_key)
    return LABELS_ALL_MODELS.get(model_key, model_key)


def collect_data_by_model(results):
    """
    Raccoglie i dati per ogni modello da una struttura results.

    Supporta due formati:
    1. Formato con method: results['CNN']['M1'][...]
    2. Formato diretto: results['CNN'][...] (lista di dict)
    """
    data_by_model = defaultdict(lambda: {'N': [], 'F': [], 'E_inf': [], 'E_train': []})

    # Determina il formato della struttura
    def get_records(category_data):
        """Estrae i record da una categoria, gestendo entrambi i formati."""
        if isinstance(category_data, list):
            # Formato diretto: già una lista di record
            return category_data
        elif isinstance(category_data, dict):
            # Formato con method: unisce tutti i metodi
            all_records = []
            for key, value in category_data.items():
                if isinstance(value, list):
                    all_records.extend(value)
            return all_records
        return []

    # CNN
    if 'CNN' in results:
        for r in get_records(results['CNN']):
            model = r['Model']
            data_by_model[model]['N'].append(int(r['N']))
            data_by_model[model]['F'].append(float(r['F_best']))
            data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
            data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

    # SCNN-Norse
    if 'SCNN-Norse' in results:
        for r in get_records(results['SCNN-Norse']):
            model = r['Model']
            data_by_model[model]['N'].append(int(r['N']))
            data_by_model[model]['F'].append(float(r['F_best']))
            data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
            data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

    # Crossbar
    if 'SCNN-Crossbar' in results:
        for bits_key in results['SCNN-Crossbar'].keys():
            crossbar_data = results['SCNN-Crossbar'][bits_key]
            for r in get_records(crossbar_data):
                model = r['Model']
                data_by_model[model]['N'].append(int(r['N']))
                data_by_model[model]['F'].append(float(r['F_best']))
                data_by_model[model]['E_inf'].append(float(r['E_inference_uJ']))
                data_by_model[model]['E_train'].append(float(r['E_training_mJ']))

    # Ordina per N
    for model in data_by_model:
        idx = np.argsort(data_by_model[model]['N'])
        for key in ['N', 'F', 'E_inf', 'E_train']:
            data_by_model[model][key] = [data_by_model[model][key][i] for i in idx]

    return data_by_model


def compute_fidelity_drop(data_by_model, baseline_model='CNN-Simple2D'):
    """
    Calcola il fidelity drop rispetto alla baseline CNN.
    Fidelity drop = F_CNN - F_SCNN (positivo = SCNN peggiore)
    """
    fidelity_drop = {}

    if baseline_model not in data_by_model:
        print(f"⚠️ Baseline {baseline_model} not found!")
        return fidelity_drop

    # Crea lookup per baseline
    baseline_F = {}
    for n, f in zip(data_by_model[baseline_model]['N'],
                    data_by_model[baseline_model]['F']):
        baseline_F[int(n)] = float(f)

    for model, data in data_by_model.items():
        if model == baseline_model:
            continue

        fidelity_drop[model] = {'N': [], 'drop': [], 'drop_pct': []}

        for n, f in zip(data['N'], data['F']):
            n_int = int(n)
            if n_int in baseline_F:
                drop = baseline_F[n_int] - f
                drop_pct = (drop / baseline_F[n_int]) * 100 if baseline_F[n_int] > 0 else 0
                fidelity_drop[model]['N'].append(n_int)
                fidelity_drop[model]['drop'].append(drop)
                fidelity_drop[model]['drop_pct'].append(drop_pct)

    return fidelity_drop


# =============================================================================
# FIGURA: Dashboard principale (aggiornata con nuove labels)
# =============================================================================
def plot_dashboard(data_by_model, method_name='', save_prefix='benchmark'):
    """Dashboard principale con labels aggiornate."""

    # Baseline per speedup
    baseline_E = {}
    if 'CNN-Simple2D' in data_by_model:
        for n, e in zip(data_by_model['CNN-Simple2D']['N'],
                       data_by_model['CNN-Simple2D']['E_inf']):
            baseline_E[int(n)] = float(e)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    title = f'Quantum State Tomography Benchmark'
    if method_name:
        title += f' - {method_name}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Fidelity vs N
    ax1 = axes[0, 0]
    for model, data in data_by_model.items():
        color = COLORS.get(model, 'gray')
        marker = MARKERS.get(model, 'o')
        label = get_label(model)
        ax1.plot(data['N'], data['F'], marker=marker, color=color,
                label=label, linewidth=2, markersize=8)
    ax1.set_xlabel('N qubits', fontsize=11)
    ax1.set_ylabel('Fidelity', fontsize=11)
    ax1.set_title('Fidelity vs System Size', fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='F=0.99')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower left', fontsize=7)

    # 2. Infidelity (1-F) vs N
    ax2 = axes[0, 1]
    for model, data in data_by_model.items():
        color = COLORS.get(model, 'gray')
        marker = MARKERS.get(model, 'o')
        label = get_label(model)
        infidelity = [max(1 - f, 1e-6) for f in data['F']]
        ax2.semilogy(data['N'], infidelity, marker=marker, color=color,
                    label=label, linewidth=2, markersize=8)
    ax2.set_xlabel('N qubits', fontsize=11)
    ax2.set_ylabel('Infidelity (1 - F)', fontsize=11)
    ax2.set_title('Infidelity vs System Size (log scale)', fontsize=12)
    ax2.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='1%')
    ax2.axhline(y=0.001, color='green', linestyle=':', alpha=0.7, linewidth=1.5, label='0.1%')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper left', fontsize=7)
    ax2.set_ylim([1e-6, 1.5])

    # 3. Inference Energy vs N
    ax3 = axes[0, 2]
    for model, data in data_by_model.items():
        color = COLORS.get(model, 'gray')
        marker = MARKERS.get(model, 'o')
        label = get_label(model)
        ax3.semilogy(data['N'], data['E_inf'], marker=marker, color=color,
                    label=label, linewidth=2, markersize=8)
    ax3.set_xlabel('N qubits', fontsize=11)
    ax3.set_ylabel('Inference Energy (µJ)', fontsize=11)
    ax3.set_title('Inference Energy vs System Size', fontsize=12)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(loc='upper left', fontsize=7)

    # 4. Speedup vs N
    ax4 = axes[1, 0]
    has_speedup_data = False

    for model, data in data_by_model.items():
        if model.startswith('SCNN'):  # Solo modelli neuromorphic (SCNN)
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)
            speedups = []
            N_valid = []

            for n, e in zip(data['N'], data['E_inf']):
                n_int = int(n)
                if n_int in baseline_E and e > 0 and baseline_E[n_int] > 0:
                    speedup = baseline_E[n_int] / e
                    speedups.append(speedup)
                    N_valid.append(n_int)

            if speedups:
                has_speedup_data = True
                ax4.semilogy(N_valid, speedups, marker=marker, color=color,
                            label=label, linewidth=2, markersize=8)

    ax4.set_xlabel('N qubits', fontsize=11)
    ax4.set_ylabel('Speedup vs CNN (GPU)', fontsize=11)
    ax4.set_title('Energy Efficiency Gain', fontsize=12)
    ax4.grid(True, alpha=0.3, which='both')

    if has_speedup_data:
        ax4.legend(loc='upper left', fontsize=7)
        ax4.set_xlim([min(baseline_E.keys())-0.5, max(baseline_E.keys())+0.5])
    else:
        ax4.text(0.5, 0.5, 'No speedup data available',
                transform=ax4.transAxes, ha='center', va='center',
                fontsize=12, color='red')

    # 5. Training Energy vs N
    ax5 = axes[1, 1]
    for model, data in data_by_model.items():
        color = COLORS.get(model, 'gray')
        marker = MARKERS.get(model, 'o')
        label = get_label(model)
        E_train_J = [e / 1000 for e in data['E_train']]
        ax5.semilogy(data['N'], E_train_J, marker=marker, color=color,
                    label=label, linewidth=2, markersize=8)
    ax5.set_xlabel('N qubits', fontsize=11)
    ax5.set_ylabel('Training Energy (J)', fontsize=11)
    ax5.set_title('Training Energy vs System Size (GPU)', fontsize=12)
    ax5.grid(True, alpha=0.3, which='both')
    ax5.legend(loc='upper left', fontsize=7)

    # 6. Efficiency (F/E) vs N
    ax6 = axes[1, 2]
    for model, data in data_by_model.items():
        color = COLORS.get(model, 'gray')
        marker = MARKERS.get(model, 'o')
        label = get_label(model)
        efficiency = []
        N_valid = []
        for n, f, e in zip(data['N'], data['F'], data['E_inf']):
            if e > 0 and f > 0.5:
                efficiency.append(f / e)
                N_valid.append(n)
        if efficiency:
            ax6.semilogy(N_valid, efficiency, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)
    ax6.set_xlabel('N qubits', fontsize=11)
    ax6.set_ylabel('Efficiency (F / µJ)', fontsize=11)
    ax6.set_title('Energy Efficiency (only F > 0.5)', fontsize=12)
    ax6.grid(True, alpha=0.3, which='both')
    ax6.legend(loc='upper right', fontsize=7)

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_dashboard.png', dpi=150, bbox_inches='tight')


# =============================================================================
# FIGURA: Confronto Hardware (4 modelli) con nuove labels
# =============================================================================
def plot_hardware_comparison(data_by_model, method_name='', save_prefix='benchmark'):
    """Confronto energia inference per hardware platform con fidelity drop."""

    N_values = sorted(set(data_by_model['CNN-Simple2D']['N']))
    N_selected = [n for n in N_values if n in [3, 5, 7, 8]]

    if len(N_selected) < 2:
        print("⚠️ Not enough N values for hardware comparison")
        return

    fig, axes = plt.subplots(1, len(N_selected), figsize=(4*len(N_selected), 5))
    if len(N_selected) == 1:
        axes = [axes]

    title = f'Inference Energy by Hardware Platform'
    if method_name:
        title += f' - {method_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Categorie con labels corte
    categories = ['CNN\n(GPU)', 'SCNN\n(Loihi)', 'SCNN\n(8-bit)', 'SCNN\n(4-bit)']
    cat_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    model_keys = ['CNN-Simple2D', 'SCNN-Norse-Simple2D',
                  'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']

    for ax, N in zip(axes, N_selected):
        energies = []
        fidelities = []

        for mk in model_keys:
            if mk in data_by_model and N in data_by_model[mk]['N']:
                idx = data_by_model[mk]['N'].index(N)
                energies.append(data_by_model[mk]['E_inf'][idx])
                fidelities.append(data_by_model[mk]['F'][idx])
            else:
                energies.append(0.001)
                fidelities.append(0)

        # Calcola fidelity drop rispetto a CNN (primo elemento)
        baseline_F = fidelities[0]
        fidelity_drops = []
        for f in fidelities:
            if baseline_F > 0:
                drop = (baseline_F - f) / baseline_F * 100
            else:
                drop = 0
            fidelity_drops.append(drop)

        bars = ax.bar(categories, energies, color=cat_colors)
        ax.set_ylabel('Energy (µJ)' if N == N_selected[0] else '')
        ax.set_title(f'N = {N} qubits\n(d = {2**N})', fontsize=11)
        ax.set_yscale('log')

        # Annota con fidelity drop (invece di fidelity)
        for bar, drop in zip(bars, fidelity_drops):
            height = bar.get_height()
            # Colore basato sul drop: verde se <1%, arancione se 1-5%, rosso se >5%
            if drop < 0.01:  # Praticamente 0
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
# NUOVO: Confronto M1 vs M2
# =============================================================================
def plot_method_comparison(results_M1, results_M2, save_prefix='benchmark'):
    """
    Confronta i risultati tra Method M1 e M2.
    Mostra ratio e differenze percentuali per fidelity, inference energy, training energy.
    """

    data_M1 = collect_data_by_model(results_M1)
    data_M2 = collect_data_by_model(results_M2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Method Comparison: M1 vs M2', fontsize=14, fontweight='bold')

    # Modelli da confrontare
    models_to_compare = ['CNN-Simple2D', 'SCNN-Norse-Simple2D',
                         'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']

    # Helper function per calcolare metriche comparative
    def get_comparison_data(data_M1, data_M2, model, metric):
        """Restituisce N comuni, valori M1, valori M2."""
        N_common = sorted(set(data_M1[model]['N']) & set(data_M2[model]['N']))
        vals_M1 = []
        vals_M2 = []
        for n in N_common:
            idx1 = data_M1[model]['N'].index(n)
            idx2 = data_M2[model]['N'].index(n)
            vals_M1.append(data_M1[model][metric][idx1])
            vals_M2.append(data_M2[model][metric][idx2])
        return N_common, vals_M1, vals_M2

    # =====================================================================
    # ROW 1: FIDELITY
    # =====================================================================

    # 1. Fidelity: Differenza percentuale (M1 - M2) / M2 * 100
    ax1 = axes[0, 0]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, model, 'F')
            # Differenza percentuale: positivo = M1 migliore
            diff_pct = [(f1 - f2) / f2 * 100 if f2 > 0 else 0
                       for f1, f2 in zip(F_M1, F_M2)]

            if diff_pct:
                ax1.plot(N_common, diff_pct, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)

    ax1.set_xlabel('N qubits', fontsize=11)
    ax1.set_ylabel('Fidelity Change (%)', fontsize=11)
    ax1.set_title('Fidelity: (M1 - M2) / M2 × 100%\n(positive = M1 better)', fontsize=11)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=7)

    # 2. Fidelity: Ratio M1/M2
    ax2 = axes[0, 1]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, model, 'F')
            ratio = [f1 / f2 if f2 > 0 else np.nan for f1, f2 in zip(F_M1, F_M2)]

            if ratio:
                ax2.plot(N_common, ratio, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)

    ax2.set_xlabel('N qubits', fontsize=11)
    ax2.set_ylabel('Fidelity Ratio (M1 / M2)', fontsize=11)
    ax2.set_title('Fidelity Ratio\n(>1 = M1 better)', fontsize=11)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=7)

    # 3. Infidelity Ratio (1-F_M1) / (1-F_M2) - più informativo per fidelity alte
    ax3 = axes[0, 2]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, F_M1, F_M2 = get_comparison_data(data_M1, data_M2, model, 'F')
            # Infidelity ratio: <1 = M1 ha meno errore
            infid_ratio = [(1 - f1) / (1 - f2) if (1 - f2) > 1e-9 else np.nan
                          for f1, f2 in zip(F_M1, F_M2)]

            if infid_ratio:
                ax3.semilogy(N_common, infid_ratio, marker=marker, color=color,
                            label=label, linewidth=2, markersize=8)

    ax3.set_xlabel('N qubits', fontsize=11)
    ax3.set_ylabel('Infidelity Ratio', fontsize=11)
    ax3.set_title('Infidelity Ratio: (1-F_M1) / (1-F_M2)\n(<1 = M1 has lower error)', fontsize=11)
    ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(loc='best', fontsize=7)

    # =====================================================================
    # ROW 2: ENERGY
    # =====================================================================

    # 4. Inference Energy: Differenza percentuale
    ax4 = axes[1, 0]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, model, 'E_inf')
            # Differenza percentuale: negativo = M1 più efficiente
            diff_pct = [(e1 - e2) / e2 * 100 if e2 > 0 else 0
                       for e1, e2 in zip(E_M1, E_M2)]

            if diff_pct:
                ax4.plot(N_common, diff_pct, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)

    ax4.set_xlabel('N qubits', fontsize=11)
    ax4.set_ylabel('Inference Energy Change (%)', fontsize=11)
    ax4.set_title('Inference Energy: (M1 - M2) / M2 × 100%\n(negative = M1 more efficient)', fontsize=11)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='best', fontsize=7)

    # 5. Inference Energy: Ratio M1/M2
    ax5 = axes[1, 1]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, model, 'E_inf')
            ratio = [e1 / e2 if e2 > 0 else np.nan for e1, e2 in zip(E_M1, E_M2)]

            if ratio:
                ax5.plot(N_common, ratio, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)

    ax5.set_xlabel('N qubits', fontsize=11)
    ax5.set_ylabel('Inference Energy Ratio (M1 / M2)', fontsize=11)
    ax5.set_title('Inference Energy Ratio\n(<1 = M1 more efficient)', fontsize=11)
    ax5.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='best', fontsize=7)

    # 6. Training Energy: Ratio M1/M2
    ax6 = axes[1, 2]
    for model in models_to_compare:
        if model in data_M1 and model in data_M2:
            color = COLORS.get(model, 'gray')
            marker = MARKERS.get(model, 'o')
            label = get_label(model)

            N_common, E_M1, E_M2 = get_comparison_data(data_M1, data_M2, model, 'E_train')
            ratio = [e1 / e2 if e2 > 0 else np.nan for e1, e2 in zip(E_M1, E_M2)]

            if ratio:
                ax6.plot(N_common, ratio, marker=marker, color=color,
                        label=label, linewidth=2, markersize=8)

    ax6.set_xlabel('N qubits', fontsize=11)
    ax6.set_ylabel('Training Energy Ratio (M1 / M2)', fontsize=11)
    ax6.set_title('Training Energy Ratio\n(<1 = M1 more efficient)', fontsize=11)
    ax6.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='best', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_method_comparison.png', dpi=150, bbox_inches='tight')


# =============================================================================
# NUOVO: Istogrammi confronto M1 vs M2 (Energia e Infidelity)
# =============================================================================
def plot_m1_vs_m2_bar_comparison(results_M1, results_M2, save_prefix='benchmark',
                                  N_selected=None):
    """
    Crea istogrammi grouped per confrontare M1 vs M2 per energia e infidelity.
    """

    data_M1 = collect_data_by_model(results_M1)
    data_M2 = collect_data_by_model(results_M2)

    if N_selected is None:
        N_selected = [5, 7, 8]

    models_to_compare = ['CNN-Simple2D', 'SCNN-Norse-Simple2D',
                         'SCNN-Crossbar-Simple2D-8b', 'SCNN-Crossbar-Simple2D-4b']

    model_labels = ['CNN\n(GPU)', 'SCNN\n(Loihi)', 'SCNN\n(8-bit)', 'SCNN\n(4-bit)']

    # ==========================================================================
    # FIGURA 1: Inference Energy M1 vs M2
    # ==========================================================================
    fig, axes = plt.subplots(1, len(N_selected), figsize=(5*len(N_selected), 5))
    if len(N_selected) == 1:
        axes = [axes]

    fig.suptitle('Inference Energy: M1 vs M2 Comparison', fontsize=14, fontweight='bold')

    bar_width = 0.35
    x = np.arange(len(models_to_compare))

    for ax, N in zip(axes, N_selected):
        energies_M1 = []
        energies_M2 = []

        for model in models_to_compare:
            # M1
            if model in data_M1 and N in data_M1[model]['N']:
                idx = data_M1[model]['N'].index(N)
                energies_M1.append(data_M1[model]['E_inf'][idx])
            else:
                energies_M1.append(np.nan)

            # M2
            if model in data_M2 and N in data_M2[model]['N']:
                idx = data_M2[model]['N'].index(N)
                energies_M2.append(data_M2[model]['E_inf'][idx])
            else:
                energies_M2.append(np.nan)

        bars1 = ax.bar(x - bar_width/2, energies_M1, bar_width, label='M1', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, energies_M2, bar_width, label='M2', color='#3498db', alpha=0.8)

        ax.set_ylabel('Inference Energy (µJ)' if N == N_selected[0] else '')
        ax.set_title(f'N = {N} qubits (d = {2**N})', fontsize=11)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
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

    fig.suptitle('Infidelity (1-F): M1 vs M2 Comparison', fontsize=14, fontweight='bold')

    for ax, N in zip(axes, N_selected):
        infidelities_M1 = []
        infidelities_M2 = []

        for model in models_to_compare:
            # M1
            if model in data_M1 and N in data_M1[model]['N']:
                idx = data_M1[model]['N'].index(N)
                f = data_M1[model]['F'][idx]
                infidelities_M1.append(max(1 - f, 1e-6))
            else:
                infidelities_M1.append(np.nan)

            # M2
            if model in data_M2 and N in data_M2[model]['N']:
                idx = data_M2[model]['N'].index(N)
                f = data_M2[model]['F'][idx]
                infidelities_M2.append(max(1 - f, 1e-6))
            else:
                infidelities_M2.append(np.nan)

        bars1 = ax.bar(x - bar_width/2, infidelities_M1, bar_width, label='M1', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + bar_width/2, infidelities_M2, bar_width, label='M2', color='#3498db', alpha=0.8)

        ax.set_ylabel('Infidelity (1 - F)' if N == N_selected[0] else '')
        ax.set_title(f'N = {N} qubits (d = {2**N})', fontsize=11)
        ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Linea threshold 1%
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, linewidth=1)

        # Annota con ratio M1/M2
        for i, (inf1, inf2) in enumerate(zip(infidelities_M1, infidelities_M2)):
            if not np.isnan(inf1) and not np.isnan(inf2) and inf2 > 1e-9:
                ratio = inf1 / inf2
                max_inf = max(inf1, inf2)
                # Mostra ratio solo se significativo
                if ratio > 1.1 or ratio < 0.9:
                    ax.annotate(f'{ratio:.2f}×',
                               xy=(x[i], max_inf),
                               xytext=(0, 5), textcoords="offset points",
                               ha='center', va='bottom', fontsize=8,
                               color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_m1_vs_m2_infidelity.png', dpi=150, bbox_inches='tight')


# =============================================================================
# NUOVO: Breakdown Consumo Energetico Crossbar
# =============================================================================
def plot_crossbar_energy_breakdown(crossbar_energy_data, method_name='', save_prefix='benchmark'):
    """
    Mostra il breakdown del consumo energetico del crossbar array.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    title = f'Crossbar Energy Breakdown'
    if method_name:
        title += f' - {method_name}'
    fig.suptitle(title, fontsize=13, fontweight='bold')

    component_colors = {
        'ADC': '#e41a1c',
        'DAC': '#377eb8',
        'crossbar_array': '#4daf4a',
        'peripheral': '#984ea3',
    }

    N_values = crossbar_energy_data.get('N', [3, 4, 5, 6, 7, 8])

    for ax, (bits, title) in zip(axes, [('8b', 'Crossbar 8-bit'), ('4b', 'Crossbar 4-bit')]):
        if bits in crossbar_energy_data:
            data = crossbar_energy_data[bits]

            # Stacked bar chart
            x = np.arange(len(N_values))
            width = 0.6

            bottom = np.zeros(len(N_values))
            for component in ['ADC', 'DAC', 'crossbar_array', 'peripheral']:
                if component in data:
                    values = data[component]
                    ax.bar(x, values, width, bottom=bottom,
                          label=component, color=component_colors.get(component, 'gray'))
                    bottom += np.array(values)

            ax.set_xlabel('N qubits', fontsize=11)
            ax.set_ylabel('Energy (µJ)', fontsize=11)
            ax.set_title(title, fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{n}' for n in N_values])
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, f'No data for {bits}',
                   transform=ax.transAxes, ha='center', va='center')

    plt.tight_layout()
    suffix = f'_{method_name}' if method_name else ''
    plt.savefig(f'{save_prefix}{suffix}_crossbar_breakdown.png', dpi=150, bbox_inches='tight')


# =============================================================================
# FUNZIONE PRINCIPALE: Genera tutti i plot per un singolo metodo
# =============================================================================
def plot_complete_benchmark(results, method_name='', save_prefix='benchmark',
                            crossbar_energy_data=None):
    """
    Genera tutti i plot per un singolo metodo (M1 o M2).
    """

    print(f"\n{'='*60}")
    print(f"📊 Generating plots{' for ' + method_name if method_name else ''}")
    print(f"{'='*60}")

    # Raccogli dati
    data_by_model = collect_data_by_model(results)

    # 1. Dashboard principale
    print("  → Dashboard...")
    plot_dashboard(data_by_model, method_name, save_prefix)

    # 2. Hardware comparison
    print("  → Hardware comparison...")
    plot_hardware_comparison(data_by_model, method_name, save_prefix)

    # 3. Crossbar breakdown (se disponibile)
    if crossbar_energy_data is not None:
        print("  → Crossbar energy breakdown...")
        plot_crossbar_energy_breakdown(crossbar_energy_data, method_name, save_prefix)

    print("\n" + "="*60)
    print("✅ All plots saved!")
    print("="*60)

    return data_by_model


# =============================================================================
# FUNZIONE: Genera tutti i plot + confronto M1 vs M2
# =============================================================================
def plot_complete_benchmark_with_comparison(results_M1, results_M2,
                                             save_prefix='benchmark',
                                             crossbar_energy_data_M1=None,
                                             crossbar_energy_data_M2=None,
                                             N_selected=None):
    """
    Genera tutti i plot per M1 e M2, più il confronto tra i due metodi.
    """

    # Plot per M1
    data_M1 = plot_complete_benchmark(results_M1, 'M1', save_prefix,
                                       crossbar_energy_data_M1)

    # Plot per M2
    data_M2 = plot_complete_benchmark(results_M2, 'M2', save_prefix,
                                       crossbar_energy_data_M2)

    # Confronto M1 vs M2 (grafici linee con ratio)
    print("\n" + "="*60)
    print("📊 Generating M1 vs M2 comparison plots")
    print("="*60)
    print("  → Method comparison (ratios)...")
    plot_method_comparison(results_M1, results_M2, save_prefix)

    # Confronto M1 vs M2 (istogrammi grouped)
    print("  → M1 vs M2 bar charts (energy & infidelity)...")
    plot_m1_vs_m2_bar_comparison(results_M1, results_M2, save_prefix, N_selected)

    return {'M1': data_M1, 'M2': data_M2}


# === cell #23 ===
# =============================================================================
# OPZIONE 1: Plot separati per M1 e M2 (come prima)
# =============================================================================

# Visualization M1
data_M1 = plot_complete_benchmark(results_complete_M1, method_name='M1', save_prefix='qst_benchmark')

# Visualization M2
data_M2 = plot_complete_benchmark(results_complete_M2, method_name='M2', save_prefix='qst_benchmark')

# Confronto M1 vs M2 (line plots con ratio)
plot_method_comparison(results_complete_M1, results_complete_M2, save_prefix='qst_benchmark')

# Confronto M1 vs M2 (istogrammi grouped per energia e infidelity)
plot_m1_vs_m2_bar_comparison(results_complete_M1, results_complete_M2,
                              save_prefix='qst_benchmark', N_selected=[5, 7, 8])

