import sys
#!/usr/bin/env python3
"""AUTO-GENERATED from M_sweep_GHZ_Mixed.ipynb.

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
# 1. SETUP & IMPORTS - GPU OPTIMIZED
# =============================================================================
import os, sys, math, random, time, numpy as np, gc

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
    print(f"✓ CUDA Alloc Conf: expandable_segments=True")
print(f"✓ Checkpoint Dir: {CHECKPOINT_DIR}")


# === cell #1 ===
# Norse imports (dopo torch imports)
import norse.torch as norse
from norse.torch import LIFParameters


# === cell #2 ===
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


# === cell #3 ===
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


# === cell #4 ===
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


# === cell #5 ===
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
    """Generalized Werner state (Eq. A4 from Hua et al., arXiv:2507.23007).
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



# =============================================================================
# 13. TRAINING FUNCTION - GPU OPTIMIZED
# =============================================================================

def to_ri(x_np):
    return torch.from_numpy(np.stack([x_np.real, x_np.imag], axis=0).astype(np.float32))


# === cell #6 ===
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


# === cell #7 ===
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


# === cell #8 ===
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


# === cell #9 ===
# =============================================================================
# CROSSBAR CONVOLUTIONAL LAYERS - FIXED
# =============================================================================

class CrossbarLIFConv2d(NorseLIFBase):
    """
    Norse LIF with memristor crossbar for convolutional layers.

    Conv2d weights are quantized to simulate crossbar implementation.
    In hardware, conv is implemented as im2col + crossbar MVM.

    Args:
        c_in: Input channels
        c_out: Output channels
        k: Kernel size (default: 3)
        s: Stride (default: 1)
        p: Padding (default: 1)
        bias: Include bias (default: False)
        weight_bits: Weight quantization (default: 8)
        adc_bits: ADC resolution (default: 8)
        noise_std: Read noise (default: 0.01)
        **kwargs: Norse LIF parameters (T, tau_mem_inv, tau_syn_inv, v_th, etc.)
    """

    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False,
                 weight_bits=8, adc_bits=8, noise_std=0.01, **kwargs):

        # Extract crossbar-specific parameters (don't pass to NorseLIFBase)
        crossbar_params = {
            'weight_bits': weight_bits,
            'adc_bits': adc_bits,
            'noise_std': noise_std
        }

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
        self.weight_bits = crossbar_params['weight_bits']
        self.adc_bits = crossbar_params['adc_bits']
        self.noise_std = crossbar_params['noise_std']

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

    def current(self, x):
        # Quantize weights
        w_quant = self.quantize_conv_weights(self.cv.weight)

        # Conv operation
        y = F.conv2d(x, w_quant, self.cv.bias,
                    stride=self.cv.stride, padding=self.cv.padding)

        # Add read noise (during training)
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise

        # ADC quantization
        y = self.adc_quantize(y)

        return y


class CrossbarLIFConvT2d(NorseLIFBase):
    """
    Norse LIF with memristor crossbar for transposed convolution.
    Similar to CrossbarLIFConv2d but for upsampling.
    """

    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False,
                 weight_bits=8, adc_bits=8, noise_std=0.01, **kwargs):

        # Extract crossbar-specific parameters
        crossbar_params = {
            'weight_bits': weight_bits,
            'adc_bits': adc_bits,
            'noise_std': noise_std
        }

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
        self.weight_bits = crossbar_params['weight_bits']
        self.adc_bits = crossbar_params['adc_bits']
        self.noise_std = crossbar_params['noise_std']

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

    def current(self, x):
        # Quantize weights
        w_quant = self.quantize_conv_weights(self.cvt.weight)

        # Transposed conv operation
        y = F.conv_transpose2d(x, w_quant, self.cvt.bias,
                              stride=self.cvt.stride,
                              padding=self.cvt.padding)

        # Add read noise
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise

        # ADC quantization
        y = self.adc_quantize(y)

        return y


# === cell #10 ===
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


# === cell #11 ===
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


# === cell #12 ===
# =============================================================================
# TRAINING LOOP CON CROSSBAR - Energy-Aware
# =============================================================================

def optimize_single_state_crossbar(model_class, N=3, method='M1', M=12, steps=200, lr=1e-3,
                                   noise_std=0.0, normalize_cond=True, warm_spiking=False,
                                   weight_bits=8, adc_bits=8, dac_bits=8,
                                   device_variation=0.02, read_noise=0.01,
                                   use_amp=True, log_energy=True):
    """
    Training con crossbar memristor simulation.

    NOTA: model_class deve essere un callable che accetta (cond_dim, d) come argomenti.
          I parametri crossbar (weight_bits, etc) devono essere già inclusi nel lambda
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

    # Create crossbar model - model_class già contiene i parametri crossbar
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

    print(f"    ✓ Final: F={F_final:.4f}, E={energy_final['E_total_mJ']:.4f}mJ, Time={elapsed:.1f}s")

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
                          M_M1=12, M_M2=2, steps=400, lr=1e-3,
                          weight_bits=8, adc_bits=8, dac_bits=8,
                          device_variation=0.02, read_noise=0.01,
                          use_amp=True):
    """
    Benchmark completo con crossbar models.

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
        M = M_M1 if method == 'M1' else M_M2

        print(f"\n{'='*70}")
        print(f"🚀 CROSSBAR BENCHMARK: {method} (M={M})")
        print(f"   Crossbar config: {weight_bits}-bit weights, {adc_bits}-bit ADC/DAC")
        print(f"={'='*70}")

        for N in N_list:
            d = 2**N
            print(f"\n--- N={N} qubit (d={d}) ---")

            # Adapt parameters for larger N
            current_steps = steps if N < 8 else min(steps, 300)

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
                print(f"\n  🔧 {model_name}")

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

def compare_crossbar_vs_standard(N=3, method='M1', M=12, steps=200,
                                weight_bits_list=[32, 8, 4, 2]):
    """
    Confronta diverse quantization levels per vedere impatto su accuracy.

    Args:
        weight_bits_list: Lista di bit-widths da testare (32 = FP32 baseline)

    Returns:
        DataFrame con risultati comparativi
    """
    print(f"\n{'='*70}")
    print(f"📊 QUANTIZATION IMPACT ANALYSIS: N={N}, {method}")
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

        print(f"  ✓ {results[-1]['Bits']:5s}: F={results[-1]['Fidelity']:.4f}")
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


# === cell #13 ===
# =============================================================================
# 13. TRAINING FUNCTION - GPU OPTIMIZED
# =============================================================================

def to_ri(x_np):
    return torch.from_numpy(np.stack([x_np.real, x_np.imag], axis=0).astype(np.float32))

def optimize_single_state(model_class, N=3, method='M1', M=12, steps=200, lr=1e-3,
                          noise_std=0.0, normalize_cond=True, warm_spiking=False,
                          use_amp=True, use_compile=False):
    """
    Ottimizzazione con Mixed Precision Training e opzionalmente torch.compile

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

    # Opzionale: compila il modello per velocità (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            G = torch.compile(G, mode="reduce-overhead")
            print("      ✓ Model compiled")
        except Exception as e:
            print(f"      ⚠ Compile failed: {e}")

    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
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


# === cell #14 ===
# =====================================================================
# M-SWEEP CONFIGURATION
# =====================================================================

# Fixed parameters (optimal from T-sweep)
T_FIXED = 8
SPARSITY = 0.1

# ---- Choose which N to run ----
# Set to 5 for quick local run, 8 for the full study
N_FIXED = 8   # <-- CHANGE THIS: 5 or 8

# Bases to sweep (paper-guided ranges for noisy/mixed states)
if N_FIXED == 5:
    M1_LIST = [4, 8, 11, 16, 32, 64, 128, 256, 512]
    M2_LIST = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
elif N_FIXED == 8:
    # Paper Table 2: noisy pure needs M1=720, M2=40 at N=8 for F≈0.99
    M1_LIST = [8, 16, 32, 64, 128, 256, 512, 720, 1024]
    M2_LIST = [2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
else:
    # Generic fallback
    M1_LIST = [4, 8, 16, 32, 64, 128, 256, 512]
    M2_LIST = [2, 4, 8, 12, 16, 32, 48]

METHODS = ['M1', 'M2']

# Training
STEPS = 500
LR = 1e-3

# Fixed SNN hyperparams (same as T-sweep)
V_TH = 0.9
ENC_GAMMA = 1.5
TAU_MEM_INV = 100.0
TAU_SYN_INV = 200.0

# Crossbar params
CROSSBAR_NOISE = 0.01
CROSSBAR_VARIATION = 0.02

MODEL_CONFIGS = [
    {'name': 'Norse-Simple2D', 'is_crossbar': False, 'bits': None},
    {'name': 'Crossbar-8b',    'is_crossbar': True,  'bits': 8},
    {'name': 'Crossbar-4b',    'is_crossbar': True,  'bits': 4},
]

# Summary
total_m1 = len(M1_LIST) * len(MODEL_CONFIGS)
total_m2 = len(M2_LIST) * len(MODEL_CONFIGS)
total = total_m1 + total_m2

d = 2**N_FIXED
print(f"M-Sweep Configuration (Mixed Werner State, p=0.5):")
print(f"  N={N_FIXED}, d={d}, T={T_FIXED}")
print(f"  M1 bases: {M1_LIST}")
print(f"  M2 bases: {M2_LIST}")
print(f"  Models: {[c['name'] for c in MODEL_CONFIGS]}")
print(f"  Total runs: {total} ({total_m1} M1 + {total_m2} M2)")
print(f"")
print(f"  M2 cond_dim range: {M2_LIST[0]*d} - {M2_LIST[-1]*d}")
print(f"  M1 cond_dim range: {M1_LIST[0]} - {M1_LIST[-1]}")
print(f"  M2 proj params range: {M2_LIST[0]*d * 32*N_FIXED*2*N_FIXED*2 /1e6:.1f}M - {M2_LIST[-1]*d * 32*N_FIXED*2*N_FIXED*2 /1e6:.1f}M")

# Paper reference
print(f"")
print(f"  Paper reference (noisy pure, F≈0.99):")
print(f"    M1(N=8)=720,  M2(N=8)=40")
print(f"    M1(N=9)=1200, M2(N=9)=—")


# === cell #15 ===
# =====================================================================
# M-SWEEP CORE
# =====================================================================

# (multiseed: removed redundant CHECKPOINT_DIR redefinition)
# CHECKPOINT_DIR = Path("checkpoints_M_sweep")
# CHECKPOINT_DIR.mkdir(exist_ok=True)
# 
N = N_FIXED
CSV_PREFIX = f'M_sweep_mixed_N{N}'
d = 2**N
T = T_FIXED


def make_model_fn_fixed_T(config, N, T):
    """Build a model factory with fixed T (from T-sweep optimal)."""
    name = config['name']
    bits = config['bits']

    if name == 'Norse-Simple2D':
        def fn(cond_dim, d):
            return SCNNGen_LIF_Simple2D(
                cond_dim=cond_dim, d=d,
                proj_hw=(N*2, N*2), ch=(32, 64),
                T=T, v_th=V_TH, enc_gamma=ENC_GAMMA,
                tau_mem_inv=TAU_MEM_INV, tau_syn_inv=TAU_SYN_INV)
        return fn

    if bits <= 4:
        vth, gamma = 0.3, ENC_GAMMA * 2
    else:
        vth, gamma = 0.5, ENC_GAMMA * 1.5

    def fn(cond_dim, d):
        return SCNNGen_Crossbar_Simple2D(
            cond_dim=cond_dim, d=d,
            proj_hw=(N*2, N*2), ch=(32, 64),
            T=T, v_th=vth, enc_gamma=gamma,
            weight_bits=bits, adc_bits=bits, dac_bits=bits,
            noise_std=CROSSBAR_NOISE, device_variation=CROSSBAR_VARIATION)
    return fn


def estimate_inference_energy_config(model, config, T):
    """Dispatch to the appropriate energy estimator."""
    if config['name'] == 'Norse-Simple2D':
        return estimate_loihi_inference_energy(model, T=T, sparsity=SPARSITY)
    return estimate_crossbar_inference_energy(
        model, T=T, sparsity=SPARSITY, bits=config['bits'])


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_M_sweep(N=N_FIXED, T=T_FIXED, methods=METHODS,
                model_configs=MODEL_CONFIGS, use_amp=True,
                checkpoint_every=10):
    """
    Sweep number of measurement bases M for all (method, model) combinations.

    Returns pd.DataFrame with columns:
        Model, N, method, M, cond_dim, F_best, F_last,
        E_inference_uJ, n_params, time_sec
    """
    # === MULTISEED PATCHED ===
    if _ARGS.quick:
        methods = ['M1']
        model_configs = model_configs[:1]

    if CSV_PATH.exists():
        _df_existing = pd.read_csv(CSV_PATH)
        rows = _df_existing.to_dict('records')
        _done_keys = {(int(r['seed']), r['Model'], int(r['N']), r['method'],
                       int(r['M'])) for r in rows if r.get('seed') == SEED}
        print(f"  [resume] {len(_done_keys)} rows for seed={SEED}", flush=True)
    else:
        rows = []
        _done_keys = set()
    rows = rows  # (was [])
    d_N = 2**N

    # Count total runs
    total = 0
    for method in methods:
        M_LIST = M1_LIST if method == 'M1' else M2_LIST
        if _ARGS.quick: M_LIST = M_LIST[:2]
        total += len(M_LIST) * len(model_configs)

    run_idx = 0

    for method in methods:
        M_LIST = M1_LIST if method == 'M1' else M2_LIST
        if _ARGS.quick: M_LIST = M_LIST[:2]

        for M_val in M_LIST:
            # For M1, check that M doesn't exceed available Pauli strings
            if method == 'M1':
                max_pauli = 3**N  # non-identity Pauli strings
                M_actual = min(M_val, max_pauli)
            else:
                M_actual = M_val

            cond_dim = M_actual if method == 'M1' else M_actual * d_N

            for config in model_configs:
                run_idx += 1
                name = config['name']
                if (SEED, name, N, method, int(M_actual)) in _done_keys:
                    print(f"  [skip] seed={SEED} {name} {method} M={M_actual}", flush=True)
                    continue

                print(f"\n[{run_idx}/{total}] {method} | M={M_actual} | cond_dim={cond_dim} | {name}")
                sys.stdout.flush()

                gpu_cleanup()

                if torch.cuda.is_available():
                    free_gb = torch.cuda.mem_get_info()[0] / 1e9
                    total_gb = torch.cuda.mem_get_info()[1] / 1e9
                    print(f"  GPU memory: {free_gb:.1f}/{total_gb:.1f} GB free")

                model_fn = make_model_fn_fixed_T(config, N, T)

                # Reset seed for reproducibility

                t0 = time.time()
                model = None
                res = None
                try:
                    if config['is_crossbar']:
                        res = optimize_single_state_crossbar(
                            model_class=model_fn,
                            N=N, method=method, M=M_actual,
                            steps=STEPS, lr=LR,
                            warm_spiking=True,
                            weight_bits=config['bits'],
                            adc_bits=config['bits'],
                            dac_bits=config['bits'],
                            device_variation=CROSSBAR_VARIATION,
                            read_noise=CROSSBAR_NOISE,
                            use_amp=use_amp, log_energy=False)
                        F_best = res['F_best']
                        F_hist = res['F_hist']
                        model = res['model']
                    else:
                        res = optimize_single_state(
                            model_class=model_fn,
                            N=N, method=method, M=M_actual,
                            steps=STEPS, lr=LR,
                            warm_spiking=True,
                            use_amp=use_amp)
                        f_hist = res['f_hist']
                        F_best = float(np.max(f_hist))
                        F_hist = f_hist.tolist() if isinstance(f_hist, np.ndarray) else f_hist
                        model = res['model']

                    elapsed = time.time() - t0
                    E_inf = estimate_inference_energy_config(model, config, T)
                    n_params = sum(p.numel() for p in model.parameters())

                    row = {
                        'Model': name, 'N': N, 'method': method,
                        'M': M_actual, 'cond_dim': cond_dim,
                        'F_best': F_best,
                        'F_last': float(F_hist[-1]) if F_hist else 0.0,
                        'E_inference_uJ': E_inf['E_total_uJ'],
                        'n_params': n_params,
                        'time_sec': elapsed,
                    }
                    rows.append(row)
                    print(f"  F_best={F_best:.4f}, cond_dim={cond_dim}, "
                          f"params={n_params}, time={elapsed:.1f}s")

                except Exception as e:
                    elapsed = time.time() - t0
                    print(f"  FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    rows.append({
                        'seed': SEED,
                        'Model': name, 'N': N, 'method': method,
                        'M': M_actual, 'cond_dim': cond_dim,
                        'F_best': 0.0, 'F_last': 0.0,
                        'E_inference_uJ': float('nan'),
                        'n_params': 0, 'time_sec': elapsed,
                    })

                # Cleanup
                del model, res
                gpu_cleanup()

                pd.DataFrame(rows).to_csv(CSV_PATH, index=False)  # multiseed: save every iter

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSweep complete! {len(df)} rows saved -> {CSV_PATH}")
    return df


# === cell #16 ===
df_sweep = run_M_sweep()

# === MULTISEED quick-exit ===
# Skip the post-sweep analysis / projection / "best M" cells in --quick mode:
# they assume a full sweep grid and crash on the smoke-test subset.
if _ARGS.quick:
    print("[multiseed] --quick: skipping post-sweep analysis cells", flush=True)
    sys.exit(0)


# === cell #17 ===
# Uncomment to reload from checkpoint if kernel dies:
# df_sweep = pd.read_csv(CHECKPOINT_DIR / f'M_sweep_mixed_N{N_FIXED}_final.csv')
# display(df_sweep)


# === cell #18 ===
# =====================================================================
# PLOT 1: Fidelity vs M (main result)
# =====================================================================

COLORS = {
    'Norse-Simple2D': '#2ca02c',
    'Crossbar-8b':    '#ff7f0e',
    'Crossbar-4b':    '#d62728',
}
MARKERS = {
    'Norse-Simple2D': 'o',
    'Crossbar-8b':    's',
    'Crossbar-4b':    'D',
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax_idx, method in enumerate(METHODS):
    ax = axes[ax_idx]
    df_m = df_sweep[df_sweep['method'] == method]

    for model_name in df_m['Model'].unique():
        df_model = df_m[df_m['Model'] == model_name].sort_values('M')
        ax.plot(df_model['M'], df_model['F_best'],
                marker=MARKERS.get(model_name, 'o'),
                color=COLORS.get(model_name, 'gray'),
                label=model_name, linewidth=2, markersize=8)

    ax.set_xlabel('Number of measurement bases |M|', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title(f'{method} — Mixed Werner State (N={N_FIXED}, T={T_FIXED})', fontsize=13)
    ax.set_xscale('log', base=2)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='F=0.8 threshold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('M_sweep_mixed_fidelity_vs_M.png', dpi=150, bbox_inches='tight')
plt.savefig('M_sweep_mixed_fidelity_vs_M.pdf', bbox_inches='tight')
print("Saved: M_sweep_mixed_fidelity_vs_M.png/pdf")


# === cell #19 ===
# =====================================================================
# SUMMARY: Minimum M for F > 0.8 threshold
# =====================================================================

print("=" * 80)
print("MINIMUM BASES FOR F > 0.8 (Mixed Werner State, N=%d, T=%d)" % (N_FIXED, T_FIXED))
print("=" * 80)

F_THRESHOLD = 0.8

for method in METHODS:
    print(f"\n--- Method: {method} ---")
    df_m = df_sweep[df_sweep['method'] == method]

    for model_name in sorted(df_m['Model'].unique()):
        df_model = df_m[df_m['Model'] == model_name].sort_values('M')
        above = df_model[df_model['F_best'] >= F_THRESHOLD]
        if len(above) > 0:
            min_M = above.iloc[0]['M']
            min_cond = above.iloc[0]['cond_dim']
            best_F = above.iloc[0]['F_best']
            print(f"  {model_name:<18s}: M >= {int(min_M):>4d} (cond_dim={int(min_cond):>6d}, F={best_F:.4f})")
        else:
            best_row = df_model.loc[df_model['F_best'].idxmax()]
            print(f"  {model_name:<18s}: NEVER reached F>0.8 (best: M={int(best_row['M'])}, F={best_row['F_best']:.4f})")

# Scaling predictions
print("\n" + "=" * 80)
print("SCALING PREDICTIONS FOR OTHER N")
print("=" * 80)
print(f"\nFor pure GHZ states (Hua et al., Table 2):")
print(f"  M1: N=8 -> 11 bases,  N=9 -> 12 bases  (very slowly growing)")
print(f"  M2: N=8 -> 2 bases,   N=9 -> 2 bases   (constant!)")
print(f"\nFor mixed Werner states (p=0.5), the identity component I/d adds")
print(f"more structure, likely requiring O(sqrt(d)) or O(log(d)) extra bases.")
print(f"\nKey insight: if M_opt is the minimum bases found here for N={N_FIXED},")
print(f"then for N=8 (d=256), a conservative estimate is:")
print(f"  M1(N=8) ~ M_opt * (256/32) = M_opt * 8")
print(f"  M2(N=8) ~ M_opt * sqrt(256/32) = M_opt * 2.83")
print(f"  But empirically it could be even less since the GHZ structure is sparse.")

# Compare with current T-sweep M values
print(f"\n" + "=" * 80)
print("COMPARISON WITH T-SWEEP (CURRENT VALUES vs RECOMMENDED)")
print("=" * 80)
for N_comp in [3, 5, 8]:
    d_comp = 2**N_comp
    old_m1 = min(4**N_comp - 1, max(128, 2 * d_comp))
    old_m2 = max(12, 5 * d_comp // 4)
    old_cond_m2 = old_m2 * d_comp
    print(f"\n  N={N_comp} (d={d_comp}):")
    print(f"    T-sweep M1: {old_m1:>6d} bases, cond_dim = {old_m1}")
    print(f"    T-sweep M2: {old_m2:>6d} bases, cond_dim = {old_cond_m2}")


# === cell #20 ===
# =====================================================================
# PROJECTION: What N=8 would look like with optimized M
# =====================================================================

print("=" * 80)
print("PROJECTED MODEL SIZE FOR N=8 WITH OPTIMIZED M")
print("=" * 80)

N8 = 8
d8 = 2**N8
H0_8, W0_8 = N8*2, N8*2
C0, C1 = 32, 64
proj_out_8 = C0 * H0_8 * W0_8  # 32 * 16 * 16 = 8192

# Get recommended M from this sweep
for method in METHODS:
    df_m = df_sweep[df_sweep['method'] == method]
    # Use Norse-Simple2D as reference
    df_norse = df_m[df_m['Model'] == 'Norse-Simple2D'].sort_values('M')
    above = df_norse[df_norse['F_best'] >= 0.8]

    if len(above) > 0:
        M_opt_n5 = int(above.iloc[0]['M'])
    else:
        M_opt_n5 = int(df_norse.loc[df_norse['F_best'].idxmax()]['M'])

    # Scale to N=8
    if method == 'M1':
        # M1 cond_dim = M (scalar per operator)
        # Scale roughly linearly or use paper's observation
        M_n8 = max(M_opt_n5, M_opt_n5 * 2)  # Conservative 2x
        cond_8 = M_n8
        proj_params_8 = cond_8 * proj_out_8
    else:
        # M2 cond_dim = M * d
        M_n8 = max(M_opt_n5, int(M_opt_n5 * 1.5))  # Conservative 1.5x
        cond_8 = M_n8 * d8
        proj_params_8 = cond_8 * proj_out_8

    print(f"\n{method}:")
    print(f"  Optimal M (N=5): {M_opt_n5}")
    print(f"  Projected M (N=8): {M_n8}")
    print(f"  cond_dim (N=8): {cond_8}")
    print(f"  Projection layer: Linear({cond_8}, {proj_out_8}) = {proj_params_8:,} params")
    print(f"  Memory (weights only): {proj_params_8 * 4 / 1e6:.1f} MB")

    # Compare to OLD M2 N=8
    if method == 'M2':
        old_cond = 320 * 256
        old_params = old_cond * proj_out_8
        print(f"\n  vs OLD T-sweep:")
        print(f"    Old M=320, cond_dim={old_cond}, proj={old_params:,} params ({old_params*4/1e9:.2f} GB)")
        print(f"    NEW is {old_params / proj_params_8:.0f}x smaller!")

