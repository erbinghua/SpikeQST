#!/usr/bin/env python3
"""T-Sweep GHZ Mixed States — multi-seed variant.

Adapted from mixed states/cluster/T_sweep_GHZ_Mixed.py.
"""

import os, math, random, time, numpy as np, gc
import argparse
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import matplotlib
matplotlib.use("Agg")

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR.parent.parent))  # multiseed/
from seed_utils import set_global_seed


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default=None)
    return p.parse_args()


_ARGS = _parse_args()
SEED = _ARGS.seed
OUT_DIR = Path(_ARGS.out_dir); OUT_DIR.mkdir(parents=True, exist_ok=True)
set_global_seed(SEED)
print(f"[multiseed] seed={SEED}  out_dir={OUT_DIR}", flush=True)

import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from collections import defaultdict
import pickle

import norse.torch as norse
from norse.torch import LIFParameters

DEVICE = torch.device(_ARGS.device) if _ARGS.device else \
         torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "results.csv"

print("=" * 70)
print("QUANTUM STATE TOMOGRAPHY: GPU-OPTIMIZED T-SWEEP (CLUSTER)")
print("=" * 70)
print(f"PyTorch {torch.__version__}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"Mixed Precision: Enabled")
else:
    print("WARNING: No GPU detected! Running on CPU (will be very slow)")
print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
sys.stdout.flush()


# =============================================================================
# ENERGY PARAMETERS
# =============================================================================

GPU_PARAMS = {
    'name': 'NVIDIA A100 (7nm)',
    'E_MAC': 35e-12,
    'E_DRAM': 640e-12,
    'E_L2_cache': 5e-12,
    'cache_hit_rate': 0.85,
    'TDP_W': 400,
    'util_small_model': 0.05,
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


# =============================================================================
# ENERGY ESTIMATION FUNCTIONS
# =============================================================================

def estimate_gpu_inference_energy(model, batch_size=1):
    n_params = sum(p.numel() for p in model.parameters())
    n_macs = n_params * 2 * batch_size
    E_compute = n_macs * GPU_PARAMS['E_MAC']
    n_mem_access = n_params * batch_size
    miss_rate = 1 - GPU_PARAMS['cache_hit_rate']
    E_memory = (n_mem_access * miss_rate * GPU_PARAMS['E_DRAM'] +
                n_mem_access * GPU_PARAMS['cache_hit_rate'] * GPU_PARAMS['E_L2_cache'])
    E_total = E_compute + E_memory
    return {
        'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
        'E_compute_J': E_compute, 'E_memory_J': E_memory,
        'n_macs': n_macs, 'n_params': n_params,
        'breakdown': {'compute_%': E_compute/E_total*100, 'memory_%': E_memory/E_total*100}
    }


def estimate_loihi_inference_energy(model, T=8, sparsity=0.1, batch_size=1):
    n_params = sum(p.numel() for p in model.parameters())
    n_neurons = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n_neurons += m.out_features
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            n_neurons += m.out_channels * 16

    n_syn_ops = n_params * T * sparsity * batch_size
    n_spikes = n_neurons * T * sparsity * batch_size
    n_leak_ops = n_neurons * T * batch_size

    E_syn = n_syn_ops * LOIHI_PARAMS['E_synapse']
    E_spikes = n_spikes * LOIHI_PARAMS['E_spike']
    E_leak = n_leak_ops * LOIHI_PARAMS['E_neuron_leak']
    E_routing = n_spikes * LOIHI_PARAMS['E_router']
    E_total = E_syn + E_spikes + E_leak + E_routing

    return {
        'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
        'E_syn_J': E_syn, 'E_spikes_J': E_spikes,
        'E_leak_J': E_leak, 'E_routing_J': E_routing,
        'n_params': n_params, 'n_neurons': n_neurons,
        'n_spikes': n_spikes, 'T': T, 'sparsity': sparsity,
    }


def estimate_crossbar_inference_energy(model, T=8, sparsity=0.1, bits=8, batch_size=1):
    E_ADC = MEMRISTOR_PARAMS['E_ADC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_ADC_4bit']
    E_DAC = MEMRISTOR_PARAMS['E_DAC_8bit'] if bits == 8 else MEMRISTOR_PARAMS['E_DAC_4bit']
    n_params = sum(p.numel() for p in model.parameters())

    total_in = 0
    total_out = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            total_in += m.in_features
            total_out += m.out_features
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fm = 16
            total_in += m.in_channels * fm
            total_out += m.out_channels * fm

    n_dac = total_in * T * sparsity * batch_size
    n_adc = total_out * T * batch_size
    n_mvm = n_params * T * sparsity * batch_size
    n_spikes = total_out * T * sparsity * batch_size

    E_dac = n_dac * E_DAC
    E_adc = n_adc * E_ADC
    E_mvm = n_mvm * MEMRISTOR_PARAMS['E_MAC_analog']
    E_spike = n_spikes * MEMRISTOR_PARAMS['E_spike_gen']
    E_leak = total_out * T * batch_size * MEMRISTOR_PARAMS['E_leakage']
    E_total = E_dac + E_adc + E_mvm + E_spike + E_leak

    return {
        'E_total_J': E_total, 'E_total_uJ': E_total * 1e6,
        'E_dac_J': E_dac, 'E_adc_J': E_adc, 'E_mvm_J': E_mvm,
        'E_spike_J': E_spike, 'E_leak_J': E_leak,
        'n_params': n_params, 'bits': bits, 'T': T, 'sparsity': sparsity,
    }


def estimate_cnn_energy(model, input_shape, steps=1):
    try:
        from torchinfo import summary
        with torch.no_grad():
            stats = summary(model, input_size=(1, *input_shape), verbose=0)
        n_macs = stats.total_mult_adds
    except Exception:
        n_params = sum(p.numel() for p in model.parameters())
        n_macs = n_params * 2

    n_params = sum(p.numel() for p in model.parameters())
    E_compute = n_macs * GPU_PARAMS['E_MAC']
    n_memory_accesses = n_params * 2
    cache_hit_rate = GPU_PARAMS['cache_hit_rate']
    E_memory = (n_memory_accesses * (1 - cache_hit_rate) * GPU_PARAMS['E_DRAM'] +
                n_memory_accesses * cache_hit_rate * GPU_PARAMS['E_L2_cache'])
    E_total = (E_compute + E_memory) * steps
    return E_total, {'n_macs': n_macs, 'n_params': n_params, 'E_compute': E_compute, 'E_memory': E_memory}


def estimate_memristor_snn_energy(model, T_spike=8, sparsity=0.1, bits=8, steps=1):
    params = MEMRISTOR_PARAMS
    E_ADC = params['E_ADC_8bit'] if bits == 8 else params['E_ADC_4bit']
    E_DAC = params['E_DAC_8bit'] if bits == 8 else params['E_DAC_4bit']
    n_synapses = sum(p.numel() for p in model.parameters())

    layer_info = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layer_info.append({'type': 'Linear', 'in_features': m.in_features,
                             'out_features': m.out_features, 'n_synapses': m.in_features * m.out_features})
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fm_size = 16
            layer_info.append({'type': 'Conv', 'in_channels': m.in_channels, 'out_channels': m.out_channels,
                             'n_synapses': m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1],
                             'n_neurons': m.out_channels * fm_size})

    n_neurons = sum(l.get('out_features', l.get('n_neurons', 0)) for l in layer_info)
    total_in = sum(l.get('in_features', l.get('in_channels', 0) * 16) for l in layer_info)
    total_out = sum(l.get('out_features', l.get('n_neurons', 0)) for l in layer_info)

    n_spikes = n_neurons * T_spike * sparsity
    E_spikes = n_spikes * params['E_spike_gen']
    n_mvm_ops = n_synapses * T_spike * sparsity
    E_analog_compute = n_mvm_ops * params['E_MAC_analog']
    n_adc_conversions = total_out * T_spike
    E_adc = n_adc_conversions * E_ADC
    n_dac_conversions = total_in * T_spike * sparsity
    E_dac = n_dac_conversions * E_DAC
    E_leak = n_neurons * T_spike * params['E_leakage']

    E_total_per_inference = E_analog_compute + E_adc + E_dac + E_spikes + E_leak
    E_total = E_total_per_inference * steps
    return E_total, {'n_synapses': n_synapses, 'n_neurons': n_neurons, 'T_spike': T_spike, 'sparsity': sparsity, 'bits': bits}


# =============================================================================
# UTILITIES
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
    if len(chosen) < M:
        for s in strings_all:
            if s not in seen:
                chosen.append(pauli_op(s)); seen.add(s)
                if len(chosen) >= M: break
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


class DensityMap(nn.Module):
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


def fidelity_batch(rho_pred_ri, rho_true_ri, eps=1e-12):
    with torch.amp.autocast('cuda', enabled=False):
        c_dtype = torch.complex128
        rho_p = torch.complex(rho_pred_ri[:,0].double(), rho_pred_ri[:,1].double()).to(c_dtype)
        rho_t = torch.complex(rho_true_ri[:,0].double(), rho_true_ri[:,1].double()).to(c_dtype)
        d = rho_t.shape[-1]
        batch_size = rho_pred_ri.shape[0]

        purity = torch.real(torch.diagonal(rho_t @ rho_t, dim1=-2, dim2=-1).sum(-1))
        is_pure = (purity > 1.0 - 1e-6)

        if is_pure.all():
            fid = torch.real(torch.diagonal(rho_t @ rho_p, dim1=-2, dim2=-1).sum(-1))
            fid = torch.clamp(fid, min=0.0, max=1.0)
            return fid, fid.mean().item()

        def sqrtm_psd(A):
            A = 0.5*(A + A.conj().transpose(-1,-2))
            A = A + eps * torch.eye(d, device=A.device, dtype=A.dtype)
            try:
                ev, V = torch.linalg.eigh(A)
            except torch.linalg.LinAlgError:
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
            fid = torch.clamp(fid, min=0.0, max=1.0)
        except (torch.linalg.LinAlgError, RuntimeError):
            fid = torch.zeros(batch_size, device=rho_pred_ri.device)

    return fid, fid.mean().item()

def to_ri(x_np):
    return torch.from_numpy(np.stack([x_np.real, x_np.imag], axis=0).astype(np.float32))


# =============================================================================
# NORSE LIF NEURONS
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


class NorseLIFBase(nn.Module):
    def __init__(self, output_features, T=8,
                 tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, return_rate=True, enc_mode='poisson',
                 enc_gamma=1.5, enc_pmin=0.02, enc_pmax=0.98):
        super().__init__()
        self.T = T
        self.return_rate = return_rate
        self.enc_mode = enc_mode
        self.enc_gamma = enc_gamma
        self.enc_pmin = enc_pmin
        self.enc_pmax = enc_pmax
        self.weight_bits = weight_bits
        self.tau_mem_inv = tau_mem_inv
        self.tau_syn_inv = tau_syn_inv
        self.v_th = v_th

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
        I_sample = self.current(x)
        state = None
        acc = torch.zeros_like(I_sample)
        for t in range(self.T):
            x_encoded = self._encode_x(x)
            I_t = self.current(x_encoded)
            spikes, state = self.lif_cell(I_t, state)
            acc = acc + spikes
        return acc / float(self.T) if self.return_rate else acc


class NorseLIFLinear(NorseLIFBase):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(output_features=out_features, **kwargs)
        self.fc = nn.Linear(in_features, out_features, bias=bias)
    def current(self, x):
        w_quant = self.quantize_weights(self.fc.weight)
        return F.linear(x, w_quant, self.fc.bias)


class NorseLIFConv2d(NorseLIFBase):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False, **kwargs):
        super().__init__(output_features=c_out, **kwargs)
        self.cv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.stride = s
        self.padding = p
    def current(self, x):
        w_quant = self.quantize_weights(self.cv.weight)
        return F.conv2d(x, w_quant, self.cv.bias, stride=self.stride, padding=self.padding)


class NorseLIFConvT2d(NorseLIFBase):
    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False, **kwargs):
        super().__init__(output_features=c_out, **kwargs)
        self.cvt = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.stride = s
        self.padding = p
    def current(self, x):
        w_quant = self.quantize_weights(self.cvt.weight)
        return F.conv_transpose2d(x, w_quant, self.cvt.bias, stride=self.stride, padding=self.padding)


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


# =============================================================================
# MEMRISTOR CROSSBAR SIMULATION
# =============================================================================

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, wire_resistance=0.0,
                 device_variation=0.02, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_bits = weight_bits
        self.adc_bits = adc_bits
        self.dac_bits = dac_bits
        self.noise_std = noise_std
        self.wire_resistance = wire_resistance
        self.device_variation = device_variation
        self.register_buffer('device_variation_mask', None)

    def _init_device_variation(self, shape):
        if self.device_variation_mask is None and self.device_variation > 0:
            self.device_variation_mask = torch.randn(shape, device=self.weight.device) * self.device_variation

    def quantize_weights(self, w):
        n_levels = 2 ** self.weight_bits
        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8
        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min
        return w + (w_quant - w).detach()

    def add_device_variation(self, w):
        if self.device_variation > 0:
            self._init_device_variation(w.shape)
            w_varied = w * (1.0 + self.device_variation_mask)
        else:
            w_varied = w
        return w_varied

    def add_read_noise(self, w):
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(w) * self.noise_std
            w_noisy = w * (1.0 + noise)
        else:
            w_noisy = w
        return w_noisy

    def dac_quantize(self, x):
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

    def apply_wire_resistance(self, y):
        if self.wire_resistance > 0:
            n_out = y.size(-1)
            position_factor = torch.linspace(1.0, 1.0 - self.wire_resistance, n_out, device=y.device)
            y = y * position_factor
        return y

    def forward(self, x):
        x_dac = self.dac_quantize(x)
        w_quant = self.quantize_weights(self.weight)
        w_varied = self.add_device_variation(w_quant)
        w_noisy = self.add_read_noise(w_varied)
        y = F.linear(x_dac, w_noisy, None)
        y = self.apply_wire_resistance(y)
        y_adc = self.adc_quantize(y)
        if self.bias is not None:
            y_adc = y_adc + self.bias
        return y_adc

    def get_energy_breakdown(self, batch_size=1, input_sparsity=0.1):
        E_DAC_per_sample = 10e-12
        E_ADC_per_sample = 20e-12
        E_MAC_analog = 2e-15
        E_DAC = E_DAC_per_sample * (2 ** (self.dac_bits - 8))
        E_ADC = E_ADC_per_sample * (2 ** (self.adc_bits - 8))
        n_in = self.weight.size(1)
        n_out = self.weight.size(0)
        E_dac_total = n_in * E_DAC * batch_size
        n_active_ops = n_in * n_out * input_sparsity
        E_mvm_total = n_active_ops * E_MAC_analog * batch_size
        E_adc_total = n_out * E_ADC * batch_size
        E_total = E_dac_total + E_mvm_total + E_adc_total
        return {'E_dac_J': E_dac_total, 'E_mvm_J': E_mvm_total, 'E_adc_J': E_adc_total, 'E_total_J': E_total,
                'breakdown_percent': {
                    'DAC': E_dac_total / E_total * 100 if E_total > 0 else 0,
                    'MVM': E_mvm_total / E_total * 100 if E_total > 0 else 0,
                    'ADC': E_adc_total / E_total * 100 if E_total > 0 else 0,
                },
                'params': {'weight_bits': self.weight_bits, 'adc_bits': self.adc_bits,
                          'dac_bits': self.dac_bits, 'n_synapses': n_in * n_out}}


# =============================================================================
# CROSSBAR + NORSE LIF INTEGRATION
# =============================================================================

class CrossbarLIFLinear(NorseLIFBase):
    def __init__(self, in_features, out_features, bias=True,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, wire_resistance=0.0,
                 device_variation=0.02, **kwargs):
        super().__init__(output_features=out_features, **kwargs)
        self.fc = QuantizedLinear(in_features, out_features, bias=bias,
                                 weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                                 noise_std=noise_std, wire_resistance=wire_resistance,
                                 device_variation=device_variation)

    def current(self, x):
        return self.fc(x)

    def get_layer_energy(self, batch_size=1, input_sparsity=0.1):
        crossbar_energy = self.fc.get_energy_breakdown(batch_size, input_sparsity)
        E_spike = 23.6e-12
        E_neuron_leak = 0.5e-12
        n_neurons = self.output_features
        expected_spikes = n_neurons * input_sparsity * batch_size
        E_lif_spikes = expected_spikes * E_spike
        E_lif_leak = n_neurons * E_neuron_leak * batch_size
        E_lif_per_timestep = E_lif_spikes + E_lif_leak
        E_crossbar_total = crossbar_energy['E_total_J'] * self.T
        E_lif_total = E_lif_per_timestep * self.T
        E_total = E_crossbar_total + E_lif_total
        return {'E_crossbar_J': E_crossbar_total, 'E_lif_J': E_lif_total, 'E_total_J': E_total,
                'params': {**crossbar_energy['params'], 'n_neurons': n_neurons, 'n_timesteps': self.T}}


class CrossbarLIFConv2d(NorseLIFBase):
    def __init__(self, c_in, c_out, k=3, s=1, p=1, bias=False,
                 weight_bits=8, adc_bits=8, noise_std=0.01, **kwargs):
        crossbar_params = {'weight_bits': weight_bits, 'adc_bits': adc_bits, 'noise_std': noise_std}
        kwargs_norse = {k_: v for k_, v in kwargs.items()
                       if k_ not in ['weight_bits', 'adc_bits', 'dac_bits',
                                   'noise_std', 'device_variation', 'wire_resistance']}
        super().__init__(output_features=c_out, **kwargs_norse)
        self.cv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.weight_bits = crossbar_params['weight_bits']
        self.adc_bits = crossbar_params['adc_bits']
        self.noise_std = crossbar_params['noise_std']

    def quantize_conv_weights(self, w):
        n_levels = 2 ** self.weight_bits
        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8
        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min
        return w + (w_quant - w).detach()

    def adc_quantize(self, y):
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
        w_quant = self.quantize_conv_weights(self.cv.weight)
        y = F.conv2d(x, w_quant, self.cv.bias, stride=self.cv.stride, padding=self.cv.padding)
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise
        y = self.adc_quantize(y)
        return y


class CrossbarLIFConvT2d(NorseLIFBase):
    def __init__(self, c_in, c_out, k=4, s=2, p=1, bias=False,
                 weight_bits=8, adc_bits=8, noise_std=0.01, **kwargs):
        crossbar_params = {'weight_bits': weight_bits, 'adc_bits': adc_bits, 'noise_std': noise_std}
        kwargs_norse = {k_: v for k_, v in kwargs.items()
                       if k_ not in ['weight_bits', 'adc_bits', 'dac_bits',
                                   'noise_std', 'device_variation', 'wire_resistance']}
        super().__init__(output_features=c_out, **kwargs_norse)
        self.cvt = nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=bias)
        self.weight_bits = crossbar_params['weight_bits']
        self.adc_bits = crossbar_params['adc_bits']
        self.noise_std = crossbar_params['noise_std']

    def quantize_conv_weights(self, w):
        n_levels = 2 ** self.weight_bits
        w_min = w.min()
        w_max = w.max()
        w_range = w_max - w_min + 1e-8
        w_normalized = (w - w_min) / w_range
        w_quant_norm = torch.round(w_normalized * (n_levels - 1)) / (n_levels - 1)
        w_quant = w_quant_norm * w_range + w_min
        return w + (w_quant - w).detach()

    def adc_quantize(self, y):
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
        w_quant = self.quantize_conv_weights(self.cvt.weight)
        y = F.conv_transpose2d(x, w_quant, self.cvt.bias, stride=self.cvt.stride, padding=self.cvt.padding)
        if self.noise_std > 0 and self.training:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise
        y = self.adc_quantize(y)
        return y


# =============================================================================
# SCNN MODELS WITH MEMRISTOR CROSSBAR
# =============================================================================

class SCNNGen_Crossbar_Simple2D(nn.Module):
    def __init__(self, cond_dim, d, proj_hw=(12,8), ch=(32,64),
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 weight_bits=8, adc_bits=8, dac_bits=8,
                 noise_std=0.01, device_variation=0.02,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        H0,W0, C0,C1 = *proj_hw, *ch
        self.d, self.C0, self.HW = d, C0, (H0,W0)
        kw_spk = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=True, enc_mode=enc_mode, enc_gamma=enc_gamma,
                      weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                      noise_std=noise_std, device_variation=device_variation)
        kw_ro = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                     v_th=v_th, return_rate=False, enc_mode=enc_mode, enc_gamma=enc_gamma,
                     weight_bits=weight_bits, adc_bits=adc_bits, dac_bits=dac_bits,
                     noise_std=noise_std, device_variation=device_variation)
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
        layers = [('proj', self.proj), ('c1', self.c1), ('c2', self.c2), ('ro', self.ro)]
        total_energy = 0
        layer_breakdown = {}
        for name, layer in layers:
            if hasattr(layer, 'get_layer_energy'):
                energy = layer.get_layer_energy(batch_size, input_sparsity)
                layer_breakdown[name] = energy
                total_energy += energy['E_total_J']
        return {'E_total_J': total_energy, 'E_total_mJ': total_energy * 1e3, 'layer_breakdown': layer_breakdown}


# =============================================================================
# CNN & SCNN MODELS
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


class SCNNGen_LIF_Simple2D(nn.Module):
    def __init__(self, cond_dim, d, proj_hw=(12,8), ch=(32,64),
                 T=8, tau_mem_inv=100.0, tau_syn_inv=200.0, v_th=1.0,
                 enc_mode='poisson', enc_gamma=1.5):
        super().__init__()
        H0,W0, C0,C1 = *proj_hw, *ch
        self.d, self.C0, self.HW = d, C0, (H0,W0)
        kw_spk = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                      v_th=v_th, return_rate=True, enc_mode=enc_mode, enc_gamma=enc_gamma)
        kw_ro = dict(T=T, tau_mem_inv=tau_mem_inv, tau_syn_inv=tau_syn_inv,
                     v_th=v_th, return_rate=False, enc_mode=enc_mode, enc_gamma=enc_gamma)
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


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def optimize_single_state_crossbar(model_class, N=3, method='M1', M=12, steps=200, lr=1e-3,
                                   noise_std=0.0, normalize_cond=True, warm_spiking=False,
                                   weight_bits=8, adc_bits=8, dac_bits=8,
                                   device_variation=0.02, read_noise=0.01,
                                   use_amp=True, log_energy=True):
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
    G = model_class(cond_dim, d).to(DEVICE)

    if warm_spiking:
        warm_init_spiking(G, w_scale=3.0, bias=0.1)
        if hasattr(G, 'kickstart'):
            G.kickstart = True

    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda t: 1.0 / (1.0 + 0.96 * t / max(steps, 1)))

    use_amp_actual = use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp_actual)

    f_hist, l_hist, e_hist = [], [], []

    print(f"    Training with {weight_bits}-bit weights, {adc_bits}-bit ADC/DAC")
    print(f"    Device variation: {device_variation*100:.1f}%, Read noise: {read_noise*100:.1f}%")
    sys.stdout.flush()

    t0 = time.time()
    for it in range(1, steps+1):
        G.train()
        with autocast(enabled=use_amp_actual):
            rho_hat = G(cond_vec)
            if method == 'M1':
                y_hat = exp(rho_hat, None)
            else:
                y_hat = probs_from_bases_torch(rho_hat, bases)
            if noise_std > 0:
                y_hat = y_hat + noise_std * torch.randn_like(y_hat)
            loss = F.mse_loss(y_hat, target)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        if it == 20 and hasattr(G, 'kickstart'):
            G.kickstart = False

        with torch.no_grad():
            F_vec, F_mean = fidelity_batch(rho_hat, rho_t_ri)
        f_hist.append(F_mean)
        l_hist.append(float(loss.item()))

        if log_energy and it % 50 == 0:
            with torch.no_grad():
                energy = G.get_model_energy(batch_size=1, input_sparsity=0.1)
                e_hist.append(energy['E_total_mJ'])

        if it % 100 == 0 or it == steps:
            energy_str = f", E={e_hist[-1]:.4f}mJ" if e_hist else ""
            print(f"      Iter {it:3d}: Loss={loss.item():.6f}, F={F_mean:.4f}{energy_str}")
            sys.stdout.flush()

    elapsed = time.time() - t0

    G.eval()
    with torch.no_grad():
        rho_final = G(cond_vec)
        F_vec, F_final = fidelity_batch(rho_final, rho_t_ri)
        energy_final = G.get_model_energy(batch_size=1, input_sparsity=0.1)

    print(f"    Final: F={F_final:.4f}, E={energy_final['E_total_mJ']:.4f}mJ, Time={elapsed:.1f}s")
    sys.stdout.flush()

    del opt, scaler, scheduler, cond_vec, target, rho_t_ri
    if method == 'M1':
        del exp, ops_ri_t
    else:
        del bases
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'model': G, 'F_best': F_final, 'F_hist': f_hist, 'loss_hist': l_hist,
        'energy_hist': e_hist, 'energy_final': energy_final, 'time_sec': elapsed,
        'config': {'N': N, 'method': method, 'weight_bits': weight_bits,
                  'adc_bits': adc_bits, 'dac_bits': dac_bits,
                  'device_variation': device_variation, 'read_noise': read_noise}
    }


def optimize_single_state(model_class, N=3, method='M1', M=12, steps=200, lr=1e-3,
                          noise_std=0.0, normalize_cond=True, warm_spiking=False,
                          use_amp=True, use_compile=False):
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
    G = model_class(cond_dim, d).to(DEVICE)

    if warm_spiking and 'SCNN' in G.__class__.__name__:
        warm_init_spiking(G, w_scale=3.0, bias=0.1)
        if hasattr(G, 'kickstart'): G.kickstart = True

    opt = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.9))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda t: 1.0 / (1.0 + 0.96 * t / max(steps, 1)))

    use_amp = use_amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    f_hist, l_hist = [], []
    for it in range(1, steps+1):
        G.train()
        with autocast(enabled=use_amp):
            rho_hat = G(cond_vec)
            if method == 'M1':
                y_hat = exp(rho_hat, None)
            else:
                y_hat = probs_from_bases_torch(rho_hat, bases)
            if noise_std > 0:
                y_hat = y_hat + noise_std * torch.randn_like(y_hat)
            loss = F.mse_loss(y_hat, target)

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

        if it % 100 == 0 or it == steps:
            print(f"      Iter {it:3d}: Loss={loss.item():.6f}, F={F_mean:.4f}")
            sys.stdout.flush()

    with torch.no_grad():
        rho_final = G(cond_vec)

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


# =============================================================================
# T-SWEEP CONFIGURATION
# =============================================================================

T_LIST = [2, 4, 8, 16, 32, 64, 128]
N_LIST = [3, 5, 8]
METHODS = ['M1', 'M2']
M_M1 = 128
M_M2 = 12
SPARSITY = 0.1

STEPS_DEFAULT = 500
STEPS_N8 = 500
LR = 1e-3

V_TH = 0.9
ENC_GAMMA = 1.5
TAU_MEM_INV = 100.0
TAU_SYN_INV = 200.0

CROSSBAR_NOISE = 0.01
CROSSBAR_VARIATION = 0.02

MODEL_CONFIGS = [
    {'name': 'Norse-Simple2D', 'is_crossbar': False, 'bits': None},
    {'name': 'Crossbar-8b',    'is_crossbar': True,  'bits': 8},
    {'name': 'Crossbar-4b',    'is_crossbar': True,  'bits': 4},
]


# =============================================================================
# T-SWEEP CORE
# =============================================================================

def make_model_fn(config, T, N):
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


def estimate_inference_energy(model, config, T):
    if config['name'] == 'Norse-Simple2D':
        return estimate_loihi_inference_energy(model, T=T, sparsity=SPARSITY)
    return estimate_crossbar_inference_energy(model, T=T, sparsity=SPARSITY, bits=config['bits'])


def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_T_sweep(T_list=T_LIST, N_list=N_LIST, methods=METHODS,
                model_configs=MODEL_CONFIGS, use_amp=True,
                checkpoint_every=10):
    if _ARGS.quick:
        T_list = [4, 8]
        N_list = [3]
        methods = ['M1']
        model_configs = model_configs[:1]
        print(f"[quick] T={T_list} N={N_list} methods={methods}", flush=True)

    if CSV_PATH.exists():
        _df_existing = pd.read_csv(CSV_PATH)
        rows = _df_existing.to_dict('records')
        _done_keys = {(int(r['seed']), r['Model'], int(r['N']),
                       r['method'], float(r['T']))
                      for r in rows if r.get('seed') == SEED}
        print(f"  [resume] {len(_done_keys)} rows for seed={SEED}", flush=True)
    else:
        rows = []
        _done_keys = set()

    total = len(T_list) * len(N_list) * len(methods) * len(model_configs)
    run_idx = 0

    for method in methods:
        for N in N_list:
            d_N = 2**N
            if method == 'M1':
                M = min(4**N - 1, max(M_M1, 2 * d_N))
            else:
                M = max(M_M2, 5 * d_N // 4)
            steps = STEPS_N8 if N == 8 else STEPS_DEFAULT
            if _ARGS.quick:
                steps = 3

            for T_val in T_list:
                for config in model_configs:
                    run_idx += 1
                    name = config['name']
                    if (SEED, name, N, method, float(T_val)) in _done_keys:
                        print(f"  [skip] seed={SEED} {name} N={N} {method} T={T_val}",
                              flush=True)
                        continue

                    print(f"\n[{run_idx}/{total}] seed={SEED} {method} | N={N} | T={T_val} | {name}")
                    sys.stdout.flush()

                    gpu_cleanup()

                    if torch.cuda.is_available():
                        free_gb = torch.cuda.mem_get_info()[0] / 1e9
                        total_gb = torch.cuda.mem_get_info()[1] / 1e9
                        print(f"  GPU memory: {free_gb:.1f}/{total_gb:.1f} GB free")

                    model_fn = make_model_fn(config, T_val, N)
                    t0 = time.time()
                    model = None
                    res = None
                    try:
                        if config['is_crossbar']:
                            res = optimize_single_state_crossbar(
                                model_class=model_fn, N=N, method=method, M=M,
                                steps=steps, lr=LR, warm_spiking=True,
                                weight_bits=config['bits'], adc_bits=config['bits'],
                                dac_bits=config['bits'],
                                device_variation=CROSSBAR_VARIATION,
                                read_noise=CROSSBAR_NOISE,
                                use_amp=use_amp, log_energy=False)
                            F_best = res['F_best']
                            F_hist = res['F_hist']
                            model = res['model']
                        else:
                            res = optimize_single_state(
                                model_class=model_fn, N=N, method=method, M=M,
                                steps=steps, lr=LR, warm_spiking=True, use_amp=use_amp)
                            f_hist = res['f_hist']
                            F_best = float(np.max(f_hist))
                            F_hist = f_hist.tolist() if isinstance(f_hist, np.ndarray) else f_hist
                            model = res['model']

                        elapsed = time.time() - t0
                        E_inf = estimate_inference_energy(model, config, T_val)
                        n_params = sum(p.numel() for p in model.parameters())

                        row = {
                            'seed': SEED,
                            'Model': name, 'N': N, 'method': method, 'T': T_val,
                            'F_best': F_best,
                            'F_last': float(F_hist[-1]) if F_hist else 0.0,
                            'E_inference_uJ': E_inf['E_total_uJ'],
                            'n_params': n_params, 'time_sec': elapsed,
                        }
                        rows.append(row)
                        print(f"  F_best={F_best:.4f}, E_inf={E_inf['E_total_uJ']:.4f}uJ, "
                              f"params={n_params}, time={elapsed:.1f}s")

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f"  FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        rows.append({
                            'seed': SEED,
                            'Model': name, 'N': N, 'method': method, 'T': T_val,
                            'F_best': 0.0, 'F_last': 0.0,
                            'E_inference_uJ': float('nan'),
                            'n_params': 0, 'time_sec': elapsed,
                        })

                    del model, res
                    gpu_cleanup()

                    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)  # save every iter
                    sys.stdout.flush()

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nSweep complete! {len(df)} rows saved -> {CSV_PATH}")
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print(f"\nT-Sweep Configuration (mixed states):")
    print(f"  T values: {T_LIST}")
    print(f"  N values: {N_LIST}")
    print(f"  Methods:  {METHODS}")
    print(f"  Models:   {[c['name'] for c in MODEL_CONFIGS]}")
    sys.stdout.flush()
    df_sweep = run_T_sweep()
    print(f"\nDone. {len(df_sweep)} rows -> {CSV_PATH}")
