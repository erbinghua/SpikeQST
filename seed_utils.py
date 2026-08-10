"""Deterministic seeding helpers shared by the experiment scripts."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for a single experiment run.

    Deterministic CUDA kernels are requested when ``deterministic`` is true.
    Some PyTorch/CUDA operations may still be nondeterministic; experiment logs
    should record the hardware, CUDA, cuDNN, and PyTorch versions used.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass
