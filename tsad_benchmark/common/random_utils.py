from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np

DEFAULT_SEED = 2026


@contextmanager
def preserve_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        import torch
    except ImportError:
        torch = None
    torch_state = torch.get_rng_state() if torch is not None else None
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_initialized()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def fix_random_seed(seed: Optional[int] = DEFAULT_SEED) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
