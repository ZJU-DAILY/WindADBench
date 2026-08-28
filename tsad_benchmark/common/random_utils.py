# -*- coding: utf-8 -*-
"""Reproducibility helpers: align benchmark runs on a fixed seed = 2026."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

DEFAULT_SEED = 2026


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
