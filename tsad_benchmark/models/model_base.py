# -*- coding: utf-8 -*-
"""
Deprecated compatibility shim.

Use instead::

    from tsad_benchmark.models import AnomalyModelBase, ModelCapability

``ModelBase`` is kept as an alias for ``AnomalyModelBase`` so that legacy
imports do not break immediately.
"""

import warnings

from tsad_benchmark.models.base import AnomalyModelBase  # noqa: F401

warnings.warn(
    "tsad_benchmark.models.model_base is deprecated. "
    "Use: from tsad_benchmark.models import AnomalyModelBase",
    DeprecationWarning,
    stacklevel=2,
)

ModelBase = AnomalyModelBase
