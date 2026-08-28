# -*- coding: utf-8 -*-

from __future__ import annotations


# --- Optional-dependency baselines (lazy-imported) ------------------------

try:
    from tsad_benchmark.baselines.deep_learning.ae import AutoEncoderModel  # merlion + torch
except Exception:  # pragma: no cover
    AutoEncoderModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.lstmed import LSTMEDModel  # merlion + torch
except Exception:  # pragma: no cover
    LSTMEDModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.tranad import (  # torch
        TranADModel,
        make_tranad,
    )
except Exception:  # pragma: no cover
    TranADModel = None  # type: ignore[assignment]
    make_tranad = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.anomaly_transformer import (  # torch
        AnomalyTransformerModel,
        make_anomaly_transformer,
    )
except Exception:  # pragma: no cover
    AnomalyTransformerModel = None  # type: ignore[assignment]
    make_anomaly_transformer = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.dcdetector import (  # torch + einops
        DCdetectorModel,
        make_dcdetector,
    )
except Exception:  # pragma: no cover
    DCdetectorModel = None  # type: ignore[assignment]
    make_dcdetector = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.timesnet import (  # torch
        TimesNetModel,
        make_timesnet,
    )
except Exception:  # pragma: no cover
    TimesNetModel = None  # type: ignore[assignment]
    make_timesnet = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.catch import (  # torch + einops
        CATCHAnomalyModel,
        make_catch,
    )
except Exception:  # pragma: no cover
    CATCHAnomalyModel = None  # type: ignore[assignment]
    make_catch = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.duet import (  # torch + einops
        DUETAnomalyModel,
        make_duet,
    )
except Exception:  # pragma: no cover
    DUETAnomalyModel = None  # type: ignore[assignment]
    make_duet = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.gdn import (  # torch + torch_geometric
        GDNModel,
        make_gdn,
    )
except Exception:  # pragma: no cover
    GDNModel = None  # type: ignore[assignment]
    make_gdn = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.d3r import (  # torch
        D3RModel,
        make_d3r,
    )
except Exception:  # pragma: no cover
    D3RModel = None  # type: ignore[assignment]
    make_d3r = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.usad import (  # torch
        USADModel,
        make_usad,
    )
except Exception:  # pragma: no cover
    USADModel = None  # type: ignore[assignment]
    make_usad = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mtad_gat import (  # torch
        MTADGATModel,
        make_mtad_gat,
    )
except Exception:  # pragma: no cover
    MTADGATModel = None  # type: ignore[assignment]
    make_mtad_gat = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.omnianomaly import (  # torch
        OmniAnomalyModel,
        make_omnianomaly,
    )
except Exception:  # pragma: no cover
    OmniAnomalyModel = None  # type: ignore[assignment]
    make_omnianomaly = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mscred import (  # torch
        MSCREDModel,
        make_mscred,
    )
except Exception:  # pragma: no cover
    MSCREDModel = None  # type: ignore[assignment]
    make_mscred = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mtgflow import (  # torch
        MTGFlowModel,
        make_mtgflow,
    )
except Exception:  # pragma: no cover
    MTGFlowModel = None  # type: ignore[assignment]
    make_mtgflow = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.sarad import (  # torch + einops
        SARADModel,
        make_sarad,
    )
except Exception:  # pragma: no cover
    SARADModel = None  # type: ignore[assignment]
    make_sarad = None  # type: ignore[assignment]


__all__ = [
    "AnomalyTransformerModel",
    "AutoEncoderModel",
    "CATCHAnomalyModel",
    "D3RModel",
    "DCdetectorModel",
    "DUETAnomalyModel",
    "GDNModel",
    "LSTMEDModel",
    "MSCREDModel",
    "MTGFlowModel",
    "MTADGATModel",
    "OmniAnomalyModel",
    "SARADModel",
    "TimesNetModel",
    "TranADModel",
    "USADModel",
    "make_anomaly_transformer",
    "make_catch",
    "make_d3r",
    "make_dcdetector",
    "make_duet",
    "make_gdn",
    "make_mscred",
    "make_mtgflow",
    "make_mtad_gat",
    "make_omnianomaly",
    "make_sarad",
    "make_timesnet",
    "make_tranad",
    "make_usad",
]
