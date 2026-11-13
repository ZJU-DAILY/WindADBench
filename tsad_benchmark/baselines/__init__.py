from tsad_benchmark.baselines.machine_learning.iforest import (
    IForestModel,
    make_isolation_forest,
)
from tsad_benchmark.baselines.machine_learning.kmeans import KMeansModel
from tsad_benchmark.baselines.machine_learning.knn import KNNModel
from tsad_benchmark.baselines.machine_learning.loda import LODAModel
from tsad_benchmark.baselines.machine_learning.ocsvm import OCSVMModel
from tsad_benchmark.baselines.machine_learning.pca import PCAModel
from tsad_benchmark.baselines.non_learning.cblof import CBLOFModel
from tsad_benchmark.baselines.non_learning.hbos import HBOSModel
from tsad_benchmark.baselines.non_learning.lof import LOFModel, make_lof


try:
    from tsad_benchmark.baselines.machine_learning.eif import EIFModel
except Exception:  # pragma: no cover
    EIFModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.torsk import TorskModel
except Exception:  # pragma: no cover
    TorskModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.dagmm import DAGMMModel
except Exception:  # pragma: no cover
    DAGMMModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.machine_learning.deeppoint import DeepPointModel
except Exception:  # pragma: no cover
    DeepPointModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.ae import AutoEncoderModel
except Exception:  # pragma: no cover
    AutoEncoderModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.lstmed import LSTMEDModel
except Exception:  # pragma: no cover
    LSTMEDModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.tranad import TranADModel
except Exception:  # pragma: no cover
    TranADModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.anomaly_transformer import (
        AnomalyTransformerModel,
    )
except Exception:  # pragma: no cover
    AnomalyTransformerModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.dcdetector import (
        DCdetectorModel,
    )
except Exception:  # pragma: no cover
    DCdetectorModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.timesnet import (
        TimesNetModel,
    )
except Exception:  # pragma: no cover
    TimesNetModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.catch import (
        CATCHAnomalyModel,
    )
except Exception:  # pragma: no cover
    CATCHAnomalyModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.duet import (
        DUETAnomalyModel,
    )
except Exception:  # pragma: no cover
    DUETAnomalyModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.gdn import GDNModel
except Exception:  # pragma: no cover
    GDNModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.d3r import D3RModel
except Exception:  # pragma: no cover
    D3RModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.usad import USADModel
except Exception:  # pragma: no cover
    USADModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mtad_gat import MTADGATModel
except Exception:  # pragma: no cover
    MTADGATModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.omnianomaly import OmniAnomalyModel
except Exception:  # pragma: no cover
    OmniAnomalyModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mscred import MSCREDModel
except Exception:  # pragma: no cover
    MSCREDModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.mtgflow import MTGFlowModel
except Exception:  # pragma: no cover
    MTGFlowModel = None  # type: ignore[assignment]

try:
    from tsad_benchmark.baselines.deep_learning.sarad import SARADModel
except Exception:  # pragma: no cover
    SARADModel = None  # type: ignore[assignment]


from tsad_benchmark.baselines import (  # noqa: E402
    deep_learning,
    finetune_llm,
    llm_based,
    machine_learning,
    non_learning,
    ts_pretrained,
)


__all__ = [
    "AnomalyTransformerModel",
    "AutoEncoderModel",
    "CATCHAnomalyModel",
    "CBLOFModel",
    "D3RModel",
    "DAGMMModel",
    "DCdetectorModel",
    "DUETAnomalyModel",
    "DeepPointModel",
    "EIFModel",
    "GDNModel",
    "HBOSModel",
    "IForestModel",
    "KMeansModel",
    "KNNModel",
    "LODAModel",
    "LOFModel",
    "LSTMEDModel",
    "MSCREDModel",
    "MTGFlowModel",
    "MTADGATModel",
    "OCSVMModel",
    "OmniAnomalyModel",
    "PCAModel",
    "SARADModel",
    "TimesNetModel",
    "TorskModel",
    "TranADModel",
    "USADModel",
    "deep_learning",
    "finetune_llm",
    "llm_based",
    "machine_learning",
    "non_learning",
    "ts_pretrained",
    "make_isolation_forest",
    "make_lof",
]
