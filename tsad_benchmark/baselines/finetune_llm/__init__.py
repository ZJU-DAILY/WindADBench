# -*- coding: utf-8 -*-


try:
    from tsad_benchmark.baselines.finetune_llm.rpcl_tcne_mts_llm import (
        RPCLTCNEMTSLLMModel,
        make_rpcl_tcne_mts_llm,
    )
except Exception:  # pragma: no cover
    RPCLTCNEMTSLLMModel = None  # type: ignore[assignment]
    make_rpcl_tcne_mts_llm = None  # type: ignore[assignment]

__all__ = ["RPCLTCNEMTSLLMModel", "make_rpcl_tcne_mts_llm"]
