
from __future__ import annotations


MODELS = (
    ("non_learning", "cblof", "CBLOF", "tsad_benchmark.baselines.non_learning.cblof.CBLOFModel", {"n_clusters": 8, "contamination": 0.05}, "point_concat", {}),
    ("non_learning", "hbos", "HBOS", "tsad_benchmark.baselines.non_learning.hbos.HBOSModel", {"n_bins": 10, "contamination": 0.05}, "point_concat", {}),
    ("non_learning", "lof", "LOF", "tsad_benchmark.baselines.non_learning.lof.LOFModel", {"n_neighbors": 20, "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "dagmm", "DAGMM", "tsad_benchmark.baselines.machine_learning.dagmm.DAGMMModel", {"hidden_dim": 64, "n_gmm": 4, "sequence_len": 1, "num_epochs": 10, "batch_size": 256, "lr": 0.0001}, "model_segmented", {}),
    ("machine_learning", "deeppoint", "DeepPoint", "tsad_benchmark.baselines.machine_learning.deeppoint.DeepPointModel", {"enable_threshold": False}, "model_segmented", {}),
    ("machine_learning", "eif", "EIF", "tsad_benchmark.baselines.machine_learning.eif.EIFModel", {"n_estimators": 200, "sample_size": 256}, "point_concat", {}),
    ("machine_learning", "iforest", "IForest", "tsad_benchmark.baselines.machine_learning.iforest.IForestModel", {"n_estimators": 200, "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "kmeans", "KMeans", "tsad_benchmark.baselines.machine_learning.kmeans.KMeansModel", {"n_clusters": 20, "window_size": 50}, "kmeans_segmented", {}),
    ("machine_learning", "knn", "KNN", "tsad_benchmark.baselines.machine_learning.knn.KNNModel", {"n_neighbors": 5, "method": "largest", "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "loda", "LODA", "tsad_benchmark.baselines.machine_learning.loda.LODAModel", {"n_bins": 10, "n_random_cuts": 100, "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "ocsvm", "OCSVM", "tsad_benchmark.baselines.machine_learning.ocsvm.OCSVMModel", {"kernel": "rbf", "nu": 0.05, "gamma": "auto", "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "pca", "PCA", "tsad_benchmark.baselines.machine_learning.pca.PCAModel", {"weighted": True, "standardization": True, "contamination": 0.05}, "point_concat", {}),
    ("machine_learning", "torsk", "Torsk", "tsad_benchmark.baselines.machine_learning.torsk.TorskModel", {"reservoir_dim": 256, "spectral_radius": 0.9, "leak_rate": 0.3, "ridge_lambda": 0.0001, "transient": 200}, "model_segmented", {}),
    ("deep_learning", "ae", "AutoEncoder", "tsad_benchmark.baselines.deep_learning.ae.AutoEncoderModel", {"hidden_dim": 64, "num_epochs": 10}, "model_segmented", {}),
    ("deep_learning", "anomaly_transformer", "AnomalyTransformer", "tsad_benchmark.baselines.deep_learning.anomaly_transformer.AnomalyTransformerModel", {"win_size": 100, "lr": 0.0001, "num_epochs": 3}, "dlbase_segmented", {}),
    ("deep_learning", "catch", "CATCH", "tsad_benchmark.baselines.deep_learning.catch.CATCHAnomalyModel", {"win_size": 96, "patch_size": 16, "patch_stride": 16, "num_epochs": 3, "batch_size": 16, "d_model": 16, "d_ff": 64, "cf_dim": 32, "head_dim": 16, "n_heads": 2, "auxi_lambda": 0.1, "dc_lambda": 0.1}, "dlbase_segmented", {}),
    ("deep_learning", "d3r", "D3R", "tsad_benchmark.baselines.deep_learning.d3r.D3RModel", {"win_size": 96, "num_epochs": 5}, "model_segmented", {}),
    ("deep_learning", "dcdetector", "DCdetector", "tsad_benchmark.baselines.deep_learning.dcdetector.DCdetectorModel", {"win_size": 105, "patch_size": [3, 5, 7], "num_epochs": 3, "batch_size": 16}, "dlbase_segmented", {}),
    ("deep_learning", "duet", "DUET", "tsad_benchmark.baselines.deep_learning.duet.DUETAnomalyModel", {"win_size": 96, "num_epochs": 3, "num_experts": 4, "k": 1, "CI": True, "batch_size": 64}, "dlbase_segmented", {}),
    ("deep_learning", "gdn", "GDN", "tsad_benchmark.baselines.deep_learning.gdn.GDNModel", {"win_size": 48, "slide_stride": 6, "num_epochs": 30, "patience": 5, "topk": 15, "eval_topk": 3, "dim": 64, "batch_size": 64, "out_layer_num": 1, "out_layer_inter_dim": 256, "val_ratio": 0.1, "decay": 0.0}, "model_segmented", {}),
    ("deep_learning", "lstmed", "LSTMED", "tsad_benchmark.baselines.deep_learning.lstmed.LSTMEDModel", {"hidden_dim": 64, "num_epochs": 10, "batch_size": 64}, "model_segmented", {}),
    ("deep_learning", "mscred", "MSCRED", "tsad_benchmark.baselines.deep_learning.mscred.MSCREDModel", {"signature_scales": [10, 30, 60], "step_max": 5, "gap_time": 10, "feature_mode": "sensor_avg", "sensor_stats": ["avg"], "batch_size": 32, "num_epochs": 5, "lr": 0.0002}, "model_segmented", {}),
    ("deep_learning", "mtad_gat", "MTAD-GAT", "tsad_benchmark.baselines.deep_learning.mtad_gat.MTADGATModel", {"win_size": 100, "batch_size": 64, "num_epochs": 5, "max_features": 256}, "model_segmented", {}),
    ("deep_learning", "mtgflow", "MTGFlow", "tsad_benchmark.baselines.deep_learning.mtgflow.MTGFlowModel", {"win_size": 60, "batch_size": 64, "num_epochs": 40, "train_stride": 10}, "model_segmented", {}),
    ("deep_learning", "omnianomaly", "OmniAnomaly", "tsad_benchmark.baselines.deep_learning.omnianomaly.OmniAnomalyModel", {"win_size": 100, "latent_dim": 8, "num_epochs": 10}, "model_segmented", {}),
    ("deep_learning", "sarad", "SARAD", "tsad_benchmark.baselines.deep_learning.sarad.SARADModel", {"win_size": 100, "batch_size": 8, "train_stride": 8, "num_epochs": 3, "model_size": 256, "num_layers": 2, "num_heads": 4, "detector_size": 64}, "model_segmented", {}),
    ("deep_learning", "timesnet", "TimesNet", "tsad_benchmark.baselines.deep_learning.timesnet.TimesNetModel", {"win_size": 100, "d_model": 64, "d_ff": 64, "e_layers": 2, "top_k": 5, "num_epochs": 3}, "dlbase_segmented", {}),
    ("deep_learning", "tranad", "TranAD", "tsad_benchmark.baselines.deep_learning.tranad.TranADModel", {"win_size": 10, "lr": 0.001, "num_epochs": 5}, "dlbase_segmented", {}),
    ("deep_learning", "usad", "USAD", "tsad_benchmark.baselines.deep_learning.usad.USADModel", {"win_size": 48, "hidden_dim": 64, "num_epochs": 30}, "model_segmented", {}),
    ("llm_based", "gpt4ts", "GPT4TS", "tsad_benchmark.baselines.llm_based.gpt4ts.GPT4TSModel", {"backbone": "models/gpt2", "local_files_only": True, "win_size": 100, "num_epochs": 10, "batch_size": 64, "sampling_rate": 0.05, "sampling_strategy": "uniform", "gpt_layers": 3, "d_ff": 32, "lradj": "type1", "device": "cuda", "max_features": 768, "feature_select": "train_variance"}, "model_segmented", {}),
    ("llm_based", "unitime", "UniTime", "tsad_benchmark.baselines.llm_based.unitime.UniTimeModel", {"model_path": "models/gpt2", "local_files_only": True, "win_size": 96, "num_epochs": 10, "patience": 10, "batch_size": 16, "sampling_rate": 0.05, "val_sample_rate": 0.05, "train_score_sample_rate": 0.05, "sampling_strategy": "uniform", "stride": 16, "max_token_num": 17, "max_backcast_len": 96, "max_forecast_len": 0, "device": "cuda"}, "model_segmented", {}),
    ("ts_pretrained", "chronos", "Chronos", "tsad_benchmark.baselines.ts_pretrained.chronos.ChronosModel", {"model_id": "models/chrones-bolt-base", "local_files_only": True, "context_length": 96, "prediction_length": 1, "batch_size": 64, "forecast_batch_size": 2048, "device": "cuda"}, "model_segmented", {}),
    ("ts_pretrained", "dada", "DADA", "tsad_benchmark.baselines.ts_pretrained.dada.DADAModel", {"model_id": "models/DADA", "local_files_only": True, "mode": "zero_shot", "win_size": 100, "batch_size": 16, "norm": False, "score_mode": "mse", "copies": 10, "device": "cuda"}, "model_segmented", {}),
    ("ts_pretrained", "moment", "MOMENT", "tsad_benchmark.baselines.ts_pretrained.moment.MOMENTModel", {"model_id": "models/MOMENT-1-large", "local_files_only": True, "win_size": 96, "batch_size": 8, "fine_tune_epochs": 3, "fine_tune_batch_size": 8, "fine_tune_lr": 0.0001, "fine_tune_step": 1, "fine_tune_sample_rate": 0.05, "fine_tune_val_ratio": 0.2, "fine_tune_val_sample_rate": 0.05, "fine_tune_patience": 3, "device": "cuda"}, "model_segmented", {}),
    ("ts_pretrained", "units", "UniTS", "tsad_benchmark.baselines.ts_pretrained.units.UniTSModel", {"checkpoint_path": "models/UniTS/units_x32_pretrain_checkpoint.pth", "local_files_only": True, "mode": "zero_shot", "win_size": 96, "device": "cuda"}, "model_segmented", {}),
    ("finetune_llm", "rpcl_tcne_mts_llm", "RPCL-TCNE-MTS-LLM", "tsad_benchmark.baselines.finetune_llm.rpcl_tcne_mts_llm.RPCLTCNEMTSLLMModel", {"backbone": "models/gpt2", "local_files_only": True, "win_size": 22, "train_stride": 4, "batch_size": 512, "num_epochs": 6, "lr": 0.0004, "tcne_hidden_dim": 44, "tcne_blocks": 10, "kernel_size": 3, "dropout": 0.2, "lora_rank": 4}, "model_segmented", {}),
)


def entries():
    return [
        {
            "category": category,
            "slug": slug,
            "model_name": name,
            "model_path": path,
            "model_hyper_params": params,
            "expected_output": "score",
            "transfer_adapter": adapter,
            "transfer_adapter_params": adapter_params,
        }
        for category, slug, name, path, params, adapter, adapter_params in MODELS
    ]


__all__ = ["MODELS", "entries"]
