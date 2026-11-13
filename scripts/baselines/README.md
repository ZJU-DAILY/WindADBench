# Baselines

Per-baseline launch scripts for the wind-turbine anomaly-detection benchmark.

Layout
------

Scripts are grouped by the categories used in our taxonomy of 36 baselines.
Each model is launched through its `score/` script.

```
baselines/
├── non_learning/         (3)  LOF, CBLOF, HBOS
├── machine_learning/     (8)  OCSVM, KNN, KMeans, IF, EIF, LODA, PCA, Torsk
├── deep_learning/       (18)  AE, AnomalyTransformer, CATCH, D3R, DAGMM,
│                              DCdetector, DeepPoint, DUET, GDN, LSTMED, MSCRED,
│                              MTAD-GAT, MTGFlow, OmniAnomaly, SARAD, TimesNet,
│                              TranAD, USAD
├── llm_based/            (2)  GPT4TS, UniTime
├── ts_pretrained/        (4)  UniTS, MOMENT, DADA, Chronos
└── finetune_llm/         (1)  RPCL-TCNE-MTS-LLM
```

Convention
----------

The `score/` launchers select the current CARE protocol through
`unfixed_score.json`. Each run fits once per series and produces continuous-score
metrics and threshold-derived alarm metrics at eight operating points together.
The model's public `detect_score` supplies training and test scores.
Thresholds use normal, published training clock states at
`0.1, 0.5, 1, 3, 5, 10, 15, 20` percent, with linear percentiles and strict
`score > threshold`. Test labels never select these deployment thresholds.

Scores publish at their actual window/batch availability; warning credit uses
publication pulses. A uses its full timeline. B/C evaluate status 0/2 on the
original timeline, with conditional Affiliation and explicit undefined values.
All curve metrics, including VUS, finish during the run.

The result CSV keeps raw scores, publication indices, held scores, alarms,
warning pulses, training scores, frozen thresholds and evaluation supports.
`inference_data` contains `[metric_scores, alarms]`: the first array is the
published, held score stream with unavailable positions filled using a low
value derived from training scores. `raw_score_data` and `warning_prediction_data`
retain the raw scores and warning pulses. Array payloads use Base64-encoded pickle.
The threshold method is `train_available_percentile_v2`.

Reports consume only this run's CSV. `tables/latest_main_by_farm.csv`,
`latest_main_macro.csv`, `by_farm_ratio.csv` and `macro_ratio.csv` use the
frozen operating-point → sequence → farm aggregation. Their time columns
are hours; raw event metrics remain sampling points. Normal exposure and
within-file false-alarm intervals use their explicit supports.

Model hyperparameters are specified in the launchers; data preparation and
random seeds follow the pipeline configuration. Each model retains its own
normalization and scoring implementation. Calibration scoring belongs to fit cost;
test scoring and alarm generation belong to inference cost. Serialization
and metric calculation are outside those phases.

Cost table
----------

The cost report exports eight metrics in `tables/latest_cost_macro.csv` and
`tables/latest_cost_by_farm.csv`: fit time, fit GPU peak, fit CPU utilization,
inference time, inference GPU peak, inference CPU utilization, neural parameter
count, and model size.
Time is in seconds; memory and serialized sizes use MiB (bytes / 1024**2).
CPU utilization is process CPU time / wall time / logical CPU count * 100,
as a percentage of the whole machine. GPU memory is PyTorch's peak allocated
memory (including resident model weights), or sampled per-process NVML memory
when the PyTorch counter is unavailable. NVML sampling can miss brief peaks.
CPU-only GPU fields and non-neural parameter counts are not applicable, not zero.

Parameter counts include frozen parameters. Model size follows each adapter's
saved representation: neural state dictionaries, or serialized estimator state
for classical/Merlion models (which may include cached training data).
EIF uses a structural estimate if its native forest does not expose its state;
its size is an estimate, not an exact checkpoint file size.

Performance and resource reports are generated from the current run by the report layer.
In-farm result CSVs retain `fit_peak_memory_mb`, `inference_peak_memory_mb`,
and `flops` as empty compatibility fields; they are outside the eight-metric
cost tables.

A typical launcher uses the following structure:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."

python scripts/run_benchmark.py \
  --config-path        unfixed_score.json \
  --model-name         <ModelName> \
  --model-path         tsad_benchmark.baselines.<category>.<module>.<Class or factory> \
  --model-hyper-params '<JSON>' \
  --save-path          results/<category>/score/<model>.csv \
  --report-dir         results/<category>/score/<model>_report
```

GDN and DeepPoint write under `results/source_frozen_v1/deep_learning/score/`;
TimesNet writes under `results/independent_window_v1/deep_learning/score/`.
Each launcher specifies its result and report paths.

The formal protocol completes VUS within the run. The compatibility
`--defer-score-vus` launcher flag does not defer the current protocol's metrics.

Status
------

The 36 score launchers use the metadata-driven `unfixed_score.json` template. Each model requires its corresponding
dependencies to be installed.

Adding a new baseline
---------------------

1. Implement the model under `tsad_benchmark/baselines/<category>/<module>.py`
   (subclass `AnomalyModelBase`) or expose a factory function.  ``<category>``
   should match the script tree (`non_learning`, `machine_learning`,
   `deep_learning`, `llm_based`, `ts_pretrained`, `finetune_llm`).
2. Register it in the matching `tsad_benchmark/baselines/<category>/__init__.py`
   (and, if it is a commonly used model, in `tsad_benchmark/baselines/__init__.py`).
3. Add the model's `.sh` launch scripts with its `--model-path`,
   `--model-hyper-params` and `--save-path`.
