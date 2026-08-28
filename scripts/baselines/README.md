# Baselines

Per-baseline launch scripts for the wind-turbine anomaly-detection benchmark.

Layout
------

Scripts are grouped by the categories used in our taxonomy of 36 baselines.
Each model gets one `.sh` per evaluation track (`score/` and `label/`).

```
baselines/
├── non_learning/         (3)  LOF, CBLOF, HBOS
├── machine_learning/    (10)  OCSVM, DeepPoint, KNN, KMeans, IF, EIF, LODA,
│                              PCA, DAGMM, Torsk
├── deep_learning/       (16)  organised by sub-family:
│   ├── transformer/      (3)  AnomalyTransformer, TranAD, DCdetector
│   ├── frequency/        (2)  TimesNet, CATCH
│   ├── classic/          (5)  AE, LSTMED, OmniAnomaly, MSCRED, USAD
│   ├── graph/            (3)  MTAD-GAT, GDN, DUET
│   └── novel/            (3)  THOC, D3R, MemMambaAD
├── llm_based/            (2)  GPT4TS, UniTime
├── ts_pretrained/        (4)  UniTS, MOMENT, DADA, Chronos
└── domain_llm/           (1)  RPCL-updated TCNE-MTS-LLM
```

Convention
----------

Every script follows the same template:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."   # adjust depth to repo root

python scripts/run_benchmark.py \
  --config-path        <strategy>.json \
  --model-name         <ModelName> \
  --model-path         tsad_benchmark.baselines.<category>.<module>.<Class or factory> \
  --model-hyper-params '<JSON>' \
  --save-path          results/<category>/<track>/<model>.csv \
  --report-dir         results/<category>/<track>/<model>_report
```

Score-track scripts include `--defer-score-vus` by default, so they compute
fast score metrics first and leave `vus_pr` / `vus_roc` for a later pass.  The
initial report is still written, but VUS entries are incomplete until the
second pass.  To fill deferred VUS metrics and regenerate the final reports:

```bash
bash scripts/recompute_score_vus.sh 8
```

The optional number is the row-level worker count passed to
`scripts/recompute_metrics.py --all --track score --metrics vus_pr vus_roc --workers <N> --report`.

Status
------

* **Implemented** scripts are marked in the script header and use the
  metadata-driven `unfixed_score.json` / `unfixed_label.json` templates by
  default.
* **Placeholder** scripts target an `--model-path` that does not yet exist;
  they will fail with a clear `ImportError` until the corresponding baseline
  is added under `tsad_benchmark/baselines/`.  The placeholder serves as the
  canonical command line the new model is expected to honour.

Adding a new baseline
---------------------

1. Implement the model under `tsad_benchmark/baselines/<category>/<module>.py`
   (subclass `AnomalyModelBase`) or expose a factory function.  ``<category>``
   should match the script tree (`non_learning`, `machine_learning`,
   `deep_learning/<sub>`, `llm_based`, `ts_pretrained`, `finetune_llm`).
2. Register it in the matching `tsad_benchmark/baselines/<category>/__init__.py`
   (and, if it is a commonly used model, in `tsad_benchmark/baselines/__init__.py`).
3. Edit the matching placeholder `.sh` file: update `--model-path`,
   `--model-hyper-params` and the `--save-path` if needed.
