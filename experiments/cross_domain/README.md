# Cross-Domain Experiments

## Build the dataset

```bash
python experiments/cross_domain/build_core3_dataset.py --source-root . --output-root derived_datasets/core3_v1_care_v6
```

## Cross-turbine

```bash
bash experiments/cross_domain/scripts/baselines/non_learning/lof.sh A
bash experiments/cross_domain/scripts/baselines/non_learning/lof.sh B
bash experiments/cross_domain/scripts/baselines/non_learning/lof.sh C
```

## Cross-farm

```bash
bash experiments/cross_domain/scripts/run_full_fleet.sh non_learning/lof --foreground
```

## All cross-turbine models

```bash
REUSE_SOURCE=0 bash experiments/cross_domain/scripts/run_all.sh
```
