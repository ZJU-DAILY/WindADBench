<h1 align="center">WindADBench</h1>

<p align="center">
  <strong>Benchmarking Accuracy, Earliness, Reliability, and Cost across Heterogeneous Wind-Farm SCADA</strong>
</p>

<p align="center">
  <a href="https://zju-daily.github.io/WindADBench/">https://zju-daily.github.io/WindADBench/</a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#data">Data</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#experiments">Experiments</a>
</p>

<p align="center">
  <a href="assets/overview.pdf">
    <img src="assets/overview.png" width="85%" alt="WindADBench overview">
  </a>
</p>

---

## ✨ Overview

WindADBench evaluates wind-turbine SCADA anomaly detectors as complete
**alarm systems**. We compare 36 detectors from six model families across four
tracks, examining not only detection quality but also warning earliness,
reliability, and cost.

### Highlights

- **Heterogeneous data** — three wind farms with different turbine fleets and sensor schemas.
- **Four evaluation tracks** — in-farm, normal-operation, cross-turbine, and cross-farm settings.
- **Broad model coverage** — 36 detectors from six families with detection and operational metrics.

### Benchmark at a Glance

| | Farm A | Farm B | Farm C | Total |
|---|---:|---:|---:|---:|
| Setting | Onshore | Offshore | Offshore | — |
| Turbines | 5 | 9 | 22 | 36 |
| Sensors | 54 | 63 | 238 | — |
| Features | 86 | 257 | 957 | — |
| Anomalous sequences | 11 | 6 | 27 | 44 |
| Normal sequences | 11 | 9 | 31 | 51 |
| All sequences | 22 | 15 | 58 | 95 |

The dataset contains 95 event-centered sequences with 10-minute SCADA
measurements spanning approximately 89 turbine-years.

### Evaluation Tracks

| Track | Setting | Evaluation target |
|---|---|---|
| **Track 1** | In-farm | Disjoint training and evaluation periods from the same turbine |
| **Track 2** | Normal operation | False-alarm stability on normal sequences |
| **Track 3** | Cross-turbine | Fixed held-out turbines within each farm |
| **Track 4** | Cross-farm | Six directed transfers among Farms A, B, and C |

Tracks 1–2 use each farm's full sensor schema, while Tracks 4 use the shared
semantic features `wind_speed`, `active_power`, and `rotor_speed`.

---

## 📊 Data

> **Data:** Download the [CARE to Compare dataset](https://doi.org/10.5281/zenodo.14006163)
> from Zenodo. The raw datasets are not included in this repository because of
> their size. Place the three wind-farm directories in the repository root as
> follows.

```text
WindADBench/
├── Wind Farm A/
│   └── Wind Farm A/
│       ├── datasets/*.csv
│       ├── event_info.csv
│       └── feature_description.csv
├── Wind Farm B/
│   └── Wind Farm B/
│       ├── datasets/*.csv
│       ├── event_info.csv
│       └── feature_description.csv
└── Wind Farm C/
    └── Wind Farm C/
        ├── datasets/*.csv
        ├── event_info.csv
        └── feature_description.csv
```

Each farm directory should contain event sequences, event metadata, and feature
descriptions. WindADBench builds `WIND_AD_META.csv` from this structure when
needed.

---

## 🚀 Quick Start

Clone the repository and create the released environment:

```bash
git clone https://github.com/ZJU-DAILY/WindADBench.git
cd WindADBench
git submodule update --init --recursive
conda env create -f environment.yml
conda activate wind_benchmark
```

We do not distribute external model weights with the repository. To prepare
them, follow the download instructions, expected paths, and checksums in
[models/README.md](models/README.md).

---

## 🧪 Experiments

Run a score-based baseline:

```bash
bash scripts/baselines/non_learning/score/lof.sh
```

Run its label-based evaluation:

```bash
bash scripts/baselines/non_learning/label/lof.sh
```

Recompute deferred VUS metrics:

```bash
bash scripts/recompute_score_vus.sh 8
```

Run a transfer experiment from source Farm A:

```bash
bash experiments/cross_domain/scripts/baselines/non_learning/lof.sh A
```

Replace `A` with `B` or `C` for another source farm.

### Repository Layout

```text
WindADBench/
├── config/                    # Evaluation templates
├── experiments/cross_domain/  # Track 3–4 experiments
├── models/                    # Model-asset instructions
├── scripts/baselines/         # Per-model launchers
└── tsad_benchmark/            # Benchmark implementation
```

You can find the complete experiment settings in the configuration files and
launch scripts.
