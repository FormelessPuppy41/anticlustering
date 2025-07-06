# Anticlustering Project

This repository contains all code needed to reproduce the experiments in our bachelor thesis at Erasmus University Rotterdam, including the original anticlustering methods and novel online extensions. The project is organized as a standard [Kedro](https://kedro.readthedocs.io/) pipeline, ensuring a clean, modular layout that can be easily adapted to new datasets or algorithms.

---

## Repository Structure

```
.
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── kaggle.json
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters_anticluster.yml
│   │   ├── parameters_data_simulation.yml
│   │   ├── parameters_kaggle_data.yml
│   │   ├── parameters_online_anticluster.yml
│   │   ├── parameters_visualisation.yml
│   │   ├── parameters.yml
│   │   └── spark.yml
│   └── local/
│       └── logging.yml
├── src/
│   └── anticlustering/
│       ├── constants/
│       │   ├── catalog.py
│       │   └── parameters.py
│       ├── core/
│       │   ├── loans/
│       │   │   └── loan.py
│       │   ├── vectorizer.py
│       │   └── [Solver base classes & streaming engine]
│       ├── pipelines/
│       │   ├── anticcluster/
│       │   │   ├── nodes.py
│       │   │   ├── pipeline.py
│       │   │   └── results.py
│       │   ├── data_simulation/
│       │   │   ├── nodes.py
│       │   │   └── pipeline.py
│       │   ├── kaggle_data/
│       │   │   ├── nodes.py
│       │   │   └── pipeline.py
│       │   ├── online_anticluster/
│       │   │   ├── nodes.py
│       │   │   └── pipeline.py
│       │   └── visualisation/
│       │       ├── nodes.py
│       │       └── pipeline.py
│       ├── solvers/
│       │   ├── edge_ilp.py
│       │   ├── exchange_heuristic.py
│       │   ├── kmeans_heuristic.py
│       │   └── matching_heuristic.py
│       └── streaming/
│           ├── random_data_store.py
│           ├── random_simulator.py
│           ├── random_stream_manager.py
│           └── stream_manager.py
└── tests/
    └── [unit and integration tests]
```

* **data/**: All raw, intermediate, and final datasets in CSV or Pickle (`.pkl`) format, organized by Kedro’s catalog stages (`01_raw` → `08_reporting`).
* **conf/**: YAML configuration files for catalog definitions, parameters, and environment-specific settings.
* **src/anticlustering/constants/**: Static definitions (catalog, parameters).
* **src/anticlustering/core/**:

  * **loans/**: Loan‐data handling modules (feature encoding, streaming standardization, arrival/departure bookkeeping).
  * **vectorizer.py**: Core feature‐extraction utilities.
  * Shared abstract “Solver” base classes and streaming orchestration engine.
* **src/anticlustering/solvers/**: Implementations of all anticlustering algorithms (ILP, exchange heuristics, k-means, matching).
* **src/anticlustering/pipelines/**: Kedro pipelines for data simulation, preprocessing, offline benchmarks, and online streaming experiments.
* **src/anticlustering/streaming/**: Classes and functions for simulating and managing streaming loan events.
* **tests/**: Automated tests to validate pipeline nodes, solver correctness, and data transformations.

---

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/FormelessPuppy41/anticlustering.git
   cd anticlustering
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kedro**
   Ensure you have [Kedro](https://kedro.readthedocs.io/) installed:

   ```bash
   pip install kedro
   ```

4. **Run Offline Benchmarks & Simulation Studies**

   ```bash
   kedro run --pipeline baseline
   ```

5. **Run Online Anticlustering Experiments**
   Synthetic & Loan data:

   ```bash
   kedro run --pipeline online
   ```

---

## Configuration

* **Swap data sources** or **adjust algorithm parameters** by editing:

  * `conf/base/parameters.yaml`
  * `conf/base/parameters_anticluster.yml`
  * Any environment-specific overrides in `conf/local/`
* **Change data catalog** entries in `conf/base/catalog.yml`.

---

## Code Availability

All code for reproducing our experiments, both the original anticlustering methods and our online extensions, is available at:

> [https://github.com/FormelessPuppy41/anticlustering](https://github.com/FormelessPuppy41/anticlustering)

The GitHub repository uses the standard Kedro project layout. After cloning, users can reproduce every result by running the commands shown above. All intermediate and final datasets are stored under `data/` in CSV or Pickle (`.pkl`) format, organized into the Kedro catalog stages (`01_raw`, `02_intermediate`, `03_primary`, …, `08_reporting`). Users can swap in alternative input files or switch solvers by editing the YAML configuration under `conf/base/` or the environment-specific folder (`conf/local/`).

This modular structure, together with clear YAML-based configuration, makes it straightforward to introduce new data sources, tweak algorithm settings, or extend the framework with novel heuristics. Detailed setup instructions and dependency versions are provided in this top-level `README.md`.
