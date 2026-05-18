# selective_encryption_FL

This repository contains the benchmark code for the selective encryption federated learning experiments.

## Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Benchmark Configs

The benchmark entrypoint is `scripts/run_benchmark.py`.

Available configs:

- `config/non_private.yaml`
- `config/full_fhe.yaml`
- `config/full_dp.yaml`
- `config/hybrid.yaml`

Each config controls the benchmark mode, federated learning settings, CKKS parameters, and selective-encryption settings.

## Run Benchmarks

Run the benchmark with a chosen config:

```bash
python3 scripts/run_benchmark.py --config config/hybrid.yaml
```

Examples:

```bash
python3 scripts/run_benchmark.py --config config/non_private.yaml
python3 scripts/run_benchmark.py --config config/full_fhe.yaml
python3 scripts/run_benchmark.py --config config/full_dp.yaml
python3 scripts/run_benchmark.py --config config/hybrid.yaml
```

Benchmark outputs are written under `results/benchmarks/` in timestamped run directories.

## Plot Saved Results

Generate plots from the latest run:

```bash
python3 metrics/plot_benchmarks.py
```

Plot a specific run:

```bash
python3 metrics/plot_benchmarks.py --run-id <run_directory_name>
```

Plots are saved under the selected run directory in `plots/` by default.

## Project Layout

- `scripts/run_benchmark.py`: main benchmark entrypoint
- `src/`: training, privacy, crypto, aggregation, metrics, and model code
- `config/`: benchmark configurations
- `metrics/plot_benchmarks.py`: plotting utility for CSV benchmark outputs
- `results/`: saved benchmark runs and generated plots
