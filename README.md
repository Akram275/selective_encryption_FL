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

The benchmark loop now uses FedSGD-style communication: each round, every client samples one local batch, computes a communicated gradient payload for that batch, the server averages those client gradients, applies the server update, and the next round starts from the updated model.

Key config knobs for this flow:

- `federated.rounds`: number of communicated FedSGD batches
- `federated.batch_size`: client batch size sampled each round
- `federated.server_learning_rate`: learning rate used when the server applies the averaged gradient
- `federated.learning_rate_decay.type`: one of `constant`, `step`, or `exponential`
- `federated.learning_rate_decay.rate`: multiplicative decay factor
- `federated.learning_rate_decay.steps`: decay interval for `step`, or smoothing divisor for `exponential`
- `federated.learning_rate_decay.min_learning_rate`: lower bound for the scheduled server learning rate
- `selective_encryption.microbatch_size`: micro-batch size used by the hybrid gradient statistics path

Example decay configuration:

```yaml
federated:
	server_learning_rate: 0.01
	learning_rate_decay:
		type: step
		rate: 0.95
		steps: 10
		min_learning_rate: 0.001
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

Generate the standard per-run plots from the latest run:

```bash
python3 metrics/plot_benchmarks.py
```

Plot a specific run:

```bash
python3 metrics/plot_benchmarks.py --run-id <run_directory_name>
```

These plots are saved under the selected run directory in `plots/` by default.

Generate a comparative convergence chart across multiple runs (with example runs) :

```bash
python3 metrics/plot_comparative_benchmarks.py \
	--run-ids \
	results/benchmarks/20260519_100822_hybrid_1/round_metrics.csv \
	results/benchmarks/20260519_101654_full_dp/round_metrics.csv \
	results/benchmarks/20260519_112339_hybrid_3_windows/round_metrics.csv \
	results/benchmarks/20260519_103700_full_fhe/round_metrics.csv
```

Generate a comparative cumulative transferred-data chart across multiple runs (with example runs):

```bash
python3 metrics/plot_comparative_transferred_data.py \
	--run-ids \
	results/benchmarks/20260519_100822_hybrid_1/round_metrics.csv \
	results/benchmarks/20260519_101654_full_dp/round_metrics.csv \
	results/benchmarks/20260519_112339_hybrid_3_windows/round_metrics.csv \
	results/benchmarks/20260519_103700_full_fhe/round_metrics.csv
```

Both comparative scripts also accept run directory names or run directory paths instead of `round_metrics.csv` paths. By default, they save outputs under `results/benchmarks/comparisons/`.

## Project Layout

- `scripts/run_benchmark.py`: main benchmark entrypoint
- `src/`: training, privacy, crypto, aggregation, metrics, and model code
- `config/`: benchmark configurations
- `metrics/plot_benchmarks.py`: single-run plotting utility for CSV benchmark outputs
- `metrics/plot_comparative_benchmarks.py`: comparative convergence plotting across runs
- `metrics/plot_comparative_transferred_data.py`: comparative cumulative transferred-data plotting across runs
- `results/`: saved benchmark runs and generated plots
