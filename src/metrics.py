# src/metrics.py
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from datetime import datetime

def calculate_serialized_sizes(encrypted_chunks, raw_weights_np):
    """Evaluates payload footprint metrics footprint discrepancies overhead numbers."""
    import pickle
    plain_bytes = len(pickle.dumps(raw_weights_np))
    enc_bytes = sum(len(chunk.serialize()) for chunk in encrypted_chunks)
    return plain_bytes, enc_bytes

def calculate_mixed_serialized_sizes(encrypted_payloads, dp_vector, raw_weights_np):
    """Measures hybrid selective-encryption payload sizes against a plaintext baseline."""
    import pickle

    plain_bytes = len(pickle.dumps(raw_weights_np))
    encrypted_bytes = sum(
        len(chunk.serialize())
        for payload in encrypted_payloads
        for chunk in payload['chunks']
    )
    dp_bytes = len(pickle.dumps(np.asarray(dp_vector, dtype=np.float64)))
    hybrid_bytes = encrypted_bytes + dp_bytes
    return plain_bytes, encrypted_bytes, dp_bytes, hybrid_bytes

def init_benchmark_logging(base_dir, cfg, client_sizes, config_path=None):
    """Creates timestamped CSV files for benchmark runs and writes static metadata."""
    benchmark_mode = cfg.get("benchmark", {}).get("mode", "hybrid")
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{benchmark_mode}"
    if benchmark_mode == 'hybrid':
        num_windows = cfg.get("selective_encryption", {}).get("num_windows", 0)
        run_id = f"{run_id}_{num_windows}_windows"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    run_metadata_path = os.path.join(run_dir, "run_metadata.csv")
    round_metrics_path = os.path.join(run_dir, "round_metrics.csv")
    client_metrics_path = os.path.join(run_dir, "client_metrics.csv")

    _write_csv_row(
        run_metadata_path,
        [
            "run_id",
            "benchmark_mode",
            "training_framework",
            "config_path",
            "rounds",
            "local_epochs",
            "num_clients",
            "client_batch_size",
            "samples_per_client",
            "total_client_samples",
            "client_sizes",
            "server_learning_rate",
            "server_lr_decay_type",
            "server_lr_decay_rate",
            "server_lr_decay_steps",
            "server_lr_min_learning_rate",
            "poly_modulus_degree",
            "scale_power",
            "slots_per_ciphertext",
            "num_windows",
            "clipping_threshold",
            "dp_sigma",
            "stats_num_samples",
            "stats_batch_size",
        ],
        {
            "run_id": run_id,
            "benchmark_mode": benchmark_mode,
            "training_framework": "fedsgd",
            "config_path": config_path or "",
            "rounds": cfg["federated"]["rounds"],
            "local_epochs": cfg["federated"]["local_epochs"],
            "num_clients": cfg["federated"]["num_clients"],
            "client_batch_size": cfg["federated"].get("batch_size", 32),
            "samples_per_client": cfg["federated"]["samples_per_client"],
            "total_client_samples": sum(client_sizes),
            "client_sizes": "|".join(str(size) for size in client_sizes),
            "server_learning_rate": cfg["federated"].get("server_learning_rate", 0.01),
            "server_lr_decay_type": cfg["federated"].get("learning_rate_decay", {}).get("type", "constant"),
            "server_lr_decay_rate": cfg["federated"].get("learning_rate_decay", {}).get("rate", 1.0),
            "server_lr_decay_steps": cfg["federated"].get("learning_rate_decay", {}).get("steps", 1),
            "server_lr_min_learning_rate": cfg["federated"].get("learning_rate_decay", {}).get("min_learning_rate", 0.0),
            "poly_modulus_degree": cfg["crypto"]["poly_modulus_degree"],
            "scale_power": cfg["crypto"]["scale_power"],
            "slots_per_ciphertext": cfg["crypto"]["poly_modulus_degree"] // 2,
            "num_windows": cfg["selective_encryption"]["num_windows"],
            "clipping_threshold": cfg["selective_encryption"]["clipping_threshold"],
            "dp_sigma": cfg["selective_encryption"]["dp_sigma"],
            "stats_num_samples": cfg["selective_encryption"].get("stats_num_samples", 64),
            "stats_batch_size": cfg["selective_encryption"].get("stats_batch_size", 16),
        },
    )

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "round_metrics_path": round_metrics_path,
        "client_metrics_path": client_metrics_path,
    }

def log_round_metrics(csv_path, metrics_row):
    """Appends one per-round metrics row for later plotting."""
    _write_csv_row(
        csv_path,
        [
            "run_id",
            "round",
            "server_learning_rate",
            "accuracy",
            "loss",
            "round_time_sec",
            "aggregation_time_sec",
            "evaluation_time_sec",
            "plaintext_size_bytes",
            "encrypted_size_bytes",
            "dp_size_bytes",
            "hybrid_payload_bytes",
            "mean_fhe_coordinates",
            "mean_dp_coordinates",
            "mean_fhe_fraction",
            "encrypted_payload_count",
        ],
        metrics_row,
    )

def log_client_metrics(csv_path, metrics_row):
    """Appends one client-per-round metrics row for local training analysis."""
    _write_csv_row(
        csv_path,
        [
            "run_id",
            "round",
            "client_index",
            "num_samples",
            "train_time_sec",
            "encrypted_chunks",
            "encrypted_size_bytes",
            "dp_size_bytes",
            "hybrid_payload_bytes",
            "plaintext_size_bytes",
            "fhe_coordinates",
            "dp_coordinates",
            "selected_indices",
            "gradient_sample_count",
            "gradient_mean_abs_p95",
            "gradient_variance_p95",
        ],
        metrics_row,
    )

def _write_csv_row(csv_path, fieldnames, row):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def plot_coordinate_stats(means, variances, client_idx, round_idx, output_dir='.', selected_indices=None):
    """Saves profiling scatter chart distributions highlighting DP vs FHE split choices."""
    num_coords = len(means)
    sample_size = min(int(num_coords * 0.1), 5000)
    indices = np.random.choice(num_coords, sample_size, replace=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(np.abs(means[indices]), variances[indices], s=10, alpha=0.5, color='royalblue')
    if selected_indices is not None and len(selected_indices) > 0:
        highlighted = np.array(selected_indices, dtype=int)
        highlighted = highlighted[highlighted < num_coords]
        if len(highlighted) > 0:
            plt.scatter(
                np.abs(means[highlighted]),
                variances[highlighted],
                s=14,
                alpha=0.8,
                color='crimson',
                label='Selected FHE coordinates',
            )

    plt.xscale('log')
    plt.yscale('log')
    plt.title(f"Sensitivity Profile: Client {client_idx+1} - Round {round_idx+1}")
    plt.xlabel("Absolute Mean Gradient Magnitude (log scale)")
    plt.ylabel("Gradient Variance (log scale)")
    plt.axvline(x=1e-3, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, which="both", alpha=0.2)
    if selected_indices is not None and len(selected_indices) > 0:
        plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"sensitivity_client_{client_idx}_round_{round_idx}.png"))
    plt.close()
