import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot benchmark CSV outputs and save PNG files for later analysis."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing timestamped benchmark runs.",
    )
    parser.add_argument(
        "--run-id",
        help="Specific benchmark run directory name to plot. Defaults to the latest run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory to save the generated plots. Defaults to <run_dir>/plots.",
    )
    return parser.parse_args()


def resolve_run_dir(results_dir, run_id=None):
    if run_id:
        run_dir = results_dir / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Benchmark run '{run_id}' was not found in {results_dir}.")
        return run_dir

    run_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No benchmark runs were found in {results_dir}.")
    return run_dirs[-1]


def read_csv_rows(csv_path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def plot_round_metrics(round_rows, output_dir):
    rounds = [int(row["round"]) for row in round_rows]
    accuracy = [float(row["accuracy"]) for row in round_rows]
    loss = [float(row["loss"]) for row in round_rows]
    round_time = [float(row["round_time_sec"]) for row in round_rows]
    aggregation_time = [float(row["aggregation_time_sec"]) for row in round_rows]
    evaluation_time = [float(row["evaluation_time_sec"]) for row in round_rows]
    plaintext_kb = [float(row["plaintext_size_bytes"]) / 1024.0 for row in round_rows]
    encrypted_kb = [float(row["encrypted_size_bytes"]) / 1024.0 for row in round_rows]
    dp_kb = [float(row.get("dp_size_bytes", 0.0)) / 1024.0 for row in round_rows]
    hybrid_kb = [float(row.get("hybrid_payload_bytes", 0.0)) / 1024.0 for row in round_rows]
    fhe_fraction = [float(row.get("mean_fhe_fraction", 0.0)) for row in round_rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(rounds, accuracy, marker="o", color="tab:blue")
    axes[0, 0].set_title("Accuracy by Round")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(rounds, loss, marker="o", color="tab:red")
    axes[0, 1].set_title("Loss by Round")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(rounds, round_time, marker="o", label="Round", color="tab:green")
    axes[1, 0].plot(rounds, aggregation_time, marker="s", label="Aggregation", color="tab:orange")
    axes[1, 0].plot(rounds, evaluation_time, marker="^", label="Evaluation", color="tab:purple")
    axes[1, 0].set_title("Timing by Round")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(rounds, plaintext_kb, marker="o", label="Plaintext baseline", color="tab:gray")
    axes[1, 1].plot(rounds, encrypted_kb, marker="s", label="Encrypted slice", color="tab:brown")
    axes[1, 1].plot(rounds, dp_kb, marker="^", label="DP slice", color="tab:olive")
    axes[1, 1].plot(rounds, hybrid_kb, marker="d", label="Hybrid total", color="tab:cyan")
    axes[1, 1].set_title("Payload Size by Round")
    axes[1, 1].set_xlabel("Round")
    axes[1, 1].set_ylabel("Size (KB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    ax_fraction = axes[0, 0].twinx()
    ax_fraction.plot(rounds, fhe_fraction, marker="x", linestyle="--", color="tab:orange", label="FHE fraction")
    ax_fraction.set_ylabel("FHE Fraction")

    accuracy_handles, accuracy_labels = axes[0, 0].get_legend_handles_labels()
    fraction_handles, fraction_labels = ax_fraction.get_legend_handles_labels()
    if fraction_handles:
        axes[0, 0].legend(accuracy_handles + fraction_handles, accuracy_labels + fraction_labels, loc="best")

    fig.tight_layout()
    figure_path = output_dir / "round_metrics.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def plot_client_metrics(client_rows, output_dir):
    client_ids = sorted({int(row["client_index"]) for row in client_rows})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for client_id in client_ids:
        rows = [row for row in client_rows if int(row["client_index"]) == client_id]
        rounds = [int(row["round"]) for row in rows]
        train_time = [float(row["train_time_sec"]) for row in rows]
        fhe_coordinates = [int(row.get("fhe_coordinates", 0)) for row in rows]
        hybrid_kb = [float(row.get("hybrid_payload_bytes", 0.0)) / 1024.0 for row in rows]

        axes[0].plot(rounds, train_time, marker="o", label=f"Client {client_id}")
        axes[1].plot(rounds, fhe_coordinates, marker="o", label=f"Client {client_id} FHE coords")
        axes[1].plot(rounds, hybrid_kb, marker="s", linestyle="--", label=f"Client {client_id} payload KB")

    axes[0].set_title("Client Training Time")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Seconds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Selective Payload by Client")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Coordinates / KB")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    figure_path = output_dir / "client_metrics.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    run_dir = resolve_run_dir(results_dir, args.run_id)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    round_metrics_path = run_dir / "round_metrics.csv"
    client_metrics_path = run_dir / "client_metrics.csv"

    if not round_metrics_path.is_file():
        raise FileNotFoundError(f"Missing round metrics CSV: {round_metrics_path}")
    if not client_metrics_path.is_file():
        raise FileNotFoundError(f"Missing client metrics CSV: {client_metrics_path}")

    round_rows = read_csv_rows(round_metrics_path)
    client_rows = read_csv_rows(client_metrics_path)
    if not round_rows:
        raise ValueError(f"No rows found in {round_metrics_path}")
    if not client_rows:
        raise ValueError(f"No rows found in {client_metrics_path}")

    round_plot = plot_round_metrics(round_rows, output_dir)
    client_plot = plot_client_metrics(client_rows, output_dir)

    print(f"Plotted benchmark run: {run_dir.name}")
    print(f"Saved: {round_plot}")
    print(f"Saved: {client_plot}")


if __name__ == "__main__":
    main()
