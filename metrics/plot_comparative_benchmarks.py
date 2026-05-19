import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / 'results' / 'benchmarks'
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / 'comparisons'


def configure_csv_field_limit():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size //= 10


configure_csv_field_limit()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot comparative convergence curves from multiple benchmark runs.'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help='Directory containing timestamped benchmark run folders.',
    )
    parser.add_argument(
        '--run-ids',
        nargs='*',
        help='Run directory names, run directory paths, or round_metrics.csv paths to compare. Defaults to all runs in the results directory.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Directory where the comparative plot will be saved.',
    )
    return parser.parse_args()


def read_csv_rows(csv_path):
    with csv_path.open(newline='') as csv_file:
        return list(csv.DictReader(csv_file))


def resolve_run_dirs(results_dir, run_ids=None):
    if run_ids:
        run_dirs = []
        for run_id in run_ids:
            candidate = Path(run_id)
            if not candidate.is_absolute() and not candidate.exists():
                candidate = results_dir / run_id

            if candidate.name == 'round_metrics.csv':
                candidate = candidate.parent

            run_dirs.append(candidate)
    else:
        run_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())

    if not run_dirs:
        raise FileNotFoundError(f'No benchmark runs were found in {results_dir}.')

    missing = [run_dir.name for run_dir in run_dirs if not run_dir.is_dir()]
    if missing:
        missing_str = ', '.join(missing)
        raise FileNotFoundError(f'Benchmark run(s) not found: {missing_str}')

    return run_dirs


def build_run_label(run_dir):
    metadata_path = run_dir / 'run_metadata.csv'
    client_metrics_path = run_dir / 'client_metrics.csv'

    mode = run_dir.name
    num_windows = None
    if metadata_path.is_file():
        metadata_rows = read_csv_rows(metadata_path)
        if metadata_rows:
            metadata_row = metadata_rows[0]
            metadata_mode = metadata_row.get('benchmark_mode', '').strip()
            if metadata_mode:
                mode = metadata_mode
            raw_num_windows = metadata_row.get('num_windows', '').strip()
            if raw_num_windows:
                num_windows = int(raw_num_windows)

    avg_ciphertexts = None
    if client_metrics_path.is_file():
        client_rows = read_csv_rows(client_metrics_path)
        encrypted_chunks = [
            float(row.get('encrypted_chunks', 0.0))
            for row in client_rows
            if row.get('encrypted_chunks', '') != ''
        ]
        if encrypted_chunks:
            avg_ciphertexts = sum(encrypted_chunks) / len(encrypted_chunks)

    label_parts = [mode]
    if num_windows is not None:
        label_parts.append(f'windows={num_windows}')
    if avg_ciphertexts is not None:
        label_parts.append(f'ctxt/client={avg_ciphertexts:.1f}')

    return ' | '.join(label_parts)


def load_round_series(run_dir):
    round_metrics_path = run_dir / 'round_metrics.csv'
    if not round_metrics_path.is_file():
        raise FileNotFoundError(f'Missing round metrics CSV: {round_metrics_path}')

    rows = read_csv_rows(round_metrics_path)
    if not rows:
        raise ValueError(f'No rows found in {round_metrics_path}')

    return {
        'rounds': [int(row['round']) for row in rows],
        'accuracy': [float(row['accuracy']) for row in rows],
        'loss': [float(row['loss']) for row in rows],
    }


def plot_comparative_convergence(run_dirs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharex=False)
    title_fontsize = 17
    axis_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    markers = ['o', 's', '^', 'd', 'x', 'P', 'v', '*']

    labels = []
    for run_index, run_dir in enumerate(run_dirs):
        label = build_run_label(run_dir)
        series = load_round_series(run_dir)
        color = color_cycle[run_index % len(color_cycle)] if color_cycle else None
        marker = markers[run_index % len(markers)]

        axes[0].plot(
            series['rounds'],
            series['accuracy'],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )
        axes[1].plot(
            series['rounds'],
            series['loss'],
            label=label,
            color=color,
            marker=marker,
            linewidth=2.2,
            markersize=6,
        )
        labels.append(label)

    axes[0].set_title('Comparative Accuracy Convergence', fontsize=title_fontsize)
    axes[0].set_xlabel('Round', fontsize=axis_fontsize)
    axes[0].set_ylabel('Accuracy', fontsize=axis_fontsize)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)

    axes[1].set_title('Comparative Loss Convergence', fontsize=title_fontsize)
    axes[1].set_xlabel('Round', fontsize=axis_fontsize)
    axes[1].set_ylabel('Loss', fontsize=axis_fontsize)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_columns = min(3, max(1, len(run_dirs)))
    fig.legend(
        handles,
        legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=legend_columns,
        frameon=False,
        fontsize=legend_fontsize,
    )
    fig.suptitle('Benchmark Convergence Comparison', y=0.98, fontsize=18)
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.94])

    figure_path = output_dir / 'comparative_convergence.png'
    fig.savefig(figure_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return figure_path, labels


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    run_dirs = resolve_run_dirs(results_dir, args.run_ids)
    figure_path, labels = plot_comparative_convergence(run_dirs, args.output_dir.resolve())

    print('Compared runs:')
    for run_dir, label in zip(run_dirs, labels):
        print(f'- {run_dir.name} as {label}')
    print(f'Saved: {figure_path}')


if __name__ == '__main__':
    main()