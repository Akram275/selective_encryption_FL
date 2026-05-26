import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / 'results' / 'benchmarks'
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / 'comparisons'

MODE_PAYLOAD_COLUMNS = {
    'non_private': 'plaintext_size_bytes',
    'full_fhe': 'encrypted_size_bytes',
    'full_dp': 'dp_size_bytes',
    'hybrid': 'hybrid_payload_bytes',
}


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
        description='Plot comparative cumulative transferred payload data from multiple benchmark runs.'
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
        help='Directory where the comparative transfer plot will be saved.',
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


def load_run_metadata(run_dir):
    metadata_path = run_dir / 'run_metadata.csv'
    if not metadata_path.is_file():
        return {}

    rows = read_csv_rows(metadata_path)
    if not rows:
        return {}

    return rows[0]


def build_run_label(run_dir, metadata_row):
    mode = metadata_row.get('benchmark_mode', '').strip() or run_dir.name
    raw_num_windows = metadata_row.get('num_windows', '').strip()
    num_windows = int(raw_num_windows) if raw_num_windows else None

    label_parts = [mode]
    if num_windows is not None and mode == 'hybrid':
        label_parts.append(f'windows={num_windows}')

    return ' | '.join(label_parts)


def infer_payload_column(mode):
    if mode not in MODE_PAYLOAD_COLUMNS:
        raise ValueError(f'Unsupported benchmark mode for payload plotting: {mode}')
    return MODE_PAYLOAD_COLUMNS[mode]


def load_transfer_series(run_dir):
    metadata_row = load_run_metadata(run_dir)
    mode = metadata_row.get('benchmark_mode', '').strip() or run_dir.name
    payload_column = infer_payload_column(mode)

    round_metrics_path = run_dir / 'round_metrics.csv'
    if not round_metrics_path.is_file():
        raise FileNotFoundError(f'Missing round metrics CSV: {round_metrics_path}')

    rows = read_csv_rows(round_metrics_path)
    if not rows:
        raise ValueError(f'No rows found in {round_metrics_path}')

    rounds = [int(row['round']) for row in rows]
    round_payload_bytes = [float(row[payload_column]) for row in rows]

    cumulative_bytes = []
    total_bytes = 0.0
    for payload_bytes in round_payload_bytes:
        total_bytes += payload_bytes
        cumulative_bytes.append(total_bytes)

    return {
        'mode': mode,
        'label': build_run_label(run_dir, metadata_row),
        'payload_column': payload_column,
        'rounds': rounds,
        'round_payload_bytes': round_payload_bytes,
        'cumulative_bytes': cumulative_bytes,
        'final_total_bytes': total_bytes,
    }


def format_bytes(num_bytes):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    return f'{value:.2f} {units[unit_index]}'


def plot_comparative_transferred_data(run_dirs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    axis_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    series_by_run = []
    for run_index, run_dir in enumerate(run_dirs):
        series = load_transfer_series(run_dir)
        color = color_cycle[run_index % len(color_cycle)] if color_cycle else None
        cumulative_mb = [value / (1024.0 ** 2) for value in series['cumulative_bytes']]

        ax.plot(
            series['rounds'],
            cumulative_mb,
            label=series['label'],
            color=color,
            linewidth=2.3,
        )

        if series['rounds']:
            ax.annotate(
                format_bytes(series['final_total_bytes']),
                xy=(series['rounds'][-1], cumulative_mb[-1]),
                xytext=(6, 0),
                textcoords='offset points',
                fontsize=10,
                color=color,
                va='center',
            )

        series_by_run.append(series)

    ax.set_xlabel('Round', fontsize=axis_fontsize)
    ax.set_ylabel('Cumulative transferred payload (MB)', fontsize=axis_fontsize)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_fontsize)

    legend_columns = min(3, max(1, len(series_by_run)))
    ax.legend(loc='upper left', fontsize=legend_fontsize, ncol=legend_columns)
    fig.tight_layout()

    figure_path = output_dir / 'comparative_transferred_data.png'
    fig.savefig(figure_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return figure_path, series_by_run


def main():
    args = parse_args()
    results_dir = args.results_dir.resolve()
    run_dirs = resolve_run_dirs(results_dir, args.run_ids)
    figure_path, series_by_run = plot_comparative_transferred_data(run_dirs, args.output_dir.resolve())

    print('Compared runs:')
    for run_dir, series in zip(run_dirs, series_by_run):
        print(
            f'- {run_dir.name} as {series["label"]} '
            f'using {series["payload_column"]} '
            f'-> total {format_bytes(series["final_total_bytes"])}'
        )
    print(f'Saved: {figure_path}')


if __name__ == '__main__':
    main()