# scripts/run_benchmark.py
import sys
import os
import time
import argparse
import math
import yaml
import numpy as np
import tensorflow as tf

# Append root directory path safely to pick up local src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_cifar10, create_clients
from src.models import build_model
from src.crypto import init_ckks_context, encrypt_selected_update
from src.federated import scale_mixed_update, aggregate_mixed_updates, decrypt_mixed_update
from src.gradient_utils import compute_batch_gradient, compute_dp_sgd_batch_gradient, compute_hybrid_selective_batch_gradient
from src.metrics import calculate_mixed_serialized_sizes, init_benchmark_logging, log_round_metrics, log_client_metrics, plot_coordinate_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a federated-learning benchmark with a selected privacy/encryption strategy.'
    )
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(__file__), '../config/hybrid.yaml'),
        help='Path to the benchmark YAML config.',
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    if cfg is None:
        cfg = {}

    cfg.setdefault('benchmark', {})
    cfg['benchmark'].setdefault('mode', 'hybrid')

    cfg.setdefault('model', {})
    cfg['model'].setdefault('name', 'lenet')

    cfg.setdefault('federated', {})
    cfg['federated'].setdefault('rounds', 100)
    cfg['federated'].setdefault('local_epochs', 2)
    cfg['federated'].setdefault('num_clients', 3)
    cfg['federated'].setdefault('batch_size', 32)
    cfg['federated'].setdefault('samples_per_client', None)
    cfg['federated'].setdefault('server_learning_rate', 0.01)
    cfg['federated'].setdefault('learning_rate_decay', {})
    cfg['federated']['learning_rate_decay'].setdefault('type', 'constant')
    cfg['federated']['learning_rate_decay'].setdefault('rate', 1.0)
    cfg['federated']['learning_rate_decay'].setdefault('steps', 1)
    cfg['federated']['learning_rate_decay'].setdefault('min_learning_rate', 0.0)

    cfg.setdefault('crypto', {})
    cfg['crypto'].setdefault('poly_modulus_degree', 8192)
    cfg['crypto'].setdefault('scale_power', 40)

    cfg.setdefault('selective_encryption', {})
    cfg['selective_encryption'].setdefault('num_windows', 1)
    cfg['selective_encryption'].setdefault('clipping_threshold', 0.5)
    cfg['selective_encryption'].setdefault('dp_sigma', 1.0)
    cfg['selective_encryption'].setdefault('dp_learning_rate', 0.01)
    cfg['selective_encryption'].setdefault('microbatch_size', 32)
    cfg['selective_encryption'].setdefault('selection_scope', 'all')
    cfg['selective_encryption'].setdefault('stats_num_samples', 100)
    cfg['selective_encryption'].setdefault('stats_batch_size', 16)
    cfg['selective_encryption'].setdefault('coordinate_plot_frequency', 0)

    return cfg


def build_full_fhe_update(local_update):
    selected_indices = np.arange(len(local_update), dtype=int)
    mask = np.ones(len(local_update), dtype=bool)
    return np.array(local_update, dtype=np.float64), np.zeros(len(local_update), dtype=np.float64), selected_indices, mask


def build_full_dp_update(local_update, clipping_threshold, dp_sigma):
    mask = np.zeros(len(local_update), dtype=bool)
    return np.array([], dtype=np.float64), np.array(local_update, dtype=np.float64), np.array([], dtype=int), mask


def build_non_private_update(local_update):
    mask = np.zeros(len(local_update), dtype=bool)
    return np.array([], dtype=np.float64), np.array(local_update, dtype=np.float64), np.array([], dtype=int), mask


def build_hybrid_update(local_update, selected_indices):
    selected_indices = np.asarray(selected_indices, dtype=int)
    mask = np.zeros(len(local_update), dtype=bool)
    mask[selected_indices] = True

    fhe_values = np.array(local_update[selected_indices], dtype=np.float64)
    dp_vector = np.array(local_update, dtype=np.float64, copy=True)
    dp_vector[mask] = 0.0
    return fhe_values, dp_vector, selected_indices, mask


def build_client_update_payload(mode, local_update, selected_indices=None):
    if mode == 'non_private':
        return build_non_private_update(local_update)

    if mode == 'full_fhe':
        return build_full_fhe_update(local_update)

    if mode == 'full_dp':
        return build_full_dp_update(local_update, clipping_threshold=0.0, dp_sigma=0.0)

    if mode == 'hybrid':
        if selected_indices is None:
            raise ValueError('hybrid payload requires selected encrypted indices')
        return build_hybrid_update(local_update, selected_indices)

    raise ValueError(f'Unsupported benchmark mode: {mode}')


def compute_dp_privacy(clipping_threshold, dp_sigma, delta=1e-5):
    if clipping_threshold <= 0 or dp_sigma <= 0:
        return None, delta

    noise_std = dp_sigma
    # Gaussian mechanism privacy guarantee with absolute noise std.
    epsilon = math.sqrt(2.0 * math.log(1.25 / delta)) * clipping_threshold / noise_std
    return epsilon, delta


def print_mode_summary(mode, plain_b, enc_b, dp_b, hybrid_b, fhe_coordinates, total_coordinates, num_ciphertexts, selective_cfg):
    if mode == 'non_private':
        payload_b = plain_b
        payload_label = 'Non-private client payload'
    elif mode == 'full_fhe':
        payload_b = enc_b
        payload_label = 'Full FHE client payload'
    elif mode == 'full_dp':
        payload_b = dp_b
        payload_label = 'Full DP client payload'
    else:
        payload_b = hybrid_b
        payload_label = 'Hybrid client payload'

    expansion_factor = 0.0 if plain_b == 0 else payload_b / plain_b
    ciphertext_text = '' if num_ciphertexts == 0 else f', {num_ciphertexts} ciphertexts'

    print(
        f"    [Metrics] Plaintext model size: {plain_b / 1024:.2f} KB "
        f"({total_coordinates} parameters) | "
        f"{payload_label}: {payload_b / 1024:.2f} KB "
        f"({expansion_factor:.2f}x{ciphertext_text})"
    )

    if mode in {'full_dp', 'hybrid'}:
        epsilon, delta = compute_dp_privacy(
            selective_cfg['clipping_threshold'],
            selective_cfg['dp_sigma'],
        )
        if epsilon is None:
            print(f"    [DP] epsilon=undefined, delta={delta:.0e}")
        else:
            print(f"    [DP] epsilon={epsilon:.4f}, delta={delta:.0e}")


def initialize_client_batch_states(clients, seed=42):
    """This function initializes per-client batch sampling states for FedSGD, including random permutations and cursors for each client's dataset. It returns a shared RNG and a list of batch states corresponding to each client."""
    rng = np.random.default_rng(seed)
    batch_states = []
    for client_x, _ in clients:
        if len(client_x) == 0:
            raise ValueError('client dataset cannot be empty for FedSGD')
        batch_states.append(
            {
                'permutation': rng.permutation(len(client_x)),
                'cursor': 0,
            }
        )
    return rng, batch_states


def sample_client_batch(client_x, client_y, batch_size, batch_state, rng):
    client_size = len(client_x)
    effective_batch_size = max(1, min(int(batch_size), client_size))

    if batch_state['cursor'] + effective_batch_size > client_size:
        batch_state['permutation'] = rng.permutation(client_size)
        batch_state['cursor'] = 0

    batch_indices = batch_state['permutation'][
        batch_state['cursor'] : batch_state['cursor'] + effective_batch_size
    ]
    batch_state['cursor'] += effective_batch_size
    return client_x[batch_indices], client_y[batch_indices]


def unflatten_trainable_vector(flat_vector, trainable_variables):
    gradient_tensors = []
    offset = 0
    for variable in trainable_variables:
        variable_size = int(np.prod(variable.shape))
        next_offset = offset + variable_size
        gradient_slice = flat_vector[offset:next_offset]
        gradient_tensors.append(
            tf.reshape(tf.convert_to_tensor(gradient_slice, dtype=tf.float32), variable.shape)
        )
        offset = next_offset
    return gradient_tensors


def apply_server_gradient(model, flat_gradient, learning_rate):
    """
    Apply the aggregated gradient to the model using Adam optimizer if provided, otherwise SGD.
    If model.optimizer is set and has 'apply_gradients', use it (e.g., Adam or other Keras optimizer).
    Otherwise, fall back to manual SGD update.
    """
    gradient_tensors = unflatten_trainable_vector(flat_gradient, model.trainable_variables)
    # Try to use model.optimizer if available and has apply_gradients
    optimizer = getattr(model, 'optimizer', None)
    if optimizer is not None and hasattr(optimizer, 'apply_gradients'):
        optimizer.apply_gradients(zip(gradient_tensors, model.trainable_variables))
    else:
        for variable, gradient in zip(model.trainable_variables, gradient_tensors):
            variable.assign_sub(tf.cast(learning_rate, variable.dtype) * tf.cast(gradient, variable.dtype))


def compute_server_learning_rate(base_learning_rate, round_index, decay_cfg):
    decay_type = str(decay_cfg.get('type', 'constant')).lower()
    decay_rate = float(decay_cfg.get('rate', 1.0))
    decay_steps = max(1, int(decay_cfg.get('steps', 1)))
    min_learning_rate = max(0.0, float(decay_cfg.get('min_learning_rate', 0.0)))

    if decay_type == 'constant':
        learning_rate = base_learning_rate
    elif decay_type == 'step':
        learning_rate = base_learning_rate * (decay_rate ** (round_index // decay_steps))
    elif decay_type == 'exponential':
        learning_rate = base_learning_rate * (decay_rate ** (round_index / decay_steps))
    else:
        raise ValueError(f'Unsupported learning rate decay type: {decay_type}')

    return max(min_learning_rate, float(learning_rate))


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    cfg = load_config(config_path)
    benchmark_cfg = cfg.get('benchmark', {})
    benchmark_mode = benchmark_cfg.get('mode', 'hybrid')
    selective_cfg = cfg['selective_encryption']
    local_batch_size = cfg['federated'].get('batch_size', 32)
    server_learning_rate = cfg['federated'].get(
        'server_learning_rate',
        selective_cfg.get('dp_learning_rate', 0.01),
    )
    lr_decay_cfg = cfg['federated'].get('learning_rate_decay', {})
    coordinate_plot_frequency = selective_cfg.get('coordinate_plot_frequency', 0)

    print(f"[Init] Setting up {benchmark_mode} benchmark from {config_path}...")
    print(f"[Init] Using model architecture: {cfg['model']['name']}")
    context = init_ckks_context(
        poly_mod_degree=cfg['crypto']['poly_modulus_degree'],
        scale_power=cfg['crypto']['scale_power']
    )

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    clients = create_clients(
        x_train, y_train,
        num_clients=cfg['federated']['num_clients'],
        samples_per_client=cfg['federated']['samples_per_client']
    )
    client_test_sets = create_clients(
        x_test, y_test,
        num_clients=cfg['federated']['num_clients'],
        samples_per_client=None,
    )
    client_sizes = [len(client_x) for client_x, _ in clients]
    total_client_samples = sum(client_sizes)
    results_dir = os.path.join(os.path.dirname(__file__), '../results/benchmarks')
    benchmark_log = init_benchmark_logging(results_dir, cfg, client_sizes, config_path=config_path)
    coordinate_plot_dir = os.path.join(benchmark_log['run_dir'], 'coordinate_stats')
    print(f"[Metrics] Saving benchmark CSVs under {benchmark_log['run_dir']}")

    global_model = build_model(cfg['model']['name'])
    _, client_batch_states = initialize_client_batch_states(clients)

    # Main Federated Loop Simulation
    for r in range(cfg['federated']['rounds']):
        current_server_learning_rate = compute_server_learning_rate(
            server_learning_rate,
            r,
            lr_decay_cfg,
        )
        print(f"\n================ COMMUNICATION ROUND {r+1}/{cfg['federated']['rounds']} ================")
        print(f"[Round] Server learning rate: {current_server_learning_rate:.6f}")
        round_start = time.perf_counter()
        client_updates = []
        round_plaintext_bytes = 0
        round_encrypted_bytes = 0
        round_dp_bytes = 0
        round_hybrid_bytes = 0
        fhe_coordinate_counts = []
        dp_coordinate_counts = []

        for i in range(len(clients)):
            print(f"  Training loop invocation for node participant #{i+1}...")
            local_model = build_model(cfg['model']['name'])
            local_model.set_weights(global_model.get_weights())
            print(f"    [Client {i+1}] Loaded global model state with {client_sizes[i]} local samples.")

            x_batch, y_batch = sample_client_batch(
                clients[i][0],
                clients[i][1],
                local_batch_size,
                client_batch_states[i],
                _,
            )
            print(f"    [Client {i+1}] Sampled batch of {len(x_batch)} examples for FedSGD.")

            gradient_stats = None
            selected_indices = np.array([], dtype=int)

            # One-batch client gradient computation
            client_train_start = time.perf_counter()
            if benchmark_mode == 'full_dp':
                print(f"    [Client {i+1}] Computing strict DP-SGD batch gradient...")
                gradient_result = compute_dp_sgd_batch_gradient(
                    local_model,
                    x_batch,
                    y_batch,
                    clipping_norm=selective_cfg['clipping_threshold'],
                    noise_std=selective_cfg['dp_sigma'],
                    learning_rate=selective_cfg['dp_learning_rate'],
                )
            elif benchmark_mode == 'hybrid':
                print(
                    f"    [Client {i+1}] Computing hybrid selective-DP batch gradient "
                    f"with micro-batch statistics..."
                )
                gradient_result = compute_hybrid_selective_batch_gradient(
                    local_model,
                    x_batch,
                    y_batch,
                    clipping_norm=selective_cfg['clipping_threshold'],
                    noise_std=selective_cfg['dp_sigma'],
                    num_windows=selective_cfg['num_windows'],
                    poly_mod_degree=cfg['crypto']['poly_modulus_degree'],
                    microbatch_size=selective_cfg['microbatch_size'],
                )
                gradient_stats = {
                    'num_samples': gradient_result['num_samples'],
                    'mean': gradient_result['mean'],
                    'mean_abs': gradient_result['mean_abs'],
                    'variance': gradient_result['variance'],
                }
                selected_indices = gradient_result['selected_indices']
                print(
                    f"    [Client {i+1}] Tracked {len(gradient_result['batch_windows'])} batch window selections, "
                    f"covering {len(selected_indices)} unique encrypted coordinates."
                )
            else:
                print(f"    [Client {i+1}] Computing plaintext batch gradient...")
                gradient_result = compute_batch_gradient(
                    local_model,
                    x_batch,
                    y_batch,
                )
            client_train_time = time.perf_counter() - client_train_start
            print(f"    [Client {i+1}] Batch gradient ready in {client_train_time:.2f}s.")

            client_gradient = gradient_result['gradient']
            fhe_values, dp_vector, selected_indices, mask = build_client_update_payload(
                benchmark_mode,
                client_gradient,
                selected_indices=selected_indices,
            )
            encrypted_payloads = encrypt_selected_update(
                context,
                selected_indices,
                fhe_values,
                poly_mod_degree=cfg['crypto']['poly_modulus_degree'],
            )
            print(
                f"    [Client {i+1}] Prepared payload with {len(selected_indices)} encrypted coordinates "
                f"and {len(mask) - int(mask.sum())} plaintext/DP coordinates."
            )

            plain_b, enc_b, dp_b, hybrid_b = calculate_mixed_serialized_sizes(
                encrypted_payloads,
                dp_vector,
                client_gradient,
            )
            round_plaintext_bytes += plain_b
            round_encrypted_bytes += enc_b
            round_dp_bytes += dp_b
            round_hybrid_bytes += hybrid_b

            fhe_coordinates = int(mask.sum())
            dp_coordinates = int(len(mask) - fhe_coordinates)
            fhe_coordinate_counts.append(fhe_coordinates)
            dp_coordinate_counts.append(dp_coordinates)

            if i == 0:
                num_ciphertexts = sum(len(payload['chunks']) for payload in encrypted_payloads)
                print_mode_summary(
                    benchmark_mode,
                    plain_b,
                    enc_b,
                    dp_b,
                    hybrid_b,
                    fhe_coordinates,
                    len(mask),
                    num_ciphertexts,
                    selective_cfg,
                )

            if benchmark_mode == 'hybrid' and coordinate_plot_frequency and (r + 1) % coordinate_plot_frequency == 0:
                plot_coordinate_stats(
                    gradient_stats['mean_abs'],
                    gradient_stats['variance'],
                    i,
                    r,
                    output_dir=coordinate_plot_dir,
                    selected_indices=selected_indices,
                )

            client_updates.append(
                {
                    'encrypted_payloads': encrypted_payloads,
                    'dp_vector': dp_vector,
                    'vector_length': len(client_gradient),
                }
            )

            log_client_metrics(
                benchmark_log['client_metrics_path'],
                {
                    'run_id': benchmark_log['run_id'],
                    'round': r + 1,
                    'client_index': i + 1,
                    'num_samples': client_sizes[i],
                    'train_time_sec': round(client_train_time, 6),
                    'encrypted_chunks': sum(len(payload['chunks']) for payload in encrypted_payloads),
                    'encrypted_size_bytes': enc_b,
                    'dp_size_bytes': dp_b,
                    'hybrid_payload_bytes': hybrid_b,
                    'plaintext_size_bytes': plain_b,
                    'fhe_coordinates': fhe_coordinates,
                    'dp_coordinates': dp_coordinates,
                    'selected_indices': '|'.join(str(idx) for idx in selected_indices),
                    'gradient_sample_count': 0 if gradient_stats is None else gradient_stats['num_samples'],
                    'gradient_mean_abs_p95': 0.0 if gradient_stats is None else round(float(np.percentile(gradient_stats['mean_abs'], 95)), 6),
                    'gradient_variance_p95': 0.0 if gradient_stats is None else round(float(np.percentile(gradient_stats['variance'], 95)), 6),
                }
            )

        # Global Server Aggregation
        print("  Aggregating updates at centralized global orchestrator server node...")
        aggregation_start = time.perf_counter()
        scaled_updates = [
            scale_mixed_update(update, 1.0 / len(client_updates))
            for update in client_updates
        ]
        aggregated_payloads, aggregated_dp = aggregate_mixed_updates(scaled_updates)
        aggregation_time = time.perf_counter() - aggregation_start

        # Update global model parameters from the averaged communicated gradient.
        aggregated_gradient = decrypt_mixed_update(
            aggregated_payloads,
            aggregated_dp,
            len(client_updates[0]['dp_vector']),
        )
        apply_server_gradient(global_model, aggregated_gradient, current_server_learning_rate)

        # Evaluate performance accuracy metrics against unseen holdout dataset matrices
        evaluation_start = time.perf_counter()
        loss, acc = global_model.evaluate(x_test, y_test, verbose=0)
        evaluation_time = time.perf_counter() - evaluation_start
        round_time = time.perf_counter() - round_start

        log_round_metrics(
            benchmark_log['round_metrics_path'],
            {
                'run_id': benchmark_log['run_id'],
                'round': r + 1,
                'server_learning_rate': round(float(current_server_learning_rate), 10),
                'accuracy': round(float(acc), 6),
                'loss': round(float(loss), 6),
                'round_time_sec': round(round_time, 6),
                'aggregation_time_sec': round(aggregation_time, 6),
                'evaluation_time_sec': round(evaluation_time, 6),
                'plaintext_size_bytes': round_plaintext_bytes,
                'encrypted_size_bytes': round_encrypted_bytes,
                'dp_size_bytes': round_dp_bytes,
                'hybrid_payload_bytes': round_hybrid_bytes,
                'mean_fhe_coordinates': round(float(np.mean(fhe_coordinate_counts)), 6),
                'mean_dp_coordinates': round(float(np.mean(dp_coordinate_counts)), 6),
                'mean_fhe_fraction': round(
                    float(np.mean(np.array(fhe_coordinate_counts) / len(client_updates[0]['dp_vector']))),
                    6,
                ),
                'encrypted_payload_count': len(aggregated_payloads),
            }
        )

        print(f"  -> Global Metrics Round Evaluation results: Acc={acc:.4f}, Loss={loss:.4f}")

if __name__ == '__main__':
    main()
