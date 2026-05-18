# src/crypto.py
import tenseal as ts
import numpy as np
import tensorflow as tf

def init_ckks_context(poly_mod_degree=8192, bit_sizes=[60, 40, 40, 60], scale_power=40):
    """Initializes and builds the TenSEAL Context for CKKS Scheme computations."""
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=bit_sizes
    )
    context.generate_galois_keys()
    context.global_scale = 2 ** scale_power
    return context

def flatten_weights(weights):
    """Flattens raw weights matrices structure arrays out cleanly."""
    flat_weights = []
    shapes = []
    for layer_weight in weights:
        shapes.append(layer_weight.shape)
        flat_weights.extend(layer_weight.flatten().astype(np.float64))
    return np.array(flat_weights, dtype=np.float64), shapes

def unflatten_weights(flat_weights_1d, shapes):
    """Rebuilds initial multi-dimensional matrices layouts from 1D layers."""
    reconstructed_weights = []
    current_idx = 0
    for shape in shapes:
        layer_size = np.prod(shape)
        layer_weights_flat = flat_weights_1d[current_idx : current_idx + layer_size]
        reconstructed_weights.append(layer_weights_flat.reshape(shape))
        current_idx += layer_size
    return reconstructed_weights

def encrypt_weights(context, weights_list_1d, poly_mod_degree=8192):
    """Slices long parameters arrays down into packed homomorphic chunks."""
    n_slots = poly_mod_degree // 2
    original_length = len(weights_list_1d)
    encrypted_chunks = []

    for i in range(0, original_length, n_slots):
        chunk = weights_list_1d[i : i + n_slots]
        if len(chunk) < n_slots:
            chunk = np.pad(chunk, (0, n_slots - len(chunk)), 'constant')

        enc_vector = ts.ckks_vector(context, chunk.tolist(), scale=context.global_scale)
        encrypted_chunks.append(enc_vector)

    return encrypted_chunks, original_length

def decrypt_weights(encrypted_vector_list, original_length):
    """Decrypts secure payload arrays elements reconstructing global structures."""
    decrypted_flat_weights = []
    for chunk in encrypted_vector_list:
        decrypted_flat_weights.extend(chunk.decrypt())
    return np.array(decrypted_flat_weights, dtype=np.float64)[:original_length]

def build_selection_mask(length, selected_indices):
    """Builds a boolean selection mask from arbitrary selected coordinates."""
    mask = np.zeros(length, dtype=bool)
    if len(selected_indices) > 0:
        mask[np.asarray(selected_indices, dtype=int)] = True
    return mask


def _window_sums_numpy(abs_g, window_size):
    if len(abs_g) == 0:
        return np.array([], dtype=np.float64)
    if len(abs_g) <= window_size:
        return np.array([float(np.sum(abs_g))], dtype=np.float64)

    prefix = np.concatenate(([0.0], np.cumsum(abs_g, dtype=np.float64)))
    return prefix[window_size:] - prefix[:-window_size]


def _select_top_window_starts_numpy(abs_g, num_windows, window_size):
    if num_windows <= 0 or len(abs_g) == 0:
        return np.array([], dtype=int)
    if len(abs_g) <= window_size:
        return np.array([0], dtype=int)

    remaining_scores = _window_sums_numpy(abs_g, window_size)
    selected_starts = []

    for _ in range(min(num_windows, len(remaining_scores))):
        best_start = int(np.argmax(remaining_scores))
        if not np.isfinite(remaining_scores[best_start]):
            break
        selected_starts.append(best_start)

        overlap_start = max(0, best_start - window_size + 1)
        overlap_end = min(len(remaining_scores), best_start + window_size)
        remaining_scores[overlap_start:overlap_end] = -np.inf

    return np.sort(np.asarray(selected_starts, dtype=int))


def _window_starts_to_indices(window_starts, total_length, window_size):
    selected_indices = []
    for start in np.asarray(window_starts, dtype=int):
        end = min(start + window_size, total_length)
        selected_indices.extend(range(start, end))
    if not selected_indices:
        return np.array([], dtype=int)
    return np.asarray(sorted(set(selected_indices)), dtype=int)


def select_top_coordinate_indices(g, num_windows, poly_mod_degree):
    """Selects contiguous windows with the largest absolute-magnitude sums."""
    if num_windows <= 0 or len(g) == 0:
        return np.array([], dtype=int)

    slots_per_ciphertext = poly_mod_degree // 2
    if slots_per_ciphertext <= 0:
        raise ValueError("poly modulus degree must provide at least one CKKS slot")

    if num_windows * slots_per_ciphertext >= len(g):
        return np.arange(len(g), dtype=int)

    abs_g = np.abs(np.asarray(g, dtype=np.float64))
    window_starts = _select_top_window_starts_numpy(
        abs_g,
        num_windows=num_windows,
        window_size=slots_per_ciphertext,
    )
    return _window_starts_to_indices(window_starts, len(abs_g), slots_per_ciphertext)


def select_top_coordinate_indices_tensor(g, num_windows, poly_mod_degree):
    """TensorFlow variant of contiguous highest-sum window selection."""
    if num_windows <= 0:
        return tf.zeros([0], dtype=tf.int32)

    slots_per_ciphertext = poly_mod_degree // 2
    if slots_per_ciphertext <= 0:
        raise ValueError("poly modulus degree must provide at least one CKKS slot")

    g = tf.reshape(tf.convert_to_tensor(g, dtype=tf.float32), [-1])
    total_coords = g.shape[0]
    if total_coords is None:
        total_coords = int(tf.shape(g)[0])

    if total_coords == 0:
        return tf.zeros([0], dtype=tf.int32)

    if num_windows * slots_per_ciphertext >= total_coords:
        return tf.range(total_coords, dtype=tf.int32)

    abs_g = tf.abs(g).numpy().astype(np.float64)
    window_starts = _select_top_window_starts_numpy(
        abs_g,
        num_windows=num_windows,
        window_size=slots_per_ciphertext,
    )
    selected_indices = _window_starts_to_indices(window_starts, total_coords, slots_per_ciphertext)
    return tf.convert_to_tensor(selected_indices, dtype=tf.int32)

def apply_hybrid_mechanism_multi(g, num_windows, poly_mod_degree, C, sigma, selector_vector=None):
    """Encrypts top coordinates and applies DP to the remaining coordinates."""
    D = len(g)
    selection_source = np.asarray(g if selector_vector is None else selector_vector, dtype=np.float64)
    if len(selection_source) != D:
        raise ValueError("selector vector must match update length")

    selected_indices = select_top_coordinate_indices(selection_source, num_windows, poly_mod_degree)
    mask = build_selection_mask(D, selected_indices)
    g_fhe = np.array(g[selected_indices], dtype=np.float64)
    g_dp = np.zeros(D, dtype=np.float64)

    clipped = np.clip(g, -C, C)
    noise = np.random.normal(loc=0.0, scale=sigma, size=D)
    g_dp[~mask] = clipped[~mask] + noise[~mask]

    return g_fhe, g_dp, selected_indices, mask

def encrypt_selected_update(context, selected_indices, selected_values, poly_mod_degree=8192):
    """Encrypts selected coordinates as a sparse payload with explicit indices."""
    if len(selected_indices) == 0:
        return []

    encrypted_chunks, original_length = encrypt_weights(
        context,
        np.asarray(selected_values, dtype=np.float64),
        poly_mod_degree=poly_mod_degree,
    )
    return [
        {
            'indices': np.asarray(selected_indices, dtype=int),
            'chunks': encrypted_chunks,
            'length': original_length,
        }
    ]

def merge_mixed_update(dp_vector, decrypted_payloads, vector_length):
    """Reassembles a full update vector from plaintext-DP and sparse encrypted coordinates."""
    merged = np.array(dp_vector, dtype=np.float64, copy=True)
    if len(merged) != vector_length:
        raise ValueError("dp vector length does not match expected vector length")

    for payload in decrypted_payloads:
        indices = np.asarray(payload['indices'], dtype=int)
        values = np.asarray(payload['values'], dtype=np.float64)
        merged[indices] += values[: len(indices)]

    return merged
