# src/federated.py
import tensorflow as tf
import numpy as np

from src.crypto import decrypt_weights, merge_mixed_update

def scale_plaintext_weights(weight, scalar):
    """Scales plaintext parameter layers arrays directly using standard tensor operators."""
    return [scalar * w for w in weight]

def sum_scaled_plaintext_weights(scaled_weight_list):
    """Performs standard vanilla sum parameters updates aggregation (FedAvg server step)."""
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def scale_encrypted_weights(encrypted_weight_chunk_list, scalar):
    """Homomorphic encrypted payload element arrays computation multiplication scaler step."""
    return [chunk * scalar for chunk in encrypted_weight_chunk_list]

def sum_scaled_encrypted_weights(scaled_encrypted_weight_chunk_lists):
    """Aggregates multiple clients' scaled, encrypted parameter updates via homomorphic addition."""
    if not scaled_encrypted_weight_chunk_lists or not scaled_encrypted_weight_chunk_lists[0]:
        return []

    num_chunks = len(scaled_encrypted_weight_chunk_lists[0])
    aggregated_chunks = []

    for chunk_idx in range(num_chunks):
        current_chunk_sum = scaled_encrypted_weight_chunk_lists[0][chunk_idx]
        for client_idx in range(1, len(scaled_encrypted_weight_chunk_lists)):
            current_chunk_sum = current_chunk_sum + scaled_encrypted_weight_chunk_lists[client_idx][chunk_idx]
        aggregated_chunks.append(current_chunk_sum)

    return aggregated_chunks

def scale_mixed_update(client_update, scalar):
    """Scales a mixed selective-encryption client update for weighted averaging."""
    scaled_payloads = []
    for payload in client_update['encrypted_payloads']:
        scaled_payloads.append(
            {
                'indices': np.asarray(payload['indices'], dtype=int),
                'chunks': [chunk * scalar for chunk in payload['chunks']],
                'length': payload['length'],
            }
        )

    return {
        'encrypted_payloads': scaled_payloads,
        'dp_vector': np.asarray(client_update['dp_vector'], dtype=np.float64) * scalar,
        'vector_length': client_update['vector_length'],
    }

def aggregate_mixed_updates(scaled_client_updates):
    """Aggregates sparse encrypted payloads and plaintext DP coordinates together."""
    if not scaled_client_updates:
        return [], np.array([], dtype=np.float64)

    vector_length = scaled_client_updates[0]['vector_length']
    aggregated_dp = np.zeros(vector_length, dtype=np.float64)
    aggregated_payloads = []

    for client_update in scaled_client_updates:
        aggregated_dp += client_update['dp_vector']
        aggregated_payloads.extend(client_update['encrypted_payloads'])

    return aggregated_payloads, aggregated_dp

def decrypt_mixed_update(aggregated_payloads, aggregated_dp, vector_length):
    """Decrypts aggregated sparse payloads and merges them with the plaintext DP contribution."""
    decrypted_payloads = []
    for payload in aggregated_payloads:
        decrypted_payloads.append(
            {
                'indices': np.asarray(payload['indices'], dtype=int),
                'values': decrypt_weights(payload['chunks'], payload['length']),
            }
        )

    return merge_mixed_update(aggregated_dp, decrypted_payloads, vector_length)
