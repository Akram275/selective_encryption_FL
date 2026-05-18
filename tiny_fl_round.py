import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import os
import sys

# Add current directory to path
sys.path.append(os.path.abspath('.'))

from src.models import build_cnn_model
from src.utils import get_model_weights, set_model_weights, flatten_weights, unflatten_weights
import src.crypto as crypto

def main():
    print("Starting tiny end-to-end hybrid FL round (TensorFlow version)...")
    
    # 1. Config
    num_clients = 3
    num_train_samples = 96
    num_test_samples = 64
    W = 32
    num_windows = 4 
    C = 1.0
    sigma = 0.01

    # 2. Data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train_small = x_train[:num_train_samples]
    y_train_small = y_train[:num_train_samples]
    x_test_small = x_test[:num_test_samples]
    y_test_small = y_test[:num_test_samples]
    
    client_data = []
    samples_per_client = num_train_samples // num_clients
    for i in range(num_clients):
        idx = range(i * samples_per_client, (i + 1) * samples_per_client)
        client_data.append((x_train_small[idx], y_train_small[idx]))

    # 3. Model & Context
    global_model = build_cnn_model()
    context = crypto.init_ckks_context()
    
    # Weights info
    initial_weights = get_model_weights(global_model)
    flat_weights, shapes = flatten_weights(initial_weights)
    D = len(flat_weights)
    print(f"Model flattened size: {D}")

    # 4. Local Training and Selective Encryption
    client_updates_fhe = []
    client_updates_dp = []
    
    for i in range(num_clients):
        print(f"Training client {i+1}...")
        local_model = build_cnn_model()
        set_model_weights(local_model, initial_weights)
        
        # Train 1 epoch
        cx, cy = client_data[i]
        local_model.fit(cx, cy, epochs=1, batch_size=8, verbose=0)
        
        # Calculate update
        trained_weights = get_model_weights(local_model)
        flat_trained, _ = flatten_weights(trained_weights)
        update = flat_trained - flat_weights
        
        # Selective encryption: pick top indices
        selector = np.zeros(D)
        # Using a very small fraction for speed or fixed W windows
        indices = np.random.choice(D, W * num_windows, replace=False)
        selector[indices] = 1.0
        
        fhe_win, dp_v, k_stars, mask = crypto.apply_hybrid_mechanism_multi(
            update, W, num_windows, C, sigma, selector_vector=selector
        )
        enc_win = crypto.encrypt_windowed_update(context, fhe_win)
        
        client_updates_fhe.append(enc_win)
        client_updates_dp.append(dp_v)

    # 5. Aggregation
    print("Aggregating updates...")
    agg_fhe = crypto.aggregate_fhe_updates(client_updates_fhe)
    agg_dp = np.sum(client_updates_dp, axis=0)
    
    dec_fhe = crypto.decrypt_windowed_update(context, agg_fhe)
    final_agg_update = crypto.reconstruct_full_update(dec_fhe, agg_dp, D, W, k_stars)
    
    # 6. Global Update
    new_flat_weights = flat_weights + (final_agg_update / num_clients)
    new_weights = unflatten_weights(new_flat_weights, shapes)
    set_model_weights(global_model, new_weights)

    # 7. Evaluation
    print("Evaluating...")
    loss, acc = global_model.evaluate(x_test_small, y_test_small, verbose=0)

    print(f"Round completed successfully.")
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc*100:.2f}%")

if __name__ == '__main__':
    main()
