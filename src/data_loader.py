# src/data_loader.py
import tensorflow as tf
import numpy as np

def load_cifar10():
    """Loads CIFAR-10 data, performing minimal range normalization."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def create_clients(x, y, num_clients=3, samples_per_client=200, seed=42):
    """Partitions dataset matrices across mock edge device nodes recursively."""
    np.random.seed(seed)
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    use_full_dataset = (
        samples_per_client is None
        or samples_per_client <= 0
        or samples_per_client * num_clients < len(x)
    )

    if use_full_dataset:
        partitioned_indices = np.array_split(indices, num_clients)
    else:
        partitioned_indices = []
        for i in range(num_clients):
            start = i * samples_per_client
            end = min(start + samples_per_client, len(x))
            if start >= len(x):
                break
            partitioned_indices.append(indices[start:end])

    client_data = []
    for idx in partitioned_indices:
        client_data.append((x[idx], y[idx]))

    sample_counts = [len(client_x) for client_x, _ in client_data]
    print(
        f"[Data] Allocated {len(client_data)} client partitions "
        f"with sample counts {sample_counts}."
    )
    return client_data
