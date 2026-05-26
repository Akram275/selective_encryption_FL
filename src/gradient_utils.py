import numpy as np
import tensorflow as tf

from src.crypto import select_top_coordinate_indices, select_top_coordinate_indices_tensor


def get_trainable_variable_layout(model):
    """Returns flattened coordinate ranges for each trainable variable."""
    layout = []
    offset = 0
    for variable_index, variable in enumerate(model.trainable_variables):
        size = int(np.prod(variable.shape))
        layout.append(
            {
                'variable_index': variable_index,
                'name': variable.name,
                'start': offset,
                'end': offset + size,
                'size': size,
            }
        )
        offset += size
    return layout


def get_candidate_coordinate_indices(model, selection_scope='all'):
    """Returns global flattened coordinate indices eligible for encryption selection."""
    layout = get_trainable_variable_layout(model)
    if selection_scope in {None, 'all'}:
        return np.arange(layout[-1]['end'] if layout else 0, dtype=int)

    dense_layers = []
    for item in layout:
        layer_name = item['name'].split('/')[0]
        if 'dense' in layer_name:
            dense_layers.append(layer_name)

    dense_layers = list(dict.fromkeys(dense_layers))
    if not dense_layers:
        return np.arange(layout[-1]['end'] if layout else 0, dtype=int)

    if selection_scope == 'last_dense':
        selected_layer_names = dense_layers[-1:]
    elif selection_scope == 'last_two_dense':
        selected_layer_names = dense_layers[-2:]
    else:
        raise ValueError(f'Unsupported selection scope: {selection_scope}')

    selected_ranges = []
    for item in layout:
        layer_name = item['name'].split('/')[0]
        if layer_name in selected_layer_names:
            selected_ranges.append(np.arange(item['start'], item['end'], dtype=int))

    if not selected_ranges:
        return np.arange(layout[-1]['end'] if layout else 0, dtype=int)
    return np.concatenate(selected_ranges)


def _compute_per_example_gradient_tensor(model, x_batch, y_batch, loss_fn, variable_indices=None):
    with tf.GradientTape(persistent=True) as tape:
        predictions = model(x_batch, training=True)
        per_example_losses = loss_fn(y_batch, predictions)

    if variable_indices is None:
        variable_indices = range(len(model.trainable_variables))

    flattened_grads = []
    for variable_index in variable_indices:
        variable = model.trainable_variables[variable_index]
        per_example_grad = tape.jacobian(
            per_example_losses,
            variable,
            experimental_use_pfor=False,
        )
        flattened_grads.append(tf.reshape(per_example_grad, [tf.shape(x_batch)[0], -1]))

    del tape
    return tf.cast(tf.concat(flattened_grads, axis=1), tf.float32)


def _compute_per_example_gradient_matrix(model, x_batch, y_batch, loss_fn):
    return _compute_per_example_gradient_tensor(model, x_batch, y_batch, loss_fn).numpy().astype(np.float64)


def _flatten_gradient_list(gradients, trainable_variables):
    flattened_grads = []
    for gradient, variable in zip(gradients, trainable_variables):
        if gradient is None:
            gradient = tf.zeros_like(variable, dtype=tf.float32)
        flattened_grads.append(tf.reshape(tf.cast(gradient, tf.float32), [-1]))
    return tf.concat(flattened_grads, axis=0)


def _compute_mean_gradient_tensor(model, x_batch, y_batch, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        per_example_losses = loss_fn(y_batch, predictions)
        mean_loss = tf.reduce_mean(per_example_losses)

    gradients = tape.gradient(mean_loss, model.trainable_variables)
    return _flatten_gradient_list(gradients, model.trainable_variables)


def _compute_microbatch_gradient_tensor(model, x_batch, y_batch, loss_fn, microbatch_size):
    batch_len = x_batch.shape[0]
    if batch_len is None:
        batch_len = int(tf.shape(x_batch)[0].numpy())

    effective_microbatch_size = max(1, min(int(microbatch_size), int(batch_len)))
    microbatch_grads = []

    for start in range(0, batch_len, effective_microbatch_size):
        end = min(start + effective_microbatch_size, batch_len)
        microbatch_grads.append(
            _compute_mean_gradient_tensor(
                model,
                x_batch[start:end],
                y_batch[start:end],
                loss_fn,
            )
        )

    return tf.stack(microbatch_grads, axis=0)


def _prepare_batch_tensors(x_batch, y_batch):
    return (
        tf.cast(x_batch, tf.float32),
        tf.reshape(tf.cast(y_batch, tf.int64), [-1]),
    )


def _summarize_gradient_tensor(gradient_tensor):
    gradient_tensor = tf.cast(gradient_tensor, tf.float64)
    grad_mean = tf.reduce_mean(gradient_tensor, axis=0)
    grad_mean_abs = tf.reduce_mean(tf.abs(gradient_tensor), axis=0)
    grad_var = tf.maximum(
        tf.reduce_mean(tf.square(gradient_tensor), axis=0) - tf.square(grad_mean),
        tf.constant(0.0, dtype=tf.float64),
    )
    return {
        'num_samples': int(gradient_tensor.shape[0]),
        'mean': grad_mean.numpy(),
        'mean_abs': grad_mean_abs.numpy(),
        'variance': grad_var.numpy(),
    }


def _flatten_trainable_variable_values(model):
    flattened_values = []
    for variable in model.trainable_variables:
        flattened_values.append(tf.reshape(tf.cast(variable, tf.float64), [-1]))
    if not flattened_values:
        return np.array([], dtype=np.float64)
    return tf.concat(flattened_values, axis=0).numpy()


def compute_batch_gradient(model, x_batch, y_batch):
    """Computes the mean gradient for a single client batch."""
    x_batch, y_batch = _prepare_batch_tensors(x_batch, y_batch)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    mean_gradient = _compute_mean_gradient_tensor(model, x_batch, y_batch, loss_fn)
    return {
        'gradient': tf.cast(mean_gradient, tf.float64).numpy(),
        'selected_indices': np.array([], dtype=int),
        'batch_windows': [],
        **_summarize_gradient_tensor(tf.expand_dims(mean_gradient, axis=0)),
    }


def compute_dp_sgd_batch_gradient(
    model,
    x_batch,
    y_batch,
    clipping_norm,
    noise_std,
    learning_rate=0.01,
):
    """Computes one strict DP-SGD client gradient from a single optimizer step."""
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive for DP-SGD")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for DP-SGD")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive for DP-SGD")

    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
            VectorizedDPKerasSGDOptimizer,
        )
    except ImportError as exc:
        raise ImportError(
            "tensorflow-privacy is required for strict per-example DP-SGD training"
        ) from exc

    x_batch, y_batch = _prepare_batch_tensors(x_batch, y_batch)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    noise_multiplier = noise_std / clipping_norm
    optimizer = VectorizedDPKerasSGDOptimizer(
        l2_norm_clip=clipping_norm,
        noise_multiplier=noise_multiplier,
        num_microbatches=None,
        learning_rate=learning_rate,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy'],
    )

    before_update = _flatten_trainable_variable_values(model)
    model.train_on_batch(x_batch, y_batch)
    after_update = _flatten_trainable_variable_values(model)
    mean_gradient = (before_update - after_update) / learning_rate

    return {
        'gradient': np.asarray(mean_gradient, dtype=np.float64),
        'selected_indices': np.array([], dtype=int),
        'batch_windows': [],
        **_summarize_gradient_tensor(tf.expand_dims(tf.convert_to_tensor(mean_gradient, dtype=tf.float64), axis=0)),
    }


def compute_hybrid_selective_batch_gradient(
    model,
    x_batch,
    y_batch,
    clipping_norm,
    noise_std,
    num_windows,
    poly_mod_degree,
    microbatch_size,
):
    """Computes one hybrid selective-DP client gradient from a single sampled batch."""
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive for hybrid DP training")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for hybrid DP training")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive for hybrid DP training")

    x_batch, y_batch = _prepare_batch_tensors(x_batch, y_batch)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    microbatch_grads = _compute_microbatch_gradient_tensor(
        model,
        x_batch,
        y_batch,
        loss_fn,
        microbatch_size=microbatch_size,
    )
    total_params = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    all_indices_tf = tf.range(total_params, dtype=tf.int32)

    microbatch_abs = tf.abs(microbatch_grads)
    batch_mean_abs = tf.reduce_mean(microbatch_abs, axis=0)
    batch_selected_indices_tf = select_top_coordinate_indices_tensor(
        batch_mean_abs,
        num_windows=num_windows,
        poly_mod_degree=poly_mod_degree,
    )
    batch_selected_indices = batch_selected_indices_tf.numpy().astype(int)

    selected_mask_tf = tf.scatter_nd(
        tf.expand_dims(batch_selected_indices_tf, axis=1),
        tf.ones_like(batch_selected_indices_tf, dtype=tf.bool),
        [total_params],
    )
    dp_indices_tf = tf.boolean_mask(all_indices_tf, tf.logical_not(selected_mask_tf))

    combined_grad = tf.reduce_mean(microbatch_grads, axis=0)

    if tf.size(dp_indices_tf) > 0:
        dp_batch_grads = tf.gather(microbatch_grads, dp_indices_tf, axis=1)
        l2_norms = tf.norm(dp_batch_grads, axis=1, keepdims=True)
        scaling = tf.minimum(
            1.0,
            clipping_norm / tf.maximum(l2_norms, tf.constant(1e-12, dtype=tf.float32)),
        )
        clipped_dp_grads = dp_batch_grads * scaling
        noisy_dp_grad = tf.reduce_mean(clipped_dp_grads, axis=0)
        if noise_std > 0:
            noisy_dp_grad += tf.random.normal(
                shape=tf.shape(noisy_dp_grad),
                mean=0.0,
                stddev=noise_std,
                dtype=tf.float32,
            )
        combined_grad = tf.tensor_scatter_nd_update(
            combined_grad,
            tf.expand_dims(dp_indices_tf, axis=1),
            noisy_dp_grad,
        )

    if tf.size(batch_selected_indices_tf) > 0:
        fhe_grad = tf.reduce_mean(
            tf.gather(microbatch_grads, batch_selected_indices_tf, axis=1),
            axis=0,
        )
        combined_grad = tf.tensor_scatter_nd_update(
            combined_grad,
            tf.expand_dims(batch_selected_indices_tf, axis=1),
            fhe_grad,
        )

    return {
        'gradient': tf.cast(combined_grad, tf.float64).numpy(),
        'selected_indices': batch_selected_indices,
        'batch_windows': [batch_selected_indices],
        **_summarize_gradient_tensor(microbatch_grads),
    }


def estimate_gradient_statistics(model, x, y, max_samples=64, batch_size=16, seed=42, selection_scope='all'):
    """Estimates per-coordinate gradient statistics from a bounded client-side sample."""
    if len(x) == 0:
        raise ValueError("cannot estimate gradient statistics from an empty client dataset")

    candidate_coordinate_indices = get_candidate_coordinate_indices(model, selection_scope)
    layout = get_trainable_variable_layout(model)
    candidate_variable_indices = [
        item['variable_index']
        for item in layout
        if np.any((candidate_coordinate_indices >= item['start']) & (candidate_coordinate_indices < item['end']))
    ]

    sample_count = min(len(x), max_samples if max_samples is not None else len(x))
    batch_size = max(1, min(batch_size, sample_count))

    if sample_count < len(x):
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(x), size=sample_count, replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
    else:
        x_sample = x
        y_sample = y

    if sample_count > batch_size:
        effective_sample_count = sample_count - (sample_count % batch_size)
        if effective_sample_count > 0:
            x_sample = x_sample[:effective_sample_count]
            y_sample = y_sample[:effective_sample_count]
            sample_count = effective_sample_count

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

    grad_sum = None
    grad_abs_sum = None
    grad_sq_sum = None
    observed_samples = 0

    dataset = tf.data.Dataset.from_tensor_slices((x_sample, y_sample)).batch(
        batch_size,
        drop_remainder=sample_count >= batch_size,
    ).prefetch(tf.data.AUTOTUNE)

    for x_batch, y_batch in dataset:
        x_batch = tf.cast(x_batch, tf.float32)
        y_batch = tf.reshape(tf.cast(y_batch, tf.int64), [-1])
        batch_grads = _compute_per_example_gradient_tensor(
            model,
            x_batch,
            y_batch,
            loss_fn,
            variable_indices=candidate_variable_indices,
        ).numpy().astype(np.float64)
        batch_abs = np.abs(batch_grads)

        if grad_sum is None:
            grad_sum = batch_grads.sum(axis=0)
            grad_abs_sum = batch_abs.sum(axis=0)
            grad_sq_sum = np.square(batch_grads).sum(axis=0)
        else:
            grad_sum += batch_grads.sum(axis=0)
            grad_abs_sum += batch_abs.sum(axis=0)
            grad_sq_sum += np.square(batch_grads).sum(axis=0)

        observed_samples += batch_grads.shape[0]

    grad_mean = grad_sum / observed_samples
    grad_mean_abs = grad_abs_sum / observed_samples
    grad_var = np.maximum(grad_sq_sum / observed_samples - np.square(grad_mean), 0.0)

    return {
        'num_samples': observed_samples,
        'mean': grad_mean,
        'mean_abs': grad_mean_abs,
        'variance': grad_var,
        'coordinate_indices': candidate_coordinate_indices,
    }


def train_with_dp_sgd(
    model,
    x,
    y,
    epochs,
    batch_size,
    clipping_norm,
    noise_std,
    learning_rate=0.01,
):
    """Trains a model with TensorFlow Privacy DP-SGD using per-example clipping."""
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras_vectorized import (
            VectorizedDPKerasSGDOptimizer,
        )
    except ImportError as exc:
        raise ImportError(
            "tensorflow-privacy is required for strict per-example DP-SGD training"
        ) from exc

    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive for DP-SGD")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for DP-SGD")

    noise_multiplier = noise_std / clipping_norm
    optimizer = VectorizedDPKerasSGDOptimizer(
        l2_norm_clip=clipping_norm,
        noise_multiplier=noise_multiplier,
        num_microbatches=None,
        learning_rate=learning_rate,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
    )
    return model.fit(
        x,
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )


def train_with_hybrid_selective_dp(
    model,
    x,
    y,
    epochs,
    batch_size,
    clipping_norm,
    noise_std,
    num_windows,
    poly_mod_degree,
    microbatch_size,
    learning_rate=0.01,
):
    """Trains with micro-batch gradient heuristics for hybrid selective DP."""
    if clipping_norm <= 0:
        raise ValueError("clipping_norm must be positive for hybrid DP training")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative for hybrid DP training")
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive for hybrid DP training")

    total_params = int(sum(np.prod(variable.shape) for variable in model.trainable_variables))
    tracked_selected_mask = tf.zeros([total_params], dtype=tf.bool)
    all_indices_tf = tf.range(total_params, dtype=tf.int32)

    variable_sizes = [int(np.prod(variable.shape)) for variable in model.trainable_variables]
    variable_shapes = [tuple(variable.shape) for variable in model.trainable_variables]

    optimizer = tf.keras.optimizers.get(tf.keras.optimizers.serialize(model.optimizer))
    optimizer.learning_rate = learning_rate
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )

    grad_sum = None
    grad_abs_sum = None
    grad_sq_sum = None
    observed_samples = 0
    batch_windows = []

    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(
        batch_size,
        drop_remainder=len(x) >= batch_size,
    ).prefetch(tf.data.AUTOTUNE)

    for _ in range(epochs):
        for x_batch, y_batch in dataset:
            x_batch = tf.cast(x_batch, tf.float32)
            y_batch = tf.reshape(tf.cast(y_batch, tf.int64), [-1])
            microbatch_grads = _compute_microbatch_gradient_tensor(
                model,
                x_batch,
                y_batch,
                loss_fn,
                microbatch_size=microbatch_size,
            )
            microbatch_abs = tf.abs(microbatch_grads)
            batch_mean_abs = tf.reduce_mean(microbatch_abs, axis=0)
            batch_selected_indices_tf = select_top_coordinate_indices_tensor(
                batch_mean_abs,
                num_windows=num_windows,
                poly_mod_degree=poly_mod_degree,
            )
            batch_selected_indices = batch_selected_indices_tf.numpy().astype(int)
            if batch_selected_indices.size > 0:
                tracked_selected_mask = tf.tensor_scatter_nd_update(
                    tracked_selected_mask,
                    tf.expand_dims(batch_selected_indices_tf, axis=1),
                    tf.ones_like(batch_selected_indices_tf, dtype=tf.bool),
                )
            batch_windows.append(batch_selected_indices)

            batch_sum = tf.reduce_sum(microbatch_grads, axis=0)
            batch_abs_sum = tf.reduce_sum(microbatch_abs, axis=0)
            batch_sq_sum = tf.reduce_sum(tf.square(microbatch_grads), axis=0)
            if grad_sum is None:
                grad_sum = tf.cast(batch_sum, tf.float64)
                grad_abs_sum = tf.cast(batch_abs_sum, tf.float64)
                grad_sq_sum = tf.cast(batch_sq_sum, tf.float64)
            else:
                grad_sum += tf.cast(batch_sum, tf.float64)
                grad_abs_sum += tf.cast(batch_abs_sum, tf.float64)
                grad_sq_sum += tf.cast(batch_sq_sum, tf.float64)
            observed_samples += int(microbatch_grads.shape[0])

            selected_mask_tf = tf.scatter_nd(
                tf.expand_dims(batch_selected_indices_tf, axis=1),
                tf.ones_like(batch_selected_indices_tf, dtype=tf.bool),
                [total_params],
            )
            dp_indices_tf = tf.boolean_mask(all_indices_tf, tf.logical_not(selected_mask_tf))
            combined_grad = tf.reduce_mean(microbatch_grads, axis=0)

            if tf.size(dp_indices_tf) > 0:
                dp_batch_grads = tf.gather(microbatch_grads, dp_indices_tf, axis=1)
                l2_norms = tf.norm(dp_batch_grads, axis=1, keepdims=True)
                scaling = tf.minimum(
                    1.0,
                    clipping_norm / tf.maximum(l2_norms, tf.constant(1e-12, dtype=tf.float32)),
                )
                clipped_dp_grads = dp_batch_grads * scaling
                noisy_dp_grad = tf.reduce_mean(clipped_dp_grads, axis=0)
                if noise_std > 0:
                    noisy_dp_grad += tf.random.normal(
                        shape=tf.shape(noisy_dp_grad),
                        mean=0.0,
                        stddev=noise_std,
                        dtype=tf.float32,
                    )
                combined_grad = tf.tensor_scatter_nd_update(
                    combined_grad,
                    tf.expand_dims(dp_indices_tf, axis=1),
                    noisy_dp_grad,
                )

            if tf.size(batch_selected_indices_tf) > 0:
                fhe_grad = tf.reduce_mean(
                    tf.gather(microbatch_grads, batch_selected_indices_tf, axis=1),
                    axis=0,
                )
                combined_grad = tf.tensor_scatter_nd_update(
                    combined_grad,
                    tf.expand_dims(batch_selected_indices_tf, axis=1),
                    fhe_grad,
                )

            gradient_tensors = []
            offset = 0
            for variable_shape, variable_size in zip(variable_shapes, variable_sizes):
                next_offset = offset + variable_size
                gradient_tensors.append(
                    tf.reshape(combined_grad[offset:next_offset], variable_shape)
                )
                offset = next_offset

            optimizer.apply_gradients(zip(gradient_tensors, model.trainable_variables))

    tracked_selected_indices = tf.cast(tf.reshape(tf.where(tracked_selected_mask), [-1]), tf.int32).numpy()
    observed_samples_value = float(observed_samples)
    grad_mean = (grad_sum / observed_samples_value).numpy()
    grad_mean_abs = (grad_abs_sum / observed_samples_value).numpy()
    grad_var = tf.maximum(
        grad_sq_sum / observed_samples_value - tf.square(grad_sum / observed_samples_value),
        tf.constant(0.0, dtype=tf.float64),
    ).numpy()

    return {
        'model': model,
        'selected_indices': tracked_selected_indices,
        'batch_windows': batch_windows,
        'num_samples': observed_samples,
        'mean': grad_mean,
        'mean_abs': grad_mean_abs,
        'variance': grad_var,
    }