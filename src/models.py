# src/models.py
from tensorflow.keras import layers, models


def _compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def build_lenet_model(input_shape=(32, 32, 3), num_classes=10):
    """Builds a compact LeNet-style CNN for CIFAR-10 experiments."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=5, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile_model(models.Model(inputs, outputs, name='lenet_cifar'))


def build_resnet_cifar_small(input_shape=(32, 32, 3), num_classes=10):
    """Builds a CIFAR-style ResNet with about 1.2M parameters."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(3):
        x = _residual_block(x, 32, stride=1)
    for block_index in range(3):
        x = _residual_block(x, 64, stride=2 if block_index == 0 else 1)
    for block_index in range(3):
        x = _residual_block(x, 128, stride=2 if block_index == 0 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile_model(models.Model(inputs, outputs, name='resnet_cifar_small'))


def build_resnet_cifar_medium(input_shape=(32, 32, 3), num_classes=10):
    """Builds a CIFAR-style ResNet with about 4.9M parameters."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(4):
        x = _residual_block(x, 64, stride=1)
    for block_index in range(4):
        x = _residual_block(x, 128, stride=2 if block_index == 0 else 1)
    for block_index in range(4):
        x = _residual_block(x, 256, stride=2 if block_index == 0 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile_model(models.Model(inputs, outputs, name='resnet_cifar_medium'))


def build_model(model_name='lenet', input_shape=(32, 32, 3), num_classes=10):
    """Builds a supported model by name."""
    normalized_name = model_name.lower()
    builders = {
        'lenet': build_lenet_model,
        'resnet_small': build_resnet_cifar_small,
        'resnet_medium': build_resnet_cifar_medium,
    }

    if normalized_name not in builders:
        supported = ', '.join(sorted(builders))
        raise ValueError(f'Unsupported model name: {model_name}. Supported models: {supported}')

    return builders[normalized_name](input_shape=input_shape, num_classes=num_classes)


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Backward-compatible wrapper around the default benchmark model."""
    return build_lenet_model(input_shape=input_shape, num_classes=num_classes)
