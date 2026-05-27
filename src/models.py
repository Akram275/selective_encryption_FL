from tensorflow.keras.applications import MobileNetV2
# --- VGG-like small model (no BatchNorm) ---
def build_vgg_small(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile_model(models.Model(inputs, outputs, name='vgg_small'))

# --- MobileNetV2-based small model (no BatchNorm) ---
def build_mobilenetv2_small(input_shape=(32, 32, 3), num_classes=10):
    # Use MobileNetV2 with alpha=0.35 for a very small model, no top, no batchnorm in head
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        alpha=0.35,
        pooling=None,
    )
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(base_model.input, outputs, name='mobilenetv2_small')
    return _compile_model(model)
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


# --- ResNet variant without BatchNorm (uses LayerNorm) ---
def _residual_block_no_bn(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.LayerNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.ReLU()(x)


def build_resnet_no_bn(input_shape=(32, 32, 3), num_classes=10):
    """ResNet-like model for CIFAR-10, but with LayerNorm instead of BatchNorm."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, padding='same', use_bias=False)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(3):
        x = _residual_block_no_bn(x, 32, stride=1)
    for block_index in range(3):
        x = _residual_block_no_bn(x, 64, stride=2 if block_index == 0 else 1)
    for block_index in range(3):
        x = _residual_block_no_bn(x, 128, stride=2 if block_index == 0 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return _compile_model(models.Model(inputs, outputs, name='resnet_no_bn'))

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
        'resnet_no_bn': build_resnet_no_bn,
        'vgg_small': build_vgg_small,
        'mobilenetv2_small': build_mobilenetv2_small,
    }

    if normalized_name not in builders:
        supported = ', '.join(sorted(builders))
        raise ValueError(f'Unsupported model name: {model_name}. Supported models: {supported}')

    return builders[normalized_name](input_shape=input_shape, num_classes=num_classes)


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Backward-compatible wrapper around the default benchmark model."""
    return build_lenet_model(input_shape=input_shape, num_classes=num_classes)
