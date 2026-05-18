# src/models.py
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """Builds a compact LeNet-style CNN for CIFAR-10 experiments."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(6, kernel_size=5, padding='same', activation='relu')(inputs)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, kernel_size=5, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(120, activation='relu')(x)
    x = layers.Dense(84, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
