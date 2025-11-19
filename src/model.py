"""
Module providing model architectures for image classification.
"""

from typing import Tuple

# pylint: disable=E0611
from tensorflow import keras

def build_baseline_cnn(
    input_shape: Tuple[int, int, int],
    num_classes: int,
) -> keras.Model:
    """
    Simple baseline CNN for image classification.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")
    return model


def build_transfer_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    train_base: bool = False,
) -> keras.Model:
    """
    Transfer learning model using MobileNetV2 as a feature extractor.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = train_base  # False for feature extraction, True for fine-tuning

    inputs = keras.Input(shape=input_shape, name="input_image")

    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mobilenetv2_transfer")
    return model
