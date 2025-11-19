"""
Module providing functionality to build tf.data pipelines.
"""

from pathlib import Path
from typing import Tuple

import tensorflow as tf
# pylint: disable=E0611
from tensorflow import keras

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image tensor to float32 and scale pixel values to [0, 1].
    This will be reused later in the real input pipeline.
    """
    image = tf.cast(image, tf.float32)
    return image / 255.0


AUTOTUNE = tf.data.AUTOTUNE

normalization_layer = keras.layers.Rescaling(1.0 / 255.0)

data_augmentation_layer = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomZoom(0.1),
    ],
    name="data_augmentation_layer",
)


def _build_raw_dataset(
    split_dir: Path,
    img_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    return keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def prepare_train_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply data augmentation and normalization to training dataset.
    """

    ds = ds.map(
        lambda x, y: (data_augmentation_layer(normalization_layer(x)), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache().prefetch(AUTOTUNE)
    return ds


def prepare_test_eval_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """
    Apply only normalization to test and evaluation dataset.
    """

    ds = ds.map(
        lambda x, y: (normalization_layer(x), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.cache().prefetch(AUTOTUNE)
    return ds


def build_datasets(
    data_dir: str | Path,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    """
    Build train/val/test datasets without relying on global variables.
    """

    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    val_dir = data_dir / "validation"

    # Load raw datasets (resize + batching handled here)
    train_ds_raw = _build_raw_dataset(train_dir, img_size, batch_size, shuffle=True)
    val_ds_raw = _build_raw_dataset(val_dir, img_size, batch_size, shuffle=False)
    test_ds_raw = _build_raw_dataset(test_dir, img_size, batch_size, shuffle=False)

    cloth_class_names = train_ds_raw.class_names

    # Apply preprocessing
    train_dataset = prepare_train_dataset(train_ds_raw)
    val_dataset = prepare_test_eval_dataset(val_ds_raw)
    test_dataset = prepare_test_eval_dataset(test_ds_raw)

    return train_dataset, val_dataset, test_dataset, cloth_class_names

if __name__ == "__main__":
    train_ds, val_ds, test_ds, class_names = build_datasets("data")
    print("Classes:", class_names)

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print(
            "Pixel range:",
            float(tf.reduce_min(images)),
            "→",
            float(tf.reduce_max(images)),
        )
