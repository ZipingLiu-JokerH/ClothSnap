"""Unit tests for data pipeline functions."""
import tensorflow as tf

from src.data_pipeline import prepare_train_dataset

def test_prepare_train_dataset():
    """Test that prepare_train_dataset normalizes images to [0, 1]."""
    images = tf.ones((2, 224, 224, 3), dtype=tf.uint8) * 255
    labels = tf.constant([0, 1])
    ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)

    prepped = prepare_train_dataset(ds)
    batch_images, batch_labels = next(iter(prepped))

    min_val = tf.reduce_min(batch_images).numpy()
    max_val = tf.reduce_max(batch_images).numpy()

    assert batch_images.shape[0] == 2
    assert batch_labels.shape[0] == 2
    assert min_val >= 0.0 - 1e-6
    assert max_val <= 1.0 + 1e-6
