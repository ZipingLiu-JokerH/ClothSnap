'''Unit tests for data pipeline functions.'''

import tensorflow as tf

from src.data_pipeline import normalize_image


def test_normalize_image_scales_to_zero_one():
    """Test that normalize_image scales pixel values to [0, 1]."""
    # Create a tiny fake "image" with known pixel values
    image = tf.constant([[0, 128, 255]], dtype=tf.uint8)

    normalized = normalize_image(image)

    # Convert to numpy for easier comparison
    arr = normalized.numpy()

    # Check that all values are between 0 and 1, inclusive
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0

    # Optional: check specific scaling for sanity
    assert arr[0, 0] == 0.0          # 0 / 255
    assert arr[0, 2] == 1.0          # 255 / 255
