"""Module providing functionality to build tf.data pipelines."""

# tf.data loader + preprocessing

import tensorflow as tf

print(tf.__version__)

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Convert an image tensor to float32 and scale pixel values to [0, 1].
    This will be reused later in the real input pipeline.
    """
    image = tf.cast(image, tf.float32)
    return image / 255.0
