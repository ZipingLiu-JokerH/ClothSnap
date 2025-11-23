"""Debug TensorFlow and Keras Environment"""
import sys
import os
import tensorflow as tf

print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path}")

os.environ["KERAS_BACKEND"] = "tensorflow"

print(f"TF Version: {tf.__version__}")
try:
    import keras
    print(f"Keras Version: {keras.__version__}")
except ImportError as e:
    print(f"Keras import failed: {e}")
