import sys
import os
print(f"Python Executable: {sys.executable}")
print(f"Python Path: {sys.path}")

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
print(f"TF Version: {tf.__version__}")
try:
    import keras
    print(f"Keras Version: {keras.__version__}")
except ImportError as e:
    print(f"Keras import failed: {e}")

