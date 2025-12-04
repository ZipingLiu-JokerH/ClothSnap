# src/evaluate.py

from pathlib import Path
from typing import Tuple
import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
# pylint: disable=E0611
from tensorflow import keras

from src.data_pipeline import build_datasets


def load_model(model_path: str | Path) -> keras.Model:
    """Load a trained Keras model from the specified path."""
    model_path = Path(model_path)
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded.")
    return model


def get_predictions(
    model: keras.Model,
    test_ds: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model on the test dataset and get their true labels and predctions.
    """
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        preds = np.argmax(preds, axis=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """
    Plot a confusion matrix for qualitative analysis.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (normalized)")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

def main() -> None:
    """Main function to evaluate the model."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Keras model on the test dataset."
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        required=True,
        help="Path to the trained model (e.g., models/transfer_20250101-123000/model.keras)",
    )

    args = parser.parse_args()

    model_path = args.model_path
    data_dir = "data"

     # Build test datasets
    _, _, test_ds, class_names = build_datasets(data_dir)

    # Load trained model
    model = load_model(model_path)

    # get predictions
    y_true, y_pred = get_predictions(model, test_ds)

    # Print classification report
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names,  digits=4))

    # Show confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()
