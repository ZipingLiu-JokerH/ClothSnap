"""
Utility script to convert a trained Keras model into a TensorFlow SavedModel
ready for TensorFlow Serving. Labels are hardcoded to match the training order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# pylint: disable=E0611
from tensorflow import keras

# Hardcoded class names in the expected order
CLASS_NAMES = [
    "dress",
    "hat",
    "longsleeve",
    "outwear",
    "pants",
    "shirt",
    "shoes",
    "shorts",
    "skirt",
    "t-shirt",
]


def export(model_path: Path, export_dir: Path) -> None:
    """Load a .keras model and export it as a SavedModel with labels.json."""
    model = keras.models.load_model(model_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.save(export_dir, save_format="tf")

    with open(export_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"Exported SavedModel to {export_dir}")
    print(f"Wrote labels.json with {len(CLASS_NAMES)} classes")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model export."""
    parser = argparse.ArgumentParser(
        description="Export a Keras model to TF SavedModel format for TensorFlow Serving."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the trained .keras model file (e.g., models/transfer_.../model.keras).",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("models/tf_serving_ready_model"),
        help="Directory to write the SavedModel (default: models/tf_serving_ready_model).",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to export the model based on command-line arguments."""
    args = parse_args()
    export(args.model_path, args.export_dir)


if __name__ == "__main__":
    main()
