"""
Utility script to convert a trained Keras model into a TensorFlow SavedModel
ready for TensorFlow Serving. Labels are hardcoded to match the training order.

Optionally bundles the export into a versioned tar.gz under ./artifacts.
"""

from __future__ import annotations

import argparse
import json
import tarfile
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

    model.export(export_dir)

    with open(export_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2)

    print(f"Exported SavedModel to {export_dir}")
    print(f"Wrote labels.json with {len(CLASS_NAMES)} classes")


def bundle(bundle_path: Path, export_dir: Path) -> None:
    """Tar/gzip the export_dir."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(bundle_path, "w:gz") as tar:
        tar.add(export_dir, arcname=export_dir.name)

    print(f"Bundled model to {bundle_path}")


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
        "--bundle-version",
        type=str,
        default=None,
        help="If set, bundle to artifacts/clothsnap_<version>.tar.gz.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to export the model based on command-line arguments."""
    args = parse_args()
    export_dir = Path("models/tf_serving_ready_model")
    export(args.model_path, export_dir)
    if args.bundle_version:
        bundle_path = Path("artifacts") / f"clothsnap_{args.bundle_version}.tar.gz"
        bundle(bundle_path, export_dir)


if __name__ == "__main__":
    main()
