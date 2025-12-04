"""
Utility script to convert a trained Keras model into a TensorFlow SavedModel
ready for TensorFlow Serving. Labels are hardcoded to match the training order.

Optionally bundles the export into a versioned tar.gz under ./artifacts.
"""

import argparse
import tarfile
from pathlib import Path

# pylint: disable=E0611
from tensorflow import keras


def export(model_path: Path, export_dir: Path) -> None:
    """Load a .keras model and export it as a SavedModel."""
    model = keras.models.load_model(model_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    model.export(export_dir)

    print(f"Exported SavedModel to {export_dir}")


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
        "--bundle",
        action="store_true",
        help="If set, bundle to artifacts/model.tar.gz.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to export the model based on command-line arguments."""
    args = parse_args()
    export_dir = Path("models/tf_serving_ready_model")
    export(args.model_path, export_dir)
    if args.bundle:
        bundle_path = Path("artifacts/model.tar.gz")
        bundle(bundle_path, export_dir)

if __name__ == "__main__":
    main()
