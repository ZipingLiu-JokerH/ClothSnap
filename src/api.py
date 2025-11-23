"""
Flask API for serving predictions via TensorFlow Serving.

Expectations:
- TF Serving running (see docker-compose) at CLOTHSNAP_TF_URL (default below).
- Exported model and labels.json at models/tf_serving_ready_model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import requests
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# Configuration
DEFAULT_TF_URL = "http://localhost:8501/v1/models/clothsnap:predict"
INPUT_SIZE: Tuple[int, int] = (224, 224)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
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

app = Flask(__name__)


def preprocess_image(file_storage) -> np.ndarray:
    """Load image file, convert to RGB, resize, and scale to [0,1]."""
    image = Image.open(file_storage.stream).convert("RGB")
    image = image.resize(INPUT_SIZE)
    arr = np.array(image).astype("float32") / 255.0
    return arr


def postprocess_predictions(preds: np.ndarray, top_k: int = 3):
    """Return top-k label/confidence pairs."""
    top_indices = preds.argsort()[::-1][:top_k]
    results = []
    for idx in top_indices:
        label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
        results.append({"label": label, "confidence": float(preds[idx])})
    return results


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "labels_loaded": len(CLASS_NAMES)}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests."""
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        image_arr = preprocess_image(file_storage)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Failed to process image: {exc}"}), 400

    payload = {"instances": [image_arr.tolist()]}
    tf_url = os.getenv("CLOTHSNAP_TF_URL", DEFAULT_TF_URL)

    try:
        resp = requests.post(tf_url, json=payload, timeout=10)
        resp.raise_for_status()
        predictions = resp.json().get("predictions")
        if not predictions:
            return jsonify({"error": "No predictions returned from TF Serving"}), 502
    except requests.RequestException as exc:
        return jsonify({"error": f"TF Serving request failed: {exc}"}), 502

    pred_array = np.array(predictions[0], dtype="float32")
    results = postprocess_predictions(pred_array)
    return jsonify({"results": results}), 200


@app.route("/", methods=["GET"])
def serve_index():
    """Serve the frontend page."""
    return send_from_directory(str(FRONTEND_DIR), "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
