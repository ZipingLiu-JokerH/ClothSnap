# ClothSnap

Ottawa DTI6302 Machine Learning Opearations end-to-end project

[![Super-Linter](https://github.com/ZipingLiu-JokerH/ClothSnap/actions/workflows/lint.yml/badge.svg)](https://github.com/marketplace/actions/super-linter)

## :star: Project Overview

This project builds an end-to-end machine learning system that classifies clothing images into predefined categories. With modern households owning large and ever-changing wardrobes, organizing and tracking clothing items can be difficult. A reliable image-based classifier can serve as the first step toward an automated wardrobe-management system, where users simply take a photo and have the item identified and categorized.

To demonstrate this concept, the project uses TensorFlow to train a clothing-classification model and integrates it into a simple web application that provides immediate predictions for user-uploaded images.

## :mag_right: Dataset Selection

For this project, we use the [Top-10 Clothing Dataset](https://github.com/alexeygrigorev/clothing-dataset-small), a subset of the [Clothing Dataset](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full) published on Kaggle by [Alexey Grigorev](https://www.linkedin.com/in/agrigorev/). This version contains images from the ten most common clothing categories, addressing the class-imbalance issues present in the full 20-class dataset. Please create a `data/` folder in the project root and place the downloaded data inside.

### Features

Each data point consists of a single RGB image of a clothing item. The images vary in background, lighting, orientation, and zoom level, providing realistic visual diversity.

### Target Variable

The target variable is the clothing category, represented as one of the ten class labels:
T-shirt, Long Sleeve, Pants, Shirt, Shoes, Dress, Shorts, Outwear, Hat, and Skirt.
Each image belongs to exactly one category.

### Relevance and Significance

This dataset aligns well with the project’s goal of building an image-based clothing classifier that could support a future wardrobe-management system. By focusing on the ten most common clothing categories, the dataset provides enough balanced examples for reliable model training while still reflecting the types of items people frequently own. Its natural variation in lighting and backgrounds makes it practical for a real user scenario where photos may be taken casually at home.

## :gear: Model Training and Pipeline

- Framework: TensorFlow/Keras with two architectures: a baseline CNN and a transfer model (MobileNetV2) using 224x224x3 inputs.
- Data pipeline (`src/data_pipeline.py`): reads `data/train`, `data/validation`, `data/test`, resizes to 224x224, normalizes to [0,1]; applies light augmentation (flip/rotate/zoom) on training only.
- Training (`src/train.py`): builds datasets, chooses baseline vs transfer (`use_transfer`), compiles with Adam + SparseCategoricalCrossentropy, trains with TensorBoard logging, EarlyStopping, and ModelCheckpoint saving the best `.keras` checkpoint. Default uses transfer learning with the base frozen (`train_base=False`).
- Outputs: best `.keras` checkpoint under `models/`, and optional TensorFlow SavedModel export via `src/export_saved_model.py` for TensorFlow Serving.

## :hammer_and_wrench: Model Development Flow

- Train locally: run `python src/train.py` to produce `.keras` checkpoints under `models/`.
- Export for serving: `python src/export_saved_model.py --model-path <your_checkpoint>` writes a TF Serving-ready SavedModel to `models/tf_serving_ready_model`.
- Bundle model for distribution: add `--bundle` to create `artifacts/model.tar.gz` for uploading to S3.
- Serving images: CI builds two images (Flask UI/API and TF Serving with baked model) on merges to `main` and pushes to GHCR.
- Deploy: CI pulls the published images onto aws ec2 instance and start Docker compose to host the application.

## :busts_in_silhouette: Developer Guide

- Prereqs: Python 3.10, Docker + docker-compose, AWS CLI (for S3), GHCR read access.
- Install deps: `pip install -r requirements.txt`
- Train: `python src/train.py`. Checkpoints go to `models/`.
- Export for serving: `python src/export_saved_model.py --model-path <your_checkpoint> [--bundle]`
- Local UI/API + TF Serving (no S3): First ensure `models/tf_serving_ready_model` exists (exported locally), then run `docker compose -f docker-compose.local.yml up --build` → open `http://localhost:5001`. To stop the container, run `docker compose -f docker-compose.local.yml down`
- Tests: use `pytest`; GitHub Actions runs tests on PRs (`.github/workflows/tests.yml`).
- Lint/format: use Visual Studio Code extensions (pylint, Prettier). CI runs Super-Linter on PRs (`.github/workflows/lint.yml`).
- CI: builds/pushes images on `main` (see `.github/workflows/build-and-push-images.yml`); model URI provided via repository variables.
- Deployment: EC2 pulls the GHCR images and runs Docker compose to host the app; the TF Serving model is baked into the image.
  - For a new model, upload the updated `artifacts/model.tar.gz` to S3 so GitHub Actions can rebuild the TF Serving image with the new artifact. Also update the `MODEL_URI` repository variable on GitHub
