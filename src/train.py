""" 
Train model for image classification using TensorFlow and Keras.
Using a baseline CNN and transfer learning with MobileNetV2.
"""
from pathlib import Path
from datetime import datetime

# pylint: disable=E0611
from tensorflow import keras

from src.data_pipeline import build_datasets
from src.model import build_baseline_cnn, build_transfer_model

# pylint: disable=R0914
def train(
    data_dir: str = "data",
    use_transfer: bool = True,
    epochs: int = 10,
    learning_rate: float = 1e-3,
):
    """
    Train an image classification model.
    Args:
        data_dir: Directory containing 'train', 'val', 'test' subdirectories.
        use_transfer: Whether to use transfer learning (MobileNetV2) or baseline CNN.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    Returns:
        model: Trained Keras model.
        history: Training history object.
        class_names: List of class names.
    """
    # 1. Build datasets
    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)
    input_shape = (224, 224, 3)

    print("Classes:", class_names)
    print("Input shape:", input_shape)

    # 2. Build model (baseline CNN or transfer learning)
    if use_transfer:
        print("Using transfer learning model (MobileNetV2)")
        model = build_transfer_model(input_shape, num_classes)
    else:
        print("Using baseline CNN model")
        model = build_baseline_cnn(input_shape, num_classes)

    # 3. Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    # 4. Set up callbacks: TensorBoard, EarlyStopping, ModelCheckpoint
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = "transfer" if use_transfer else "baseline_CNN"
    log_dir = Path("logs") / f"run_{run_id}"
    checkpoint_dir = Path("models") / f"{model_name}_{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    )
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_dir / "model.keras"),
        monitor="val_loss",
        save_best_only=True,
    )

    # 5. Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[tensorboard_cb, early_stopping_cb, checkpoint_cb],
    )

    # 6. Evaluate on test set
    print("\nEvaluating on test set:")
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    return model, history, class_names


if __name__ == "__main__":
    # run with transfer learning with MobileNetV2
    train(
        data_dir="data",
        use_transfer=True,
        epochs=10,
        learning_rate=1e-3,
    )

    # run baseline CNN
    train(
        data_dir="data",
        use_transfer=False,
        epochs=10,
        learning_rate=1e-3,
    )
