# src/train.py
from pathlib import Path
from datetime import datetime

import tensorflow as tf
# pylint: disable=E0611
from tensorflow import keras

from data_pipeline import build_datasets
from model import build_baseline_cnn, build_transfer_model


def get_run_logdir(base_dir: str = "logs") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(Path(base_dir) / f"run_{timestamp}")


def train(
    data_dir: str = "data",
    use_transfer: bool = True,
    train_base: bool = False,
    epochs: int = 10,
    learning_rate: float = 1e-3,
):
    # 1. Build datasets
    train_ds, val_ds, test_ds, class_names = build_datasets(data_dir)
    num_classes = len(class_names)
    IMG_SIZE = (224, 224)
    input_shape = (*IMG_SIZE, 3)

    print("Classes:", class_names)
    print("Input shape:", input_shape)

    # 2. Build model (baseline CNN or transfer learning)
    if use_transfer:
        print("Using transfer learning model (MobileNetV2)")
        model = build_transfer_model(input_shape, num_classes, train_base=train_base)
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
    log_dir = get_run_logdir()
    checkpoint_dir = Path("models") / "best_model"
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
    # Example: run with transfer learning
    train(
        data_dir="data",
        use_transfer=True,
        train_base=False,  # later you can try True for fine-tuning
        epochs=10,
        learning_rate=1e-3,
    )

