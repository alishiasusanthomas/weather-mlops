import numpy as np
import json
import yaml
import os
import subprocess
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_model(input_shape, horizon, units, dropout, lr):
    model = Sequential([
        LSTM(units, input_shape=input_shape, return_sequences=False),
        Dropout(dropout),
        Dense(horizon)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model


def get_git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    units = params["model"]["lstm_units"]
    dropout = params["model"]["dropout"]
    epochs = params["model"]["epochs"]
    patience = params["model"]["patience"]
    batch_size = params["model"]["batch_size"]
    lr = params["model"]["learning_rate"]
    horizon = params["preprocess"]["horizon"]

    os.makedirs("models", exist_ok=True)

    metrics = {}
    locations = ["technopark", "thampanoor"]

    for name in locations:
        print(f"\n[{name}] Loading data...")
        X_train = np.load(f"data/processed/{name}_X_train.npy")
        y_train = np.load(f"data/processed/{name}_y_train.npy")
        X_test  = np.load(f"data/processed/{name}_X_test.npy")
        y_test  = np.load(f"data/processed/{name}_y_test.npy")

        print(f"[{name}] Building model...")
        model = build_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            horizon=horizon,
            units=units,
            dropout=dropout,
            lr=lr
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

        print(f"[{name}] Training...")
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1
        )

        y_pred = model.predict(X_test)
        mae  = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}
        print(f"[{name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        model.save(f"models/{name}_model.keras")
        print(f"[{name}] Model saved.")

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    version_data = {
        "version": datetime.today().strftime("%Y%m%d"),
        "trained_on": datetime.today().strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": get_git_sha(),
        "rmse_technopark": metrics["technopark"]["rmse"],
        "rmse_thampanoor": metrics["thampanoor"]["rmse"],
        "mae_technopark": metrics["technopark"]["mae"],
        "mae_thampanoor": metrics["thampanoor"]["mae"]
    }

    with open("version.json", "w") as f:
        json.dump(version_data, f, indent=2)

    print("\nTraining complete. metrics.json and version.json written.")


if __name__ == "__main__":
    main()
