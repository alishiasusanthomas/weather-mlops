import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import yaml
import os


def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=["time"])
    df.dropna(subset=["temperature_2m"], inplace=True)
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_windows(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon, 0])  # temperature_2m is col 0
    return np.array(X), np.array(y)


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    window_size = params["preprocess"]["window_size"]
    horizon = params["preprocess"]["horizon"]
    test_split = params["preprocess"]["test_split"]

    feature_cols = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "hour", "day_of_week"]

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    locations = ["technopark", "thampanoor"]

    for name in locations:
        print(f"[{name}] Preprocessing...")
        df = load_and_clean(f"data/raw/{name}.csv")

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[feature_cols])

        with open(f"models/{name}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        X, y = build_windows(scaled, window_size, horizon)

        split = int(len(X) * (1 - test_split))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        np.save(f"data/processed/{name}_X_train.npy", X_train)
        np.save(f"data/processed/{name}_y_train.npy", y_train)
        np.save(f"data/processed/{name}_X_test.npy", X_test)
        np.save(f"data/processed/{name}_y_test.npy", y_test)

        print(f"[{name}] Train: {X_train.shape}, Test: {X_test.shape}")

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
