import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os


def fetch_weather(lat, lon, start_date, end_date, variables):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "Asia/Kolkata"
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        raise

    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    variables = params["collect"]["variables"]
    locations = params["collect"]["locations"]
    days_ago = params["collect"]["start_days_ago"]

    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs("data/raw", exist_ok=True)

    for name, coords in locations.items():
        out_path = f"data/raw/{name}.csv"

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, parse_dates=["time"])
            last_date = existing["time"].max()
            start_date = (last_date + timedelta(hours=1)).strftime("%Y-%m-%d")
            if start_date > end_date:
                print(f"[{name}] Already up to date.")
                continue
            print(f"[{name}] Fetching {start_date} to {end_date}...")
            new_df = fetch_weather(coords["lat"], coords["lon"], start_date, end_date, variables)
            df = pd.concat([existing, new_df], ignore_index=True)
            df.drop_duplicates(subset="time", inplace=True)
        else:
            start_date = (datetime.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            print(f"[{name}] Initial fetch: {start_date} to {end_date}...")
            df = fetch_weather(coords["lat"], coords["lon"], start_date, end_date, variables)

        df.sort_values("time", inplace=True)
        df.to_csv(out_path, index=False)
        print(f"[{name}] Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
