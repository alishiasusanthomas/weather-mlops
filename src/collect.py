
import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os
import time


def fetch_weather(lat, lon, start_date, end_date, variables, retries=3):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(variables),
        "timezone": "Asia/Kolkata"
    }
    for attempt in range(1, retries + 1):
        try:
            print(f"  Attempt {attempt}/{retries}...")
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data["hourly"])
            df["time"] = pd.to_datetime(df["time"])
            return df
        except requests.exceptions.ReadTimeout:
            print(f"  Timeout on attempt {attempt}. Waiting 10s before retry...")
            if attempt < retries:
                time.sleep(10)
            else:
                raise
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
            raise


def main():
    print("Starting collect.py...")

    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    variables = params["collect"]["variables"]
    locations = params["collect"]["locations"]
    days_ago = params["collect"]["start_days_ago"]

    end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"End date: {end_date}")
    print(f"Locations: {list(locations.keys())}")

    os.makedirs("data/raw", exist_ok=True)

    for name, coords in locations.items():
        out_path = f"data/raw/{name}.csv"
        print(f"[{name}] Checking {out_path}...")
        print(f"[{name}] File exists: {os.path.exists(out_path)}")

        start_date = (datetime.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        print(f"[{name}] Fetching {start_date} to {end_date}...")
        df = fetch_weather(coords["lat"], coords["lon"], start_date, end_date, variables)
        df.sort_values("time", inplace=True)
        df.to_csv(out_path, index=False)
        print(f"[{name}] Saved {len(df)} rows to {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()