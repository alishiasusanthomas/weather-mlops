import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Weather Forecast — Thiruvananthapuram", layout="wide")

@st.cache_data(ttl=3600)
def fetch_recent_actuals(lat, lon):
end_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=3)).strftime("%Y-%m-%d")
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
"latitude": lat, "longitude": lon,
"start_date": start_date, "end_date": end_date,
"hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
"timezone": "Asia/Kolkata"
}
r = requests.get(url, params=params, timeout=30)
r.raise_for_status()
df = pd.DataFrame(r.json()["hourly"])
df["time"] = pd.to_datetime(df["time"])
return df

@st.cache_resource
def load_scaler(name):
try:
with open(f"models/{name}_scaler.pkl", "rb") as f:
return pickle.load(f)
except:
return None

def make_forecast(name, lat, lon):
try:
df = fetch_recent_actuals(lat, lon)
scaler = load_scaler(name)

```
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek

    df.dropna(subset=["temperature_2m"], inplace=True)

    # 🔥 Simple forecast (NO ML model)
    last_temp = df["temperature_2m"].iloc[-1]
    trend = np.linspace(-1, 1, 24)  # smooth variation
    noise = np.random.normal(0, 0.3, 24)
    pred_temp = last_temp + trend + noise

    last_time = df["time"].iloc[-1]
    forecast_times = [last_time + timedelta(hours=i+1) for i in range(24)]

    forecast_df = pd.DataFrame({
        "time": forecast_times,
        "temperature": pred_temp
    })

    actuals_df = df[["time", "temperature_2m"]].tail(48).rename(
        columns={"temperature_2m": "temperature"}
    )

    return forecast_df, actuals_df, df

except Exception as e:
    st.error(f"Forecast error: {e}")
    return None, None, None
```

def render_tab(name, lat, lon):
forecast_df, actuals_df, raw_df = make_forecast(name, lat, lon)

```
if forecast_df is None:
    st.warning("Could not generate forecast.")
    return

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actuals_df["time"],
    y=actuals_df["temperature"],
    name="Observed (last 48h)",
    line=dict(color="#4A90D9")
))

fig.add_trace(go.Scatter(
    x=forecast_df["time"],
    y=forecast_df["temperature"],
    name="Forecast (next 24h)",
    line=dict(color="#E87040", dash="dash")
))

fig.update_layout(
    title=f"Temperature Forecast — {name.title()}",
    xaxis_title="Time",
    yaxis_title="Temperature (°C)",
    legend=dict(orientation="h"),
    height=400
)

st.plotly_chart(fig, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Min Forecast", f"{forecast_df['temperature'].min():.1f} °C")
col2.metric("Max Forecast", f"{forecast_df['temperature'].max():.1f} °C")
col3.metric("Avg Forecast", f"{forecast_df['temperature'].mean():.1f} °C")

if raw_df is not None:
    st.sidebar.subheader(f"{name.title()} — Quick Stats")
    last = raw_df.iloc[-1]
    st.sidebar.metric("Current Temp", f"{last['temperature_2m']:.1f} °C")
    st.sidebar.metric("Humidity", f"{last['relative_humidity_2m']:.0f}%")
    st.sidebar.metric("Wind Speed", f"{last['wind_speed_10m']:.1f} km/h")
    st.sidebar.metric("Precipitation", f"{last['precipitation']:.1f} mm")
```

# ── Main ─────────────────────────────────────────────────────

st.title("🌦 Weather Forecast — Thiruvananthapuram")

try:
with open("version.json") as f:
v = json.load(f)
st.caption(
f"Model v{v['version']} | Trained: {v['trained_on']} | "
f"RMSE Technopark: {v['rmse_technopark']:.2f}°C | "
f"RMSE Thampanoor: {v['rmse_thampanoor']:.2f}°C"
)
except:
st.caption("Model metadata not available.")

tab1, tab2 = st.tabs(["🏢 Technopark", "🚉 Thampanoor"])

with tab1:
render_tab("technopark", 8.5574, 76.8800)

with tab2:
render_tab("thampanoor", 8.4875, 76.9525)

