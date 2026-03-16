# 🌦 Weather Forecasting MLOps — Thiruvananthapuram

A production-style MLOps project that forecasts hourly temperature for **Technopark** and **Thampanoor** using an LSTM model, retrained daily via GitHub Actions + DVC.

## Architecture

```
Open-Meteo API → collect.py → preprocess.py → train.py → Streamlit App
                     ↑                                          ↓
               GitHub Actions (daily)              Streamlit Community Cloud
                     ↑
               DVC (Google Drive remote)
```

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd weather-mlops
pip install -r requirements.txt

# 2. Initialize DVC
dvc init
dvc remote add -d gdrive_remote gdrive://YOUR_FOLDER_ID

# 3. Run the full pipeline
python src/collect.py        # fetch data
dvc repro                    # preprocess + train

# 4. Launch the app
streamlit run app.py
```

## Project Structure

| Path | Purpose |
|------|---------|
| `src/collect.py` | Fetches hourly weather from Open-Meteo API |
| `src/preprocess.py` | Cleans data, engineers features, builds LSTM windows |
| `src/train.py` | Trains LSTM, saves model, writes metrics & version |
| `app.py` | Streamlit forecast dashboard |
| `dvc.yaml` | 3-stage DVC pipeline definition |
| `params.yaml` | All hyperparameters and config |
| `.github/workflows/daily_pipeline.yml` | Daily CI/CD retrain workflow |

## GitHub Secrets Required

| Secret | Purpose |
|--------|---------|
| `GDRIVE_CREDENTIALS_DATA` | Base64-encoded Google service account JSON |
| `GIT_TOKEN` | GitHub PAT for committing back to repo |

## MLOps Concepts Covered

- Data versioning with DVC
- Reproducible ML pipelines (DAG stages)
- Parameterised training via `params.yaml`
- Time-series LSTM training & evaluation
- Automated daily retraining (GitHub Actions)
- CI/CD for ML (push → auto-deploy)
- Model metadata tracking (`version.json`)
- Free-tier cloud deployment (Streamlit Cloud)
