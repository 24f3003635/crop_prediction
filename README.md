# AI-Powered Crop Recommendation System

An end-to-end machine learning application that recommends the most suitable crops to grow based on soil composition and local climate conditions. Given Nitrogen (N), Phosphorous (P), Potassium (K), temperature, humidity, pH, and rainfall, the system returns a ranked top-5 list of recommended crops, each with a confidence score.

## Problem

Farmers often decide what to plant based on general knowledge rather than the specific interplay between soil nutrients and local weather. This project turns raw agricultural data into a data-driven, ranked recommendation instead of a single rigid guess.

## How it works

1. **Model training** — a `RandomForestClassifier` is trained on a labeled agricultural dataset (N/P/K, temperature, humidity, pH, rainfall → crop label, 22 crop classes).
2. **Preprocessing** — features are scaled with `RobustScaler` (resistant to outliers in real-world soil/weather data); crop labels are encoded with `LabelEncoder`.
3. **Prediction** — instead of a single label, `predict_proba()` is used to get a confidence score for every crop class. Results are ranked and the top 5 above a configurable confidence threshold are returned.
4. **Serving** — the trained model and preprocessing artifacts are serialized with `joblib` and served through a Flask REST API, so the model can be retrained independently of the backend code.

## Tech stack

- **ML / Data**: Python, scikit-learn, pandas, NumPy
- **Backend**: Flask (REST API)
- **Model persistence**: joblib
- **Environment management**: uv

## Project structure

```
Crop_Prediction
├── ML
│   ├── AI_Crop_prediction.ipynb
│   ├── crop_model.pkl
│   ├── label_encoder.pkl
│   └── preprocessing_pipeline.pkl
├── README.md
├── app
│   ├── __init__.py
│   └── api.py
├── main.py
├── pyproject.toml
└── uv.lock

```

## Setup

```bash
uv python pin 3.12
uv venv
uv sync
```

## Retrain the model

```bash
uv run python train.py --csv data/Crop_recommendation.csv
```

This regenerates `crop_model.pkl`, `preprocessing_pipeline.pkl`, and `label_encoder.pkl` — no changes to `app.py` are needed afterward.

## Run the API

```bash
uv run python app.py
```

## API

### `GET /test`
Health check.

```json
{ "message": "Backend Online" }
```

### `POST /predict`

**Request body**

```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "min_confidence": 10,
  "top_n": 5
}
```

- `min_confidence` *(optional, default 10)* — minimum confidence percentage for a crop to be included.
- `top_n` *(optional, default 5)* — maximum number of recommendations to return.

**Response**

```json
{
  "recommendations": [
    { "rank": 1, "crop": "rice", "confidence": 99.0 }
  ]
}
```

Fewer than `top_n` results are returned when the model is highly confident in one answer, rather than padding the list with near-zero-probability crops.

## Design notes

- Top-N ranked output with confidence scores instead of a single prediction.
- Adaptive result count based on model confidence.
- Clean separation between training (`train.py`) and serving (`app.py`) — the model can be retrained on new data without touching the API layer.