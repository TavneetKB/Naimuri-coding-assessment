# app.py
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import tensorflow as tf
import joblib

# ---- Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "../../model/lstm_covid_model.h5")
DATA_PATH = os.getenv(
    "DATA_PATH", "../../data/time_series_data"
)  # TSV (sep='\t') or CSV
SCALER_PATH = os.getenv("SCALER_PATH", "../../model/scaler.pkl")  # REQUIRED

DEFAULT_HORIZON = 30

app = FastAPI(
    title="LSTM COVID Forecaster (single feature, scaler required)", version="1.0"
)

model = None
sequence_length = None
df = None
scaler = None  # MinMaxScaler used in training (required)


class ForecastResponse(BaseModel):
    country: str
    horizon: int
    start_date: str
    dates: list[str]
    predictions: list[int]


def _find_columns(d: pd.DataFrame):
    # Country
    country_col = None
    for c in ["Country", "Country/Region", "country"]:
        if c in d.columns:
            country_col = c
            break
    if not country_col:
        raise RuntimeError(
            "Expected a country column: one of ['Country', 'Country/Region', 'country']."
        )

    # Date
    date_col = None
    for c in ["ObservationDate", "Date", "date", "observation_date"]:
        if c in d.columns:
            date_col = c
            break
    if not date_col:
        raise RuntimeError(
            "Expected a date column: one of ['ObservationDate', 'Date', 'date', 'observation_date']."
        )

    # Daily cases (you said this is 'Confirmed')
    if "Confirmed" not in d.columns:
        raise RuntimeError("Expected 'Confirmed' column with daily new cases.")
    return country_col, date_col, "Confirmed"


@app.on_event("startup")
def load():
    global model, sequence_length, df, scaler_X

    # --- Scaler (required) ---
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            f"Missing required scaler at {SCALER_PATH}. "
            "Export the exact MinMaxScaler used in training (joblib.dump) and place it here."
        )
    scaler_X = joblib.load(SCALER_PATH)

    # --- Model ---
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # Expect (None, L, 1)
    ishape = model.input_shape
    if not ishape or len(ishape) != 3 or ishape[2] != 1:
        raise RuntimeError(
            f"Model must take one feature per timestep; got input_shape={ishape}"
        )
    sequence_length = int(ishape[1])

    # --- Data (TSV first; fallback to CSV) ---
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Data file not found at {DATA_PATH}")
    try:
        raw = pd.read_csv(DATA_PATH, sep="\t")
    except Exception:
        raw = pd.read_csv(DATA_PATH)

    ccol, dcol, ycol = _find_columns(raw)

    # Keep only necessary columns
    d = raw[[ccol, dcol, ycol]].copy()
    d.columns = ["Country", "ObservationDate", "daily_cases"]
    d["ObservationDate"] = pd.to_datetime(d["ObservationDate"])
    d["daily_cases"] = (
        pd.to_numeric(d["daily_cases"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    # Sum across provinces if present
    d = (
        d.groupby(["Country", "ObservationDate"], as_index=False)["daily_cases"]
        .sum()
        .sort_values(["Country", "ObservationDate"])
    )
    df = d.reset_index(drop=True)


def _get_country_series(country: str) -> pd.DataFrame:
    sub = df.loc[df["Country"].str.lower() == country.lower()].copy()
    if sub.empty:
        # allow partial contains fallback
        mask = df["Country"].str.lower().str.contains(country.lower())
        sub = df.loc[mask].copy()
        if sub.empty:
            raise HTTPException(
                status_code=404, detail=f"Country '{country}' not found."
            )
    sub.sort_values("ObservationDate", inplace=True)
    return sub


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    country: str = Query(..., description="Country name (case-insensitive)"),
    horizon: int = Query(DEFAULT_HORIZON, ge=1, le=120),
    raw: int = Query(0, description="Return raw floats if 1; integers otherwise"),
):
    sub = _get_country_series(country)
    y_nat = sub["daily_cases"].astype(float).values
    if len(y_nat) < sequence_length:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {sequence_length} days; got {len(y_nat)}.",
        )

    # Seed: last L natural values -> scale -> (L,)
    last_sequence = y_nat[-sequence_length:]
    scaled_last_seq = scaler_X.transform(last_sequence.reshape(-1, 1)).flatten()
    current_seq = scaled_last_seq.copy()

    preds = []
    for _ in range(horizon):
        X_input = current_seq.reshape(1, sequence_length, 1)
        pred_scaled = model.predict(X_input, verbose=0)  # (1,1) on scaled space
        pred_nat = float(
            scaler_X.inverse_transform(pred_scaled)[0, 0]
        )  # back to natural units
        pred_nat = max(0.0, pred_nat)
        preds.append(pred_nat)
        # roll window with the scaled prediction
        current_seq = np.append(current_seq[1:], float(pred_scaled[0, 0]))

    # Future dates
    last_date = sub["ObservationDate"].max()
    start_date = (last_date + timedelta(days=1)).date()
    dates = [str(start_date + timedelta(days=i)) for i in range(horizon)]

    out = np.maximum(preds, 0.0)
    out_list = out.tolist() if raw else np.rint(out).astype(int).tolist()

    return ForecastResponse(
        country=country,
        horizon=horizon,
        start_date=str(start_date),
        dates=dates,
        predictions=out_list,
    )


@app.get("/")
def root():
    return {
        "message": "OK",
        "requires": {
            "model": MODEL_PATH,
            "data_tsv_or_csv": DATA_PATH,
            "scaler": SCALER_PATH,
        },
        "model_input_sequence_length": sequence_length,
        "example": "/forecast?country=Spain&horizon=30&raw=1",
        "notes": [
            "Uses 'Confirmed' column as daily new cases.",
            "Uses the SAME MinMaxScaler as training.",
            "Single-feature (L,1) input; no extra covariates.",
        ],
    }
