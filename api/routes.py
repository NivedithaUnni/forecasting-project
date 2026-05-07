from fastapi import APIRouter, HTTPException
import joblib
import json
import numpy as np
import pandas as pd

from src.features.feature_engineering import create_features
from src.models.lstm_model import forecast_lstm, prepare_state_data

router = APIRouter()

# -----------------------------
# LOAD ONCE (IMPORTANT FIX)
# -----------------------------
with open("models/final_model_selection.json", "r") as f:
    MODEL_SELECTION = json.load(f)

df = pd.read_csv("data/processed/cleaned_data.csv")
df["date"] = pd.to_datetime(df["date"])


# -----------------------------
# HEALTH CHECK
# -----------------------------
@router.get("/health")
def health():
    return {"status": "OK"}


# -----------------------------
# CORE PREDICTOR
# -----------------------------
@router.get("/predict")
def predict(state: str):

    state = state.strip()

    if state not in MODEL_SELECTION:
        raise HTTPException(status_code=404, detail="State not found")

    best_model = MODEL_SELECTION[state]["best_model"]

    state_df = df[df["state"] == state].copy()

    if state_df.empty:
        raise HTTPException(status_code=404, detail="No data for state")

    print(f"🚀 State: {state} | Model: {best_model}")


    # -----------------------------
    # XGBOOST
    # -----------------------------
    if best_model == "XGBoost":

        model = joblib.load(f"models/saved_models/xgb/{state}.pkl")

        df_feat = create_features(state_df).dropna()

        if df_feat.empty:
            raise HTTPException(status_code=500, detail="Not enough feature data")

        input_row = df_feat.iloc[-1].copy()

        features = [
            "lag_1", "lag_7", "lag_30",
            "roll_mean_7", "roll_std_7",
            "dayofweek", "month", "is_holiday"
        ]

        forecast = []

        for _ in range(8):

            X_input = np.array([input_row[features].values])
            pred = model.predict(X_input)[0]

            forecast.append(float(pred))

            # 🔥 recursive update (IMPORTANT)
            input_row["lag_30"] = input_row["lag_7"]
            input_row["lag_7"] = input_row["lag_1"]
            input_row["lag_1"] = pred

            input_row["roll_mean_7"] = np.mean(forecast[-7:])
            input_row["roll_std_7"] = np.std(forecast[-7:]) if len(forecast) > 1 else 0

        return {
            "state": state,
            "model": best_model,
            "forecast_8_weeks": forecast
        }


    # -----------------------------
    # PROPHET
    # -----------------------------
    elif best_model == "Prophet":

        model = joblib.load(f"models/saved_models/prophet/{state}.pkl")

        future = model.make_future_dataframe(periods=8)
        forecast_df = model.predict(future)

        return {
            "state": state,
            "model": best_model,
            "forecast_8_weeks": forecast_df["yhat"].tail(8).tolist()
        }


    # -----------------------------
    # ARIMA
    # -----------------------------
    elif best_model == "ARIMA":

        model = joblib.load(f"models/saved_models/arima/{state}.pkl")

        forecast = model.forecast(steps=8)

        return {
            "state": state,
            "model": best_model,
            "forecast_8_weeks": forecast.tolist()
        }


    # -----------------------------
    # LSTM
    # -----------------------------
    elif best_model == "LSTM":

        model, scaler = joblib.load(f"models/saved_models/lstm/{state}.pkl")

        data = prepare_state_data(df, state)

        forecast = forecast_lstm(model, scaler, data)

        return {
            "state": state,
            "model": best_model,
            "forecast_8_weeks": forecast
        }


    else:
        raise HTTPException(status_code=500, detail="Unsupported model")