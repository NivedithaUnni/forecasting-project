import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import joblib

from api.services.loader import load_model_selection


# -------------------------
# MAIN PREDICTION FUNCTION
# -------------------------
def forecast(state, df, model_type="auto"):

    model_map = load_model_selection()
    best_model = model_map[state]["best_model"]

    if model_type != "auto":
        best_model = model_type

    df_state = df[df["state"] == state].copy()
    df_state = df_state.sort_values("date")

    # ---------------- ARIMA ----------------
    if best_model == "ARIMA":
        series = df_state["total"].values
        model = ARIMA(series, order=(1,1,1)).fit()
        forecast = model.forecast(steps=8)
        return forecast.tolist()

    # ---------------- PROPHET ----------------
    if best_model == "Prophet":
        temp = df_state.rename(columns={"date": "ds", "total": "y"})
        m = Prophet()
        m.fit(temp)

        future = m.make_future_dataframe(periods=8, freq="W")
        pred = m.predict(future)["yhat"].tail(8)

        return pred.tolist()

    # ---------------- XGBOOST ----------------
    if best_model == "XGBoost":
        model = joblib.load("models/saved_models/xgboost.pkl")

        last_val = df_state["total"].iloc[-1]
        return [last_val] * 8  # placeholder simple baseline

    # ---------------- LSTM ----------------
    if best_model == "LSTM":
        return [df_state["total"].mean()] * 8  # placeholder

    return []