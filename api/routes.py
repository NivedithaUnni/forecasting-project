from fastapi import APIRouter, HTTPException
import joblib
import json
import numpy as np
import pandas as pd
import os

from keras.models import load_model

from src.features.feature_engineering import create_features
from src.models.lstm_model import forecast_lstm, prepare_state_data

router = APIRouter()

# ---------------------------------------------------
# LOAD MODEL SELECTION
# ---------------------------------------------------
try:
    with open("models/final_model_selection.json", "r") as f:
        MODEL_SELECTION = json.load(f)

except Exception as e:
    print("❌ Error loading model selection:", e)
    MODEL_SELECTION = {}

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------
try:
    df = pd.read_csv("data/processed/cleaned_data.csv")

    df["date"] = pd.to_datetime(df["date"])

    # Clean state names
    df["state"] = df["state"].astype(str).str.strip()

except Exception as e:
    print("❌ Error loading dataset:", e)
    df = pd.DataFrame()


# ---------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------
@router.get("/health")
def health():
    return {"status": "OK"}


# ---------------------------------------------------
# PREDICT ROUTE
# ---------------------------------------------------
@router.get("/predict")
def predict(state: str):

    try:

        # ---------------------------------------------------
        # CLEAN INPUT
        # ---------------------------------------------------
        state = state.strip()

        print(f"\n🚀 Prediction request for: {state}")

        # ---------------------------------------------------
        # CHECK STATE
        # ---------------------------------------------------
        if state not in MODEL_SELECTION:

            print("❌ State missing in MODEL_SELECTION")

            raise HTTPException(
                status_code=404,
                detail=f"State '{state}' not found"
            )

        # ---------------------------------------------------
        # GET BEST MODEL
        # ---------------------------------------------------
        best_model = MODEL_SELECTION[state]["best_model"]

        print(f"✅ Best Model: {best_model}")

        # ---------------------------------------------------
        # FILTER STATE DATA
        # ---------------------------------------------------
        state_df = df[df["state"] == state].copy()

        if state_df.empty:

            print("❌ No data found for state")

            raise HTTPException(
                status_code=404,
                detail=f"No data for state '{state}'"
            )

        # ===================================================
        # XGBOOST
        # ===================================================
        if best_model == "XGBoost":

            model_path = f"models/saved_models/xgb/{state}.pkl"

            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Model file missing: {model_path}"
                )

            model = joblib.load(model_path)

            # Create features
            df_feat = create_features(state_df).dropna()

            if df_feat.empty:
                raise HTTPException(
                    status_code=500,
                    detail="Feature engineering returned empty dataframe"
                )

            input_row = df_feat.iloc[-1].copy()

            features = [
                "lag_1",
                "lag_7",
                "lag_30",
                "roll_mean_7",
                "roll_std_7",
                "dayofweek",
                "month",
                "is_holiday"
            ]

            forecast = []

            for _ in range(8):

                X_input = np.array(
                    [input_row[features].values]
                )

                pred = model.predict(X_input)[0]

                pred = float(pred)

                forecast.append(pred)

                # Recursive update
                input_row["lag_30"] = input_row["lag_7"]
                input_row["lag_7"] = input_row["lag_1"]
                input_row["lag_1"] = pred

                input_row["roll_mean_7"] = np.mean(
                    forecast[-7:]
                )

                input_row["roll_std_7"] = np.std(
                    forecast[-7:]
                ) if len(forecast) > 1 else 0

            return {
                "state": state,
                "model": best_model,
                "forecast_8_weeks": forecast
            }

        # ===================================================
        # PROPHET
        # ===================================================
        elif best_model == "Prophet":

            model_path = f"models/saved_models/prophet/{state}.pkl"

            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Model file missing: {model_path}"
                )

            model = joblib.load(model_path)

            future = model.make_future_dataframe(
                periods=8
            )

            forecast_df = model.predict(future)

            forecast = forecast_df["yhat"].tail(8).tolist()

            return {
                "state": state,
                "model": best_model,
                "forecast_8_weeks": forecast
            }

        # ===================================================
        # ARIMA
        # ===================================================
        elif best_model == "ARIMA":

            model_path = f"models/saved_models/arima/{state}.pkl"

            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Model file missing: {model_path}"
                )

            model = joblib.load(model_path)

            forecast = model.forecast(steps=8)

            return {
                "state": state,
                "model": best_model,
                "forecast_8_weeks": forecast.tolist()
            }

        # ===================================================
        # LSTM
        # ===================================================
        elif best_model == "LSTM":

            model_path = f"models/saved_models/lstm/{state}.keras"

            scaler_path = f"models/saved_models/lstm/{state}_scaler.pkl"

            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"LSTM model missing: {model_path}"
                )

            if not os.path.exists(scaler_path):
                raise HTTPException(
                    status_code=500,
                    detail=f"Scaler missing: {scaler_path}"
                )

            # Load model
            model = load_model(model_path)

            # Load scaler
            scaler = joblib.load(scaler_path)

            # Prepare data
            data = prepare_state_data(df, state)

            # Forecast
            forecast = forecast_lstm(
                model,
                scaler,
                data
            )

            return {
                "state": state,
                "model": best_model,
                "forecast_8_weeks": forecast
            }

        # ===================================================
        # UNSUPPORTED MODEL
        # ===================================================
        else:

            raise HTTPException(
                status_code=500,
                detail=f"Unsupported model '{best_model}'"
            )

    except HTTPException as e:
        raise e

    except Exception as e:

        print("\n🔥 INTERNAL ERROR")
        print(str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )