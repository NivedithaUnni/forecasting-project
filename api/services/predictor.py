import os
import json
import joblib
import numpy as np
import pandas as pd

from keras.models import load_model

from api.services.loader import load_model_selection

from src.features.feature_engineering import create_features

from src.models.lstm_model import (
    forecast_lstm,
    prepare_state_data
)


# ---------------------------------------------------
# MAIN FORECAST FUNCTION
# ---------------------------------------------------
def forecast(state, df, model_type="auto"):

    try:

        # ---------------------------------------------------
        # CLEAN INPUT
        # ---------------------------------------------------
        state = state.strip()

        df["state"] = (
            df["state"]
            .astype(str)
            .str.strip()
        )

        # ---------------------------------------------------
        # LOAD MODEL SELECTION
        # ---------------------------------------------------
        model_map = load_model_selection()

        if state not in model_map:

            raise Exception(
                f"State '{state}' not found"
            )

        # ---------------------------------------------------
        # GET BEST MODEL
        # ---------------------------------------------------
        best_model = model_map[state]["best_model"]

        if model_type != "auto":
            best_model = model_type

        print(f"\n🚀 {state} → {best_model}")

        # ---------------------------------------------------
        # FILTER STATE DATA
        # ---------------------------------------------------
        df_state = df[df["state"] == state].copy()

        df_state = df_state.sort_values("date")

        if df_state.empty:

            raise Exception(
                f"No data found for {state}"
            )

        # ===================================================
        # ARIMA
        # ===================================================
        if best_model == "ARIMA":

            model_path = (
                f"models/saved_models/arima/{state}.pkl"
            )

            if not os.path.exists(model_path):

                raise Exception(
                    f"Missing ARIMA model: {model_path}"
                )

            model = joblib.load(model_path)

            forecast_values = model.forecast(
                steps=8
            )

            return forecast_values.tolist()

        # ===================================================
        # PROPHET
        # ===================================================
        elif best_model == "Prophet":

            model_path = (
                f"models/saved_models/prophet/{state}.pkl"
            )

            if not os.path.exists(model_path):

                raise Exception(
                    f"Missing Prophet model: {model_path}"
                )

            model = joblib.load(model_path)

            future = model.make_future_dataframe(
                periods=8,
                freq="W"
            )

            forecast_df = model.predict(future)

            forecast_values = (
                forecast_df["yhat"]
                .tail(8)
                .tolist()
            )

            return forecast_values

        # ===================================================
        # XGBOOST
        # ===================================================
        elif best_model == "XGBoost":

            model_path = (
                f"models/saved_models/xgb/{state}.pkl"
            )

            if not os.path.exists(model_path):

                raise Exception(
                    f"Missing XGBoost model: {model_path}"
                )

            # Load model
            model = joblib.load(model_path)

            # Feature engineering
            df_feat = create_features(df_state)

            df_feat = df_feat.dropna()

            if df_feat.empty:

                raise Exception(
                    "Feature dataframe empty"
                )

            # Latest row
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

            forecast_values = []

            # Recursive forecasting
            for _ in range(8):

                X_input = np.array([
                    input_row[features].values
                ])

                pred = model.predict(X_input)[0]

                pred = float(pred)

                forecast_values.append(pred)

                # Update recursive features
                input_row["lag_30"] = input_row["lag_7"]

                input_row["lag_7"] = input_row["lag_1"]

                input_row["lag_1"] = pred

                input_row["roll_mean_7"] = np.mean(
                    forecast_values[-7:]
                )

                input_row["roll_std_7"] = np.std(
                    forecast_values[-7:]
                )

            return forecast_values

        # ===================================================
        # LSTM
        # ===================================================
        elif best_model == "LSTM":

            model_path = (
                f"models/saved_models/lstm/{state}.keras"
            )

            scaler_path = (
                f"models/saved_models/lstm/{state}_scaler.pkl"
            )

            if not os.path.exists(model_path):

                raise Exception(
                    f"Missing LSTM model: {model_path}"
                )

            if not os.path.exists(scaler_path):

                raise Exception(
                    f"Missing scaler: {scaler_path}"
                )

            # Load keras model
            model = load_model(model_path)

            # Load scaler
            scaler = joblib.load(scaler_path)

            # Prepare data
            data = prepare_state_data(
                df,
                state
            )

            # Forecast
            forecast_values = forecast_lstm(
                model,
                scaler,
                data
            )

            return forecast_values

        # ===================================================
        # UNSUPPORTED MODEL
        # ===================================================
        else:

            raise Exception(
                f"Unsupported model '{best_model}'"
            )

    except Exception as e:

        print("\n🔥 FORECAST ERROR")
        print(str(e))

        raise e