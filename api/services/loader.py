import os
import json
import joblib

from keras.models import load_model


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
MODEL_PATH = "models/saved_models"

META_PATH = "models/final_model_selection.json"


# ---------------------------------------------------
# LOAD MODEL SELECTION
# ---------------------------------------------------
def load_model_selection():

    if not os.path.exists(META_PATH):

        raise FileNotFoundError(
            f"Missing metadata file: {META_PATH}"
        )

    with open(META_PATH, "r") as f:

        return json.load(f)


# ---------------------------------------------------
# LOAD XGBOOST MODEL
# ---------------------------------------------------
def load_xgboost(state):

    model_path = (
        f"{MODEL_PATH}/xgb/{state}.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Missing XGBoost model: {model_path}"
        )

    return joblib.load(model_path)


# ---------------------------------------------------
# LOAD ARIMA MODEL
# ---------------------------------------------------
def load_arima(state):

    model_path = (
        f"{MODEL_PATH}/arima/{state}.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Missing ARIMA model: {model_path}"
        )

    return joblib.load(model_path)


# ---------------------------------------------------
# LOAD PROPHET MODEL
# ---------------------------------------------------
def load_prophet(state):

    model_path = (
        f"{MODEL_PATH}/prophet/{state}.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Missing Prophet model: {model_path}"
        )

    return joblib.load(model_path)


# ---------------------------------------------------
# LOAD LSTM MODEL + SCALER
# ---------------------------------------------------
def load_lstm(state):

    model_path = (
        f"{MODEL_PATH}/lstm/{state}.keras"
    )

    scaler_path = (
        f"{MODEL_PATH}/lstm/{state}_scaler.pkl"
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Missing LSTM model: {model_path}"
        )

    if not os.path.exists(scaler_path):

        raise FileNotFoundError(
            f"Missing scaler: {scaler_path}"
        )

    model = load_model(model_path)

    scaler = joblib.load(scaler_path)

    return model, scaler


# ---------------------------------------------------
# LOAD FEATURE COLUMNS
# ---------------------------------------------------
def load_columns():

    columns_path = (
        f"{MODEL_PATH}/xgb_columns.pkl"
    )

    if not os.path.exists(columns_path):

        raise FileNotFoundError(
            f"Missing columns file: {columns_path}"
        )

    return joblib.load(columns_path)