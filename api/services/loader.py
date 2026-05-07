import joblib
import json
import os

MODEL_PATH = "models/saved_models"
META_PATH = "models/final_model_selection.json"

def load_model_selection():
    with open(META_PATH, "r") as f:
        return json.load(f)


def load_xgboost():
    return joblib.load(f"{MODEL_PATH}/xgboost.pkl")


def load_columns():
    return joblib.load(f"{MODEL_PATH}/xgb_columns.pkl")