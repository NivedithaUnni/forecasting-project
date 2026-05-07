# src/inference/predictor.py

import numpy as np
import joblib
from keras.models import load_model

class Predictor:

    def __init__(self):
        self.arima_models = joblib.load("src/models/arima.pkl")
        self.xgb_model = joblib.load("src/models/xgboost.pkl")
        self.lstm_model = load_model("src/models/lstm.keras")
        # Prophet usually stored per state OR separately handled

    def predict_arima(self, state, steps=8):
        model = self.arima_models[state]
        return model.forecast(steps=steps).tolist()

    def predict_xgboost(self, features):
        return self.xgb_model.predict(features).tolist()

    def predict_lstm(self, X):
        pred = self.lstm_model.predict(X)
        return pred.flatten().tolist()

    def get_best_model(self, state_mae_dict):
        # simple logic (you can improve later)
        return min(state_mae_dict, key=state_mae_dict.get)