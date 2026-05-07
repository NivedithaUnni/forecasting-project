import joblib
import os

def save_all_model_metrics(xgb_mae, prophet_mae, arima_mae, lstm_mae):

    os.makedirs("models/saved_models", exist_ok=True)

    metrics = {
        "XGBoost": xgb_mae,
        "Prophet": prophet_mae,
        "ARIMA": arima_mae,
        "LSTM": lstm_mae
    }

    joblib.dump(metrics, "models/saved_models/all_model_mae.pkl")

    print("\n✅ Saved all model metrics!")