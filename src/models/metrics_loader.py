import joblib

def load_xgb_metrics():
    return joblib.load("models/saved_models/xgb_metadata.pkl")

def load_prophet_metrics():
    return joblib.load("models/saved_models/prophet_metadata.pkl")

def load_arima_metrics():
    return joblib.load("models/saved_models/arima_metadata.pkl")

def load_lstm_metrics():
    return joblib.load("models/saved_models/lstm_metadata.pkl")