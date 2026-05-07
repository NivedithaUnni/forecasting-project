import tensorflow as tf
import joblib

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def get_model():
    return model, scaler