import pandas as pd

from src.data.preprocess import preprocess
from features.feature_engineering import create_features

from src.models.xgboost_model import train_xgboost
from src.models.prophet_model import train_prophet
from src.models.arima_model import train_arima
from src.models.model_selector import select_best_model
from src.models.lstm_evaluator import evaluate_lstm_all_states
from src.data.predict import predict_future


def load_data():
    print("🚀 load_data() STARTED")  # DEBUG LINE

    file_path = "data/raw/sales_data.xlsx"

    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()

    print("\n📊 Data Loaded Successfully!\n")

    df = preprocess(df)
    df, feature_cols = create_features(df)

    print("\n🚀 Training Models...\n")

    xgb_output = train_xgboost(df)

    print("✅ XGBoost training completed")

    return df


if __name__ == "__main__":
    print("🔥 SCRIPT STARTED")
    load_data()
    print("🔥 SCRIPT FINISHED")