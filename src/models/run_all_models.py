import pandas as pd
import json

from src.models.xgboost_model import train_xgboost
from src.models.prophet_model import train_prophet
from src.models.arima_model import train_arima
from src.models.lstm_model import train_lstm_model

from src.models.save_all_metrics import save_all_model_metrics


# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")
df = df.dropna().reset_index(drop=True)

states = df["state"].unique()


# -----------------------------
# 2. TRAIN MODELS
# -----------------------------
print("\n🚀 Training XGBoost...")
xgb_result = train_xgboost(df)

xgb_mae_dict = xgb_result["state_mae"]
xgb_avg_mae = xgb_result["avg_mae"]


print("\n🚀 Training Prophet...")
prophet_mae_dict, prophet_avg_mae = train_prophet(df)


print("\n🚀 Training ARIMA...")
arima_output = train_arima(df)

arima_mae_dict = arima_output["state_mae"]
arima_avg_mae = arima_output["avg_mae"]


print("\n🚀 Training LSTM...")
lstm_mae_dict = {}

for state in states:
    _, _, mae = train_lstm_model(df, state)
    lstm_mae_dict[state] = mae


# -----------------------------
# 3. MODEL SELECTION PER STATE
# -----------------------------
final_selection = {}

print("\n📊 Selecting Best Model Per State...\n")

for state in states:

    state_scores = {
        "XGBoost": xgb_mae_dict.get(state, float("inf")),
        "Prophet": prophet_mae_dict.get(state, float("inf")),
        "ARIMA": arima_mae_dict.get(state, float("inf")),
        "LSTM": lstm_mae_dict.get(state, float("inf"))
    }

    best_model = min(state_scores, key=state_scores.get)

    final_selection[state] = {
        "best_model": best_model,
        "scores": state_scores
    }

    print(f"{state} → Best: {best_model}")


# -----------------------------
# 4. SAVE RESULTS
# -----------------------------
with open("models/final_model_selection.json", "w") as f:
    json.dump(final_selection, f, indent=4)

print("\n✅ Final model selection saved!")


# -----------------------------
# 5. SAVE METRICS SUMMARY
# -----------------------------
save_all_model_metrics(
    xgb_mae=xgb_avg_mae,
    prophet_mae=sum(prophet_mae_dict.values()) / len(prophet_mae_dict),
    arima_mae=arima_avg_mae,
    lstm_mae=sum(lstm_mae_dict.values()) / len(lstm_mae_dict)
)