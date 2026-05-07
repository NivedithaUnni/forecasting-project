import pandas as pd
import json

from src.models.xgboost_model import train_xgboost
from src.models.arima_model import train_arima
from src.models.lstm_model import train_lstm_model

from src.models.save_all_metrics import save_all_model_metrics


# ---------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------
df = pd.read_csv("data/processed/cleaned_data.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.dropna().reset_index(drop=True)

states = df["state"].unique()


# ---------------------------------------------------
# 2. TRAIN XGBOOST
# ---------------------------------------------------
print("\n🚀 Training XGBoost...")

xgb_result = train_xgboost(df)

xgb_mae_dict = xgb_result["state_mae"]

xgb_avg_mae = xgb_result["avg_mae"]


# ---------------------------------------------------
# 3. SKIP PROPHET TEMPORARILY
# ---------------------------------------------------
print("\n⚠️ Skipping Prophet Training...")

prophet_mae_dict = {
    state: float("inf")
    for state in states
}

prophet_avg_mae = float("inf")


# ---------------------------------------------------
# 4. TRAIN ARIMA
# ---------------------------------------------------
print("\n🚀 Training ARIMA...")

arima_output = train_arima(df)

arima_mae_dict = arima_output["state_mae"]

arima_avg_mae = arima_output["avg_mae"]


# ---------------------------------------------------
# 5. TRAIN LSTM
# ---------------------------------------------------
print("\n🚀 Training LSTM...")

lstm_mae_dict = {}

for state in states:

    try:

        _, _, mae = train_lstm_model(
            df,
            state
        )

        lstm_mae_dict[state] = mae

    except Exception as e:

        print(f"❌ LSTM failed for {state}")

        print(e)

        lstm_mae_dict[state] = float("inf")


# ---------------------------------------------------
# 6. MODEL SELECTION
# ---------------------------------------------------
final_selection = {}

print("\n📊 Selecting Best Model Per State...\n")

for state in states:

    state_scores = {

        "XGBoost": xgb_mae_dict.get(
            state,
            float("inf")
        ),

        "Prophet": prophet_mae_dict.get(
            state,
            float("inf")
        ),

        "ARIMA": arima_mae_dict.get(
            state,
            float("inf")
        ),

        "LSTM": lstm_mae_dict.get(
            state,
            float("inf")
        )
    }

    # Select best model
    best_model = min(
        state_scores,
        key=state_scores.get
    )

    final_selection[state] = {

        "best_model": best_model,

        "scores": state_scores
    }

    print(
        f"✅ {state} → Best Model: {best_model}"
    )


# ---------------------------------------------------
# 7. SAVE MODEL SELECTION
# ---------------------------------------------------
with open(
    "models/final_model_selection.json",
    "w"
) as f:

    json.dump(
        final_selection,
        f,
        indent=4
    )

print("\n✅ Final model selection saved!")


# ---------------------------------------------------
# 8. SAVE METRICS SUMMARY
# ---------------------------------------------------
save_all_model_metrics(

    xgb_mae=xgb_avg_mae,

    prophet_mae=prophet_avg_mae,

    arima_mae=arima_avg_mae,

    lstm_mae=sum(
        lstm_mae_dict.values()
    ) / len(lstm_mae_dict)
)

print("\n🎉 ALL MODELS FINISHED!")