from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import holidays

def train_prophet(df):

    print("\n📈 Training Prophet Model (State-wise)...\n")

    results = {}

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["date"] = pd.to_datetime(df["date"])
    df["total"] = pd.to_numeric(df["total"])
    df = df.dropna(subset=["date", "total"])
    df = df.sort_values("date")

    # -----------------------------
    # HOLIDAYS (IMPORTANT FIX)
    # -----------------------------
    ind_holidays = holidays.India()

    holiday_df = pd.DataFrame({
        "ds": list(ind_holidays.keys()),
        "holiday": "india_holiday"
    })

    for state in df["state"].unique():

        state_df = df[df["state"] == state].copy()

        if len(state_df) < 20:
            continue

        if state_df["total"].nunique() == 1:
            continue

        data = state_df[["date", "total"]].rename(columns={
            "date": "ds",
            "total": "y"
        })

        split = int(len(data) * 0.8)
        train = data.iloc[:split]
        test = data.iloc[split:]

        model = Prophet(holidays=holiday_df)

        model.fit(train)

        future = model.make_future_dataframe(periods=len(test), freq="D")
        forecast = model.predict(future)

        preds = forecast["yhat"].iloc[-len(test):].values

        mae = mean_absolute_error(test["y"], preds)
        results[state] = mae

        print(f"{state} → MAE: {mae:.4f}")

    avg_mae = np.mean(list(results.values())) if results else float("inf")

    print(f"\n✅ Average Prophet MAE: {avg_mae:.4f}")

    return results, avg_mae