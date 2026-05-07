import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os
import holidays

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    df = df.copy()
    df = df.sort_values(["state", "date"])

    df["date"] = pd.to_datetime(df["date"])

    # =========================
    # HOLIDAY FEATURE (ADDED)
    # =========================
    ind_holidays = holidays.India()
    df["is_holiday"] = df["date"].dt.date.apply(
        lambda x: 1 if x in ind_holidays else 0
    )

    # LAGS
    df["lag_1"] = df.groupby("state")["total"].shift(1)
    df["lag_7"] = df.groupby("state")["total"].shift(7)
    df["lag_30"] = df.groupby("state")["total"].shift(30)

    # ROLLING FEATURES
    df["roll_mean_7"] = df.groupby("state")["total"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    df["roll_std_7"] = df.groupby("state")["total"].transform(
        lambda x: x.shift(1).rolling(7).std()
    )

    # TIME FEATURES
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    return df

# -----------------------------
# TRAIN STATE-WISE MODEL
# -----------------------------
def train_xgboost(df):

    print("\n🚀 Training XGBoost (STATE-WISE Time-Series)...\n")

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df = create_features(df)

    print("\n📊 FEATURE ENGINEERING CHECK")

    print("\n👉 Columns in dataset:")
    print(df.columns.tolist())

    print("\n👉 Sample rows:")
    print(df.head())

    print("\n👉 Missing values per column:")
    print(df.isna().sum())

    required_features = [
        "lag_1", "lag_7", "lag_30",
        "roll_mean_7", "roll_std_7",
        "dayofweek", "month", "is_holiday"
    ]

    print("\n👉 Feature validation:")
    for f in required_features:
        print(f, "✔" if f in df.columns else "❌ MISSING")

    features = required_features

    results = {}

    os.makedirs("models/saved_models/xgb", exist_ok=True)

    # -----------------------------
    # STATE-WISE TRAINING
    # -----------------------------
    for state in df["state"].unique():

        df_state = df[df["state"] == state].copy()

        # drop AFTER split preparation (safe approach)
        df_state = df_state.dropna()

        if len(df_state) < 50:
            print(f"⚠️ Skipping {state} (not enough data)")
            continue

        X = df_state[features]
        y = df_state["total"]

        split = int(len(df_state) * 0.8)

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        results[state] = mae

        print(f"{state} → XGBoost MAE: {mae:.2f}")

        joblib.dump(model, f"models/saved_models/xgb/{state}.pkl")

    avg_mae = sum(results.values()) / len(results)

    print(f"\n📊 Average XGBoost MAE: {avg_mae:.2f}")

    return {
        "state_mae": results,
        "avg_mae": avg_mae,
        "feature_columns": features
    }