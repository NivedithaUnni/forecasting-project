import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error


# -----------------------------
# 1. TRAIN + EVALUATE ARIMA
# -----------------------------
def train_arima(df):

    print("\n📈 Training ARIMA Model (State-wise)...\n")

    state_mae = {}
    state_models = {}

    df = df.copy()

    # =========================
    # GLOBAL CLEANING
    # =========================
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['total'] = pd.to_numeric(df['total'], errors='coerce')

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['date', 'total'])
    df = df.sort_values('date')

    # -----------------------------
    # STATE LOOP
    # -----------------------------
    for state in df['state'].unique():

        state_df = df[df['state'] == state].copy()

        if state_df.empty:
            continue

        # time series setup
        data = state_df.set_index('date')['total']

        # enforce weekly frequency
        data = data.asfreq('W')

        # fill missing values (critical for ARIMA stability)
        data = data.interpolate(method='linear').ffill().bfill()

        if len(data) < 20:
            print(f"⚠️ Skipping {state} (insufficient data)")
            continue

        split = int(len(data) * 0.8)

        train = data.iloc[:split]
        test = data.iloc[split:]

        # -----------------------------
        # ARIMA GRID SEARCH
        # -----------------------------
        best_aic = np.inf
        best_model = None
        best_order = None

        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):

                    try:
                        model = ARIMA(train, order=(p, d, q))
                        model_fit = model.fit()

                        if model_fit.aic < best_aic:
                            best_aic = model_fit.aic
                            best_model = model_fit
                            best_order = (p, d, q)

                    except:
                        continue

        if best_model is None:
            print(f"{state} → ARIMA failed")
            continue

        # -----------------------------
        # FORECAST
        # -----------------------------
        forecast = best_model.forecast(steps=len(test))

        test_values = np.array(test)
        forecast_values = np.array(forecast)

        # safe mask (extra stability)
        mask = np.isfinite(test_values) & np.isfinite(forecast_values)

        test_values = test_values[mask]
        forecast_values = forecast_values[mask]

        mae = mean_absolute_error(test_values, forecast_values) if len(test_values) > 0 else np.nan

        state_mae[state] = mae
        state_models[state] = best_model

        print(f"{state} → MAE: {mae:.4f} | Best order: {best_order}")

    # -----------------------------
    # FINAL SCORE
    # -----------------------------
    avg_mae = np.nanmean(list(state_mae.values())) if state_mae else np.nan

    print(f"\n📊 Average ARIMA MAE: {avg_mae:.4f}")

    return {
        "state_mae": state_mae,
        "avg_mae": avg_mae,
        "models": state_models
    }


# -----------------------------
# 2. FUTURE FORECAST (8 WEEKS)
# -----------------------------
def forecast_arima(model, steps=8):

    forecast = model.forecast(steps=steps)

    forecast = np.asarray(forecast)

    # strict safety cleanup
    forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)

    return forecast.tolist()