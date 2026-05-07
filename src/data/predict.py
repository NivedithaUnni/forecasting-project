import pandas as pd
import numpy as np


def predict_future(model, last_df, feature_columns, steps=8):

    print("\n🔮 Generating Future Predictions...\n")

    future_preds = []
    current = last_df.copy()

    history = list(current['total'].values)

    # Ensure datetime format
    current['date'] = pd.to_datetime(current['date'])

    for i in range(steps):

        # -----------------------------
        # Take last row
        # -----------------------------
        features = current.iloc[-1:].copy()

        # -----------------------------
        # IMPORTANT: align features
        # -----------------------------
        features = features.reindex(columns=feature_columns, fill_value=0)

        # -----------------------------
        # Predict
        # -----------------------------
        pred = model.predict(features)[0]
        future_preds.append(float(pred))

        # -----------------------------
        # Update history
        # -----------------------------
        history.append(pred)

        new_row = current.iloc[-1:].copy()

        new_row['total'] = pred

        # -----------------------------
        # Safe lag handling
        # -----------------------------
        new_row['lag_1'] = history[-2] if len(history) > 1 else pred
        new_row['lag_7'] = history[-8] if len(history) > 7 else pred
        new_row['lag_30'] = history[-31] if len(history) > 30 else pred

        temp_series = pd.Series(history)

        new_row['rolling_mean_7'] = temp_series.tail(7).mean()
        new_row['rolling_std_7'] = temp_series.tail(7).std()

        # -----------------------------
        # FIX DATE UPDATE (IMPORTANT)
        # -----------------------------
        last_date = current['date'].iloc[-1]
        new_date = last_date + pd.Timedelta(days=7)

        new_row['date'] = new_date

        # -----------------------------
        # Time features
        # -----------------------------
        new_row['day_of_week'] = new_date.dayofweek
        new_row['month'] = new_date.month

        # -----------------------------
        # Append safely
        # -----------------------------
        current = pd.concat([current, new_row], ignore_index=True)

    print("📊 Forecast:", future_preds)

    return future_preds