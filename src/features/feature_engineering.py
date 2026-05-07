import pandas as pd
import numpy as np

# -----------------------------
# 1. BASIC FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    df = df.copy()

    # Ensure sorted (VERY IMPORTANT for time series)
    df = df.sort_values(['state', 'date'])

    # -----------------------------
    # Calendar features
    # -----------------------------
    df['dayofweek'] = df['date'].dt.dayofweek   # 0=Mon, 6=Sun
    df['month'] = df['date'].dt.month

    # Simple holiday flag (you can expand later with real calendar)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # -----------------------------
    # Lag features (CRITICAL)
    # -----------------------------
    df['lag_1'] = df.groupby('state')['sales'].shift(1)
    df['lag_7'] = df.groupby('state')['sales'].shift(7)
    df['lag_30'] = df.groupby('state')['sales'].shift(30)

    # -----------------------------
    # Rolling features (trend capture)
    # -----------------------------
    df['rolling_mean_7'] = (
        df.groupby('state')['sales']
        .shift(1)
        .rolling(window=7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['rolling_std_7'] = (
        df.groupby('state')['sales']
        .shift(1)
        .rolling(window=7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # -----------------------------
    # Handle missing values created by lagging
    # -----------------------------
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df