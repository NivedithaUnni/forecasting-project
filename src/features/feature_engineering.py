import pandas as pd
import numpy as np


# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
def create_features(df):

    df = df.copy()

    # ---------------------------------------------------
    # SORTING (VERY IMPORTANT)
    # ---------------------------------------------------
    df = df.sort_values(['state', 'date'])

    # ---------------------------------------------------
    # CALENDAR FEATURES
    # ---------------------------------------------------
    df['dayofweek'] = df['date'].dt.dayofweek

    df['month'] = df['date'].dt.month

    # Weekend / holiday proxy
    df['is_holiday'] = (
        df['dayofweek']
        .isin([5, 6])
        .astype(int)
    )

    # ---------------------------------------------------
    # LAG FEATURES
    # ---------------------------------------------------
    df['lag_1'] = (
        df.groupby('state')['sales']
        .shift(1)
    )

    df['lag_7'] = (
        df.groupby('state')['sales']
        .shift(7)
    )

    df['lag_30'] = (
        df.groupby('state')['sales']
        .shift(30)
    )

    # ---------------------------------------------------
    # ROLLING FEATURES
    # ---------------------------------------------------
    df['roll_mean_7'] = (
        df.groupby('state')['sales']
        .shift(1)
        .rolling(window=7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df['roll_std_7'] = (
        df.groupby('state')['sales']
        .shift(1)
        .rolling(window=7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # ---------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------
    df = df.bfill().ffill()

    return df