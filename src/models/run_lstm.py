from lstm_model import train_lstm_model, forecast_lstm, prepare_state_data
import pandas as pd

df = pd.read_csv("data/cleaned_data.csv")

states = df['state'].unique()

results = {}

for state in states:
    print(f"\n🚀 Training LSTM for {state}")

    model, scaler = train_lstm_model(df, state)

    data = prepare_state_data(df, state)

    forecast = forecast_lstm(model, scaler, data)

    results[state] = forecast

    print(f"{state} next 8-week forecast:")
    print(forecast)