from src.models.lstm_model import train_lstm_model, prepare_state_data, create_sequences
from sklearn.metrics import mean_absolute_error
import numpy as np


def evaluate_lstm_all_states(df):

    states = df['state'].unique()
    maes = []

    for state in states:
        print(f"\n🚀 Evaluating LSTM for {state}")

        model, scaler = train_lstm_model(df, state)

        data = prepare_state_data(df, state)

        scaled = scaler.transform(data)

        X, y = create_sequences(scaled, window_size=30)

        split = int(len(X) * 0.8)
        X_test, y_test = X[split:], y[split:]

        preds = model.predict(X_test)

        preds = scaler.inverse_transform(preds)
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(y_true, preds)

        maes.append(mae)

        print(f"{state} MAE: {mae:.2f}")

    return sum(maes) / len(maes)