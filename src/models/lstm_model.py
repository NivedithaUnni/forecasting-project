import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


# ----------------------------
# 1. Prepare state-wise data
# ----------------------------
def prepare_state_data(df, state):
    df_state = df[df['state'] == state].copy()
    df_state = df_state.sort_values('date')
    return df_state[['total']].values


# ----------------------------
# 2. Create sequences
# ----------------------------
def create_sequences(data, window_size=30):
    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])

    X = np.array(X)
    y = np.array(y)

    return X.reshape(X.shape[0], X.shape[1], 1), y


# ----------------------------
# 3. Train LSTM model
# ----------------------------
def train_lstm_model(df, state, window_size=30, epochs=10):

    # Prepare data
    data = prepare_state_data(df, state)

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, window_size)

    # Train-test split (time series logic)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),

        LSTM(64),
        Dropout(0.2),

        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ----------------------------
    # PREDICTION + MAE
    # ----------------------------
    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_test_inv, y_pred_inv)

    # ----------------------------
    # SAVE MODEL (IMPORTANT FIX)
    # ----------------------------
    os.makedirs("models/saved_models/lstm", exist_ok=True)

    joblib.dump(
        (model, scaler),
        f"models/saved_models/lstm/{state}.pkl"
    )

    print(f"✅ LSTM {state} MAE: {mae:.4f}")

    return model, scaler, mae


# ----------------------------
# 4. Forecast future
# ----------------------------
def forecast_lstm(model, scaler, data, window_size=30, steps=8):

    scaled = scaler.transform(data)

    input_seq = scaled[-window_size:]
    predictions = []

    for _ in range(steps):

        X_input = input_seq.reshape(1, window_size, 1)

        pred = model.predict(X_input, verbose=0)[0][0]
        predictions.append(pred)

        # update sequence
        input_seq = np.append(input_seq[1:], pred).reshape(-1, 1)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten().tolist()