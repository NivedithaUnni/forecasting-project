import numpy as np
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model


# ---------------------------------------------------
# 1. PREPARE STATE DATA
# ---------------------------------------------------
def prepare_state_data(df, state):

    df_state = df[df['state'] == state].copy()

    df_state = df_state.sort_values('date')

    return df_state[['total']].values


# ---------------------------------------------------
# 2. CREATE SEQUENCES
# ---------------------------------------------------
def create_sequences(data, window_size=30):

    X = []
    y = []

    for i in range(window_size, len(data)):

        X.append(data[i - window_size:i, 0])

        y.append(data[i, 0])

    X = np.array(X)

    y = np.array(y)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y


# ---------------------------------------------------
# 3. TRAIN LSTM MODEL
# ---------------------------------------------------
def train_lstm_model(
    df,
    state,
    window_size=30,
    epochs=10
):

    print(f"\n🚀 Training LSTM for {state}")

    # ---------------------------------------------------
    # PREPARE DATA
    # ---------------------------------------------------
    data = prepare_state_data(df, state)

    # ---------------------------------------------------
    # SCALING
    # ---------------------------------------------------
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    # ---------------------------------------------------
    # CREATE SEQUENCES
    # ---------------------------------------------------
    X, y = create_sequences(
        scaled_data,
        window_size
    )

    # ---------------------------------------------------
    # TRAIN / TEST SPLIT
    # ---------------------------------------------------
    split = int(len(X) * 0.8)

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    # ---------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------
    model = Sequential([

        LSTM(
            64,
            return_sequences=True,
            input_shape=(window_size, 1)
        ),

        Dropout(0.2),

        LSTM(64),

        Dropout(0.2),

        Dense(1)
    ])

    # ---------------------------------------------------
    # COMPILE
    # ---------------------------------------------------
    model.compile(
        optimizer='adam',
        loss='mse'
    )

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ---------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------
    y_pred = model.predict(
        X_test,
        verbose=0
    )

    # Inverse transform
    y_test_inv = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    y_pred_inv = scaler.inverse_transform(
        y_pred
    )

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------
    mae = mean_absolute_error(
        y_test_inv,
        y_pred_inv
    )

    print(f"✅ LSTM {state} MAE: {mae:.4f}")

    # ---------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------
    save_dir = "models/saved_models/lstm"

    os.makedirs(save_dir, exist_ok=True)

    # Save keras model
    model.save(
        f"{save_dir}/{state}.keras"
    )

    # Save scaler separately
    joblib.dump(
        scaler,
        f"{save_dir}/{state}_scaler.pkl"
    )

    print(f"✅ Saved LSTM model for {state}")

    return model, scaler, mae


# ---------------------------------------------------
# 4. FORECAST FUTURE
# ---------------------------------------------------
def forecast_lstm(
    model,
    scaler,
    data,
    window_size=30,
    steps=8
):

    # ---------------------------------------------------
    # SCALE DATA
    # ---------------------------------------------------
    scaled = scaler.transform(data)

    # Last sequence
    input_seq = scaled[-window_size:]

    predictions = []

    # ---------------------------------------------------
    # RECURSIVE FORECASTING
    # ---------------------------------------------------
    for _ in range(steps):

        X_input = input_seq.reshape(
            1,
            window_size,
            1
        )

        pred = model.predict(
            X_input,
            verbose=0
        )[0][0]

        predictions.append(pred)

        # Update sequence
        input_seq = np.append(
            input_seq[1:],
            pred
        ).reshape(-1, 1)

    # ---------------------------------------------------
    # INVERSE SCALE
    # ---------------------------------------------------
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions.flatten().tolist()


# ---------------------------------------------------
# 5. LOAD SAVED MODEL
# ---------------------------------------------------
def load_lstm_model(state):

    model_path = (
        f"models/saved_models/lstm/{state}.keras"
    )

    scaler_path = (
        f"models/saved_models/lstm/{state}_scaler.pkl"
    )

    model = load_model(model_path)

    scaler = joblib.load(scaler_path)

    return model, scaler