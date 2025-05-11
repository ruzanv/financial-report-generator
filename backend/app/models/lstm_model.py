from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_lstm(seq_len: int, n_features: int, units: int = 64):
    """Return a compiled LSTM model; used offline during training or fine‑tune"""
    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_features)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model