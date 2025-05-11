import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from app.services import utils
from app import config

data_df = pd.read_csv("data/financial_data.csv")
data_df = data_df[utils.EXPECTED_LSTM_COLUMNS]
scaled_df = data_df.copy()

data_values = scaled_df.values.astype(float)
n_steps = 3
X, y = [], []
for i in range(len(data_values) - n_steps):
    X.append(data_values[i:i+n_steps])
    y.append(data_values[i+n_steps])
X = np.array(X)
y = np.array(y)

model = keras.Sequential([
    layers.LSTM(50, activation="relu", input_shape=(n_steps, data_values.shape[1])),
    layers.Dense(data_values.shape[1])  # выходной слой с 4 нейронами (прогноз каждого из 4 показателей)
])
model.compile(optimizer="adam", loss="mse")

history = model.fit(X, y, epochs=50, batch_size=1, verbose=2)

model.save("models/lstm_model.h5")
