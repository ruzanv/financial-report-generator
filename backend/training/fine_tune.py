# resume training the saved LSTM with a lower LR
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
from backend.app.models.lstm_model import build_lstm

DATA = Path("data/rfsd_2015_2020.parquet")
MODEL_DIR = Path("backend/app/models")

seq_len = 4
features = ["B_revenue", "B_assets", "line_2110", "line_2120"]

df = pd.read_parquet(DATA).sort_values("year")
scaler = MinMaxScaler().fit(df[features])
scaled = scaler.transform(df[features])

X, y = [], []
for i in range(len(scaled) - seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len][0])
X, y = np.asarray(X), np.asarray(y)

model = build_lstm(seq_len, len(features))
model.load_weights(MODEL_DIR / "lstm_model.h5")
model.optimizer.lr.assign(1e-4)  # fine‑tune at lower LR

ckpt = ModelCheckpoint(MODEL_DIR / "lstm_model_ft.h5", save_best_only=True)
model.fit(X, y, epochs=20, batch_size=64, callbacks=[ckpt], validation_split=0.2)