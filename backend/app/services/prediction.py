import os
# ограничиваем OpenMP/MKL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import joblib
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from xgboost import DMatrix
from celery.signals import worker_process_init
from ..config import settings

WINDOW = 4

# ограничим TF-потоки
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# JSON-логгер
logger = logging.getLogger("prediction")
if not logger.handlers:
    from pythonjsonlogger import jsonlogger
    h = logging.StreamHandler()
    h.setFormatter(jsonlogger.JsonFormatter())
    logger.addHandler(h)
logger.setLevel(logging.INFO)

MODEL_DIR = Path(settings.model_dir)

xgb_path = MODEL_DIR / "xgb_model.joblib"
if not xgb_path.exists():
    logger.error("XGB model not found", extra={"path": str(xgb_path)})
    raise FileNotFoundError(f"{xgb_path} missing")
logger.info("Loading XGB model", extra={"path": str(xgb_path)})
_xgb = joblib.load(xgb_path)
_booster = _xgb.get_booster()

_lstm = None
_lstm_warmed = False

@worker_process_init.connect
def init_worker(**kwargs):
    global _lstm, _lstm_warmed
    # загружаем модель внутри каждого воркера после fork
    lstm_path = MODEL_DIR / "lstm_model.h5"
    if not lstm_path.exists():
        logger.error("LSTM model not found", extra={"path": str(lstm_path)})
        raise FileNotFoundError(f"{lstm_path} missing")
    logger.info("Loading LSTM model in worker", extra={"path": str(lstm_path)})
    _lstm = load_model(str(lstm_path), compile=False)
    # прогрев для быстрого первого запуска
    dummy = np.zeros((1, WINDOW, 1), dtype=np.float32)
    _lstm.predict(dummy, batch_size=1, verbose=0)
    _lstm_warmed = True
    logger.info("LSTM warmed up in worker", extra={})

def predict_financials(df):
    # проверяем поля
    req = ["line_1600", "line_2110", "line_2120", "line_2400"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Starting predictions", extra={"shape": df.shape})

    X_xgb = df[req[:3]].astype("float32")
    logger.info("XGB.predict", extra={"n_rows": len(X_xgb)})
    dm = DMatrix(X_xgb)
    xgb_preds = _booster.predict(dm)
    logger.info("XGB done", extra={"n_preds": len(xgb_preds)})

    global _lstm, _lstm_warmed
    if _lstm is None:
        raise RuntimeError("LSTM model not initialized in worker")
    series = df["line_2400"].astype("float32").values
    # нормализация временного ряда для LSTM
    min_val = float(series.min())
    max_val = float(series.max())
    if max_val - min_val != 0:
        series_scaled = (series - min_val) / (max_val - min_val)
    else:
        series_scaled = np.zeros_like(series)
    if len(series) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} points for LSTM, got {len(series)}")
    seqs = [
        series_scaled[i: i + WINDOW].reshape(WINDOW, 1)
        for i in range(len(series_scaled) - WINDOW + 1)
    ]
    X_lstm = np.stack(seqs, axis=0)
    logger.info("LSTM.predict", extra={"shape": X_lstm.shape})
    lstm_out_scaled = _lstm.predict(X_lstm, batch_size=32, verbose=0).squeeze()
    lstm_out = lstm_out_scaled * (max_val - min_val) + min_val
    logger.info("LSTM done", extra={"n_preds": int(np.atleast_1d(lstm_out).shape[0])})

    return {
        "xgboost": xgb_preds.tolist(),
        "lstm": lstm_out.tolist(),
    }
