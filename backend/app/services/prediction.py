import os
# ограничиваем потоки OpenMP / MKL, чтобы не было конфликтов в Celery
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import joblib
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
from xgboost import DMatrix
from ..config import settings

# ── параметры окна для LSTM ────────────────────────────
WINDOW = 4

# ── ограничим TF-потоки ────────────────────────────────
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ── логгер ──────────────────────────────────────────────
logger = logging.getLogger("prediction")
if not logger.handlers:
    handler = logging.StreamHandler()
    from pythonjsonlogger import jsonlogger
    handler.setFormatter(jsonlogger.JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ── загрузим модели при импорте ─────────────────────────
MODEL_DIR = Path(settings.model_dir)

# XGBoost
xgb_path = MODEL_DIR / "xgb_model.joblib"
if not xgb_path.exists():
    logger.error("XGB model not found", extra={"path": str(xgb_path)})
    raise FileNotFoundError(f"{xgb_path} missing")
logger.info("Loading XGB model", extra={"path": str(xgb_path)})
_xgb = joblib.load(xgb_path)
_booster = _xgb.get_booster()

# LSTM
lstm_path = MODEL_DIR / "lstm_model.h5"
if not lstm_path.exists():
    logger.error("LSTM model not found", extra={"path": str(lstm_path)})
    raise FileNotFoundError(f"{lstm_path} missing")
logger.info("Loading LSTM model", extra={"path": str(lstm_path)})
_lstm = load_model(str(lstm_path), compile=False)
# прогрев LSTM для первого быстрого predict
_dummy = np.zeros((1, WINDOW, 1), dtype=np.float32)
_lstm.predict(_dummy, batch_size=1, verbose=0)


def predict_financials(df):
    """
    Принимает DataFrame с колонками line_1600, line_2110, line_2120, line_2400.
    Возвращает предсказания XGBoost (точечные) и LSTM (скользящее окно).
    """
    # проверка наличия всех колонок
    required = ["line_1600", "line_2110", "line_2120", "line_2400"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info("Starting predictions", extra={"shape": df.shape})

    # ── XGBoost ─────────────────────────────────────────────────
    X_xgb = df[["line_1600", "line_2110", "line_2120"]].astype("float32")
    logger.info("XGB.predict", extra={"n_rows": len(X_xgb)})
    dm = DMatrix(X_xgb)
    xgb_preds = _booster.predict(dm)
    logger.info("XGB done", extra={"n_preds": len(xgb_preds)})

    # ── LSTM ────────────────────────────────────────────────────
    series = df["line_2400"].astype("float32").values
    if len(series) < WINDOW:
        raise ValueError(f"Need at least {WINDOW} points for LSTM, got {len(series)}")
    # составляем скользящие окна
    seqs = [series[i : i + WINDOW].reshape(WINDOW, 1)
            for i in range(len(series) - WINDOW + 1)]
    X_lstm = np.stack(seqs, axis=0)
    logger.info("LSTM.predict", extra={"shape": X_lstm.shape})
    #lstm_out = _lstm.predict(X_lstm, batch_size=32, verbose=0).squeeze()
   # logger.info("LSTM done", extra={"n_preds": lstm_out.shape[0]})

    return {
        "xgboost": xgb_preds.tolist(),
        #"lstm": lstm_out.tolist(),
    }
