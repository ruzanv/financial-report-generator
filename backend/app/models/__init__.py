from pathlib import Path
from joblib import load as _jload
from tensorflow.keras.models import load_model as _kload

MODEL_DIR = Path(__file__).resolve().parent

_xgb_path = MODEL_DIR / "xgb_model.joblib"
_lstm_path = MODEL_DIR / "lstm_model.h5"

xgb_model = _jload(_xgb_path) if _xgb_path.exists() else None
lstm_model = _kload(_lstm_path, compile=False) if _lstm_path.exists() else None