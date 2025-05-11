"""Incremental training of XGBoost on RFSD 2015‑2020 slice.
* Полная тренировка первый раз, далее ⬆️ доучивание + добавочные деревья.
* 5‑fold CV (RMSE, MAE) только при первом запуске; при доучивании – пропускаем.
* Модель хранится в `backend/app/models/xgb_model.joblib`; скрипт
  автоматически загружает её и добавляет ещё `ADDITIONAL_TREES`.
* Метрики финальной модели + CV (если было) логируются в `logs/xgb_runs.csv`.
* Максимум памяти < 1 GB благодаря Polars scan.
"""
from pathlib import Path
import logging, time, json
from pythonjsonlogger import jsonlogger
import polars as pl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, DMatrix, cv
import joblib

# constants
ADDITIONAL_TREES = 500
PARAMS = dict(
    objective="reg:squarederror",
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
)

# logger
logger = logging.getLogger("train_xgb")
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

DATA_DIR = Path("data/rfsd_2015_2020")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = Path("logs"); LOGS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "xgb_model.joblib"

logger.info("Loading data", extra={"file": str(DATA_DIR)})

cols = ["line_1600", "line_2110", "line_2120", "line_2400"]
df_pl = pl.scan_parquet(str(DATA_DIR / "*.parquet")).select(cols).collect()
logger.info("Converting to pandas …")
df = df_pl.to_pandas()

# clean NaN / Inf
initial = len(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=cols, inplace=True)
logger.info("Dropped rows", extra={"removed": initial - len(df)})

X = df[["line_1600", "line_2110", "line_2120"]]
y = df["line_2400"]

# determine mode: fresh vs incremental
incremental = MODEL_PATH.exists()
cv_metrics = {}

if not incremental:
    # ── first time: run 5‑fold CV ------------------------------------
    logger.info("Running 5‑fold CV …")
    dtrain = DMatrix(X, label=y)
    cv_res = cv(PARAMS, dtrain, num_boost_round=ADDITIONAL_TREES, nfold=5,
                metrics=["rmse", "mae"], seed=42, verbose_eval=True)
    cv_metrics = {
        "RMSE_CV_mean": cv_res["test-rmse-mean"].iloc[-1],
        "RMSE_CV_std":  cv_res["test-rmse-std"].iloc[-1],
        "MAE_CV_mean":  cv_res["test-mae-mean"].iloc[-1],
        "MAE_CV_std":   cv_res["test-mae-std"].iloc[-1],
    }
    logger.info("CV finished", extra=cv_metrics)

# ── build / load model ----------------------------------------------
if incremental:
    logger.info("Loading existing model and adding trees", extra={"add": ADDITIONAL_TREES})
    base_model = joblib.load(MODEL_PATH)
    n_total = base_model.n_estimators + ADDITIONAL_TREES
    model = XGBRegressor(n_estimators=n_total, **PARAMS, n_jobs=-1, random_state=42)
    sample_idx = np.random.RandomState(42).choice(len(X), size=min(10_000, len(X)), replace=False)
    eval_X, eval_y = X.iloc[sample_idx], y.iloc[sample_idx]
    model.fit(
        X,
        y,
        xgb_model=base_model.get_booster(),
        eval_set=[(eval_X, eval_y)],
        verbose=50,
    )
else:
    logger.info("Training model from scratch", extra={"trees": ADDITIONAL_TREES})
    model = XGBRegressor(n_estimators=ADDITIONAL_TREES, **PARAMS, n_jobs=1)
    model.fit(X, y, verbose=True)

# ── evaluate on full dataset ----------------------------------------
preds = model.predict(X)
try:
    rmse_full = mean_squared_error(y, preds, squared=False)
except TypeError:
    rmse_full = np.sqrt(mean_squared_error(y, preds))
mae_full  = mean_absolute_error(y, preds)
r2_full   = r2_score(y, preds)
metrics = {
    "RMSE_full": rmse_full,
    "MAE_full":  mae_full,
    "R2_full":   r2_full,
    **cv_metrics,
}
logger.info("Final metrics", extra=metrics)

# ── append log -------------------------------------------------------
row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics, "params": json.dumps(model.get_params())}
log_path = LOGS_DIR / "xgb_runs.csv"
pd.DataFrame([row]).to_csv(log_path, mode="a", index=False, header=not log_path.exists())

# ── save model -------------------------------------------------------
joblib.dump(model, MODEL_PATH)
logger.info("Model saved", extra={"path": str(MODEL_PATH)})