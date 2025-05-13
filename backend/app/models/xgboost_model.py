from xgboost import XGBRegressor

def build_xgb(**kwargs) -> XGBRegressor:
    defaults = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    defaults.update(kwargs)
    return XGBRegressor(**defaults)