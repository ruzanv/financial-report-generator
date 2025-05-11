import pandas as pd
from typing import List

EXPECTED_COLS: List[str] = [
    "line_1600", "line_2110", "line_2120", "line_2400"
]

def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    if not set(EXPECTED_COLS).issubset(df.columns):
        missing = set(EXPECTED_COLS) - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    if df.isnull().any().any():
        df = df.fillna(0)
    return df[EXPECTED_COLS].astype(float)