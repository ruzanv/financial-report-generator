from docx import Document
from pathlib import Path
import pandas as pd
from datetime import datetime


def build_report(df: pd.DataFrame, preds: dict, path: Path):
    doc = Document()
    doc.add_heading("Financial Forecast Report", level=1)
    doc.add_paragraph(f"Generated: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")

    doc.add_heading("Input Summary", level=2)
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = col
    for _, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = f"{value:,.2f}"

    doc.add_heading("Model Predictions", level=2)
    doc.add_paragraph("XGBoost Predictions:")
    doc.add_paragraph(str(preds["xgboost"]))
    doc.add_paragraph("LSTM Predictions:")
    #doc.add_paragraph(str(preds["lstm"]))

    doc.save(path)