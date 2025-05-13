import pandas as pd
from celery import shared_task
from uuid import uuid4
from pathlib import Path
import logging
from ..config import settings
from ..services.preprocessing import validate_and_prepare
from ..services.prediction import predict_financials
from ..services.docx_generator import create_report

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@shared_task(bind=True, name="generate_report")
def generate_report_task(self, csv_path: str) -> str:
    # celery task
    try:
        logger.info(f"[TASK] Starting report generation for file: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"[TASK] CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        df_prepared = validate_and_prepare(df)
        logger.info("[TASK] Data validated and preprocessed")


        logger.info("[TASK] Starting prediction", extra={"shape": df_prepared.shape})
        preds = predict_financials(df_prepared)
        logger.info("[TASK] Predictions completed")

        report_name = f"report_{uuid4().hex}.docx"
        report_path = Path(settings.reports_dir) / report_name
        report_path.parent.mkdir(exist_ok=True, parents=True)

        create_report(df_prepared, preds, report_path)
        logger.info(f"[TASK] Report saved to: {report_path}")

        return report_name

    except Exception as exc:
        logger.exception("[TASK] Error occurred during report generation, retrying…")
        self.retry(exc=exc, countdown=10, max_retries=3)
