from celery import Celery
from .config import settings
from .tasks import generate_report

celery_app = Celery(
    "financial_report_generator",
    broker=settings.redis_url,
    backend=settings.result_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)