from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    # FastAPI
    api_v1_prefix: str = "/api/v1"
    allowed_origins: list[str] = ["*"]  # CORS – tighten in prod

    # Files & storage
    upload_dir: Path = Path("/code/uploads")
    model_dir: Path = Path("models")
    reports_dir: Path = Path("/code/reports")

    # Celery / Redis
    redis_url: str = "redis://redis:6379/0"
    result_backend: str = redis_url

    # Training
    rfsd_url: str = "https://huggingface.co/datasets/sigoy/RFSD/resolve/main/rfsd.parquet"
    start_year: int = 2015
    end_year: int = 2020
    batch_size: int = 64

    class Config:
        env_file = ".env"

settings = Settings()