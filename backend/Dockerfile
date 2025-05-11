# Build context = ./backend  (set in docker‑compose.yml)
FROM python:3.11-slim
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /code/backend
ENV PYTHONPATH=/code
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]