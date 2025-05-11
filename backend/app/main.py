from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from uuid import uuid4
from pathlib import Path
from .config import settings
from .tasks.generate_report import generate_report_task
from .celery_app import celery_app

app = FastAPI(title="Financial Report Generator with Forecasting")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

@app.post("/upload", status_code=202)
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    file_id = uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}.csv"
    content = await file.read()
    dest.write_bytes(content)

    task = generate_report_task.delay(str(dest))
    return {"task_id": task.id}

@app.get("/status/{task_id}")
def task_status(task_id: str):  # local import to avoid circular
    res = celery_app.AsyncResult(task_id)
    if res.state == "PENDING":
        return {"state": "PENDING"}
    if res.state == "SUCCESS":
        return {"state": "SUCCESS", "download_url": f"/download/{res.result}"}
    if res.state == "FAILURE":
        return JSONResponse(status_code=500, content={"state": "FAILURE", "error": str(res.info)})
    return {"state": res.state}

@app.get("/download/{report_name}")
def download_report(report_name: str):
    report_path = Path(settings.reports_dir) / report_name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(str(report_path), media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=report_name)