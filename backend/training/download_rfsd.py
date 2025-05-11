from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
from backend.app.config import settings

REPO = "irlspbru/RFSD"
TEMPLATE = "RFSD/year={year}/part-0.parquet"
DEST_DIR = Path("data/rfsd_2015_2020")
DEST_DIR.mkdir(parents=True, exist_ok=True)

for year in range(settings.start_year, settings.end_year + 1):
    dest_file = DEST_DIR / f"year={year}.parquet"
    if dest_file.exists():
        print(f"✔ {dest_file.name} already present, skipping")
        continue

    print(f"Downloading year {year} …")
    src_path = hf_hub_download(
        repo_id=REPO,
        filename=TEMPLATE.format(year=year),
        repo_type="dataset",
        resume_download=True,
    )
    shutil.copy2(src_path, dest_file)
    try:
        rel = dest_file.relative_to(Path.cwd())
    except ValueError:
        rel = dest_file
    print(f"  → saved {rel}")

print("All shards copied →", DEST_DIR)