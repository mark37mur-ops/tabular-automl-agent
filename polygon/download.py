"""Download the open polygon datasets from OpenML into polygon/raw/<openml_id>/data.pq.

Usage: python polygon/download.py [task names...]   (default: all tasks from build_tasks.TASKS)
Files that already exist are skipped; OpenML often answers 503, so each file is retried.
Standard library only.
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_tasks import TASKS  # noqa: E402

ATTEMPTS = 5


def parquet_url(openml_id: int) -> str:
    """OpenML stores dataset parquet files in buckets of 10 000 ids: 4135 -> 0000, 42712 -> 0004."""
    return f"https://data.openml.org/datasets/{openml_id // 10000:04d}/{openml_id}/dataset_{openml_id}.pq"


def download(openml_id: int) -> Path:
    target = ROOT / "raw" / str(openml_id) / "data.pq"
    if target.exists() and target.stat().st_size > 0:
        print(f"{openml_id}: already downloaded")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(".part")
    url = parquet_url(openml_id)
    print(f"{openml_id}: {url}")
    for attempt in range(1, ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            break
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == ATTEMPTS:
                raise
            print(f"{openml_id}: {error}; retry {attempt}/{ATTEMPTS - 1} in {10 * attempt}s")
            time.sleep(10 * attempt)
    part.replace(target)
    print(f"{openml_id}: {target.stat().st_size / 2**20:.1f} MB")
    return target


if __name__ == "__main__":
    only = set(sys.argv[1:])
    failed = []
    for spec in TASKS:
        if not only or spec["name"] in only:
            try:
                download(spec["openml_id"])
            except Exception as error:  # keep going: one unavailable dataset must not block the others
                print(f"{spec['name']}: download failed: {error}")
                failed.append(spec["name"])
    sys.exit(1 if failed else 0)
