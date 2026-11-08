"""Dress rehearsal of a submission: run a built ZIP exactly as the platform does.

For each task folder (task.md + data in any layout) the script copies the task to a fresh
dir, puts ONLY root *.py files and requirements.txt from the ZIP next to task.md, runs
`python solution.py task.md` there and checks submission.csv against test.csv and the
sample: columns, row count, id order, missing values, allowed labels.

Usage:
  python agent/scripts/platform_check.py path/to/solution.zip path/to/task_dir [more task dirs]
         [--time-limit 1800] [--env-file llm.env] [--set KEY=VALUE]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import read_env_file  # noqa: E402


def find(task_dir: Path, name: str) -> Path:
    found = sorted(task_dir.rglob(name), key=lambda p: len(p.parts))
    if not found:
        raise FileNotFoundError(name)
    return found[0]


def check_submission(task_dir: Path) -> list[str]:
    sub_path = task_dir / "submission.csv"
    if not sub_path.exists():
        return ["submission.csv is missing next to task.md"]
    sub = pd.read_csv(sub_path)
    sample = pd.read_csv(find(task_dir, "sample_submission.csv"))
    test = pd.read_csv(find(task_dir, "test.csv"))
    train = pd.read_csv(find(task_dir, "train.csv"), usecols=[sample.columns[1]]) \
        if sample.columns[1] in pd.read_csv(find(task_dir, "train.csv"), nrows=0).columns else None
    errors = []
    if list(sub.columns) != list(sample.columns):
        errors.append(f"columns {list(sub.columns)} != sample {list(sample.columns)}")
        return errors
    id_col, target = sample.columns[0], sample.columns[1]
    if len(sub) != len(test):
        errors.append(f"{len(sub)} rows, test has {len(test)}")
    elif id_col in test.columns and list(sub[id_col]) != list(test[id_col]):
        errors.append("ids differ from test.csv order")
    if sub[id_col].duplicated().any():
        errors.append("duplicate ids")
    if sub[target].isna().any():
        errors.append(f"{int(sub[target].isna().sum())} missing predictions")
    if train is not None and not pd.api.types.is_float_dtype(sub[target]):
        unknown = set(sub[target]) - set(train[target].dropna())
        if unknown:
            errors.append(f"labels not seen in train: {sorted(map(str, unknown))[:5]}")
    return errors


def run(zip_path: Path, task_src: Path, time_limit: float, extra_env: dict) -> dict:
    work = Path(tempfile.mkdtemp(prefix="alfa-platform-")) / task_src.name
    shutil.copytree(task_src, work)
    for leftover in ("submission.csv", "agent_journal.jsonl", "agent_llm.jsonl"):
        (work / leftover).unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        copied = [n for n in z.namelist() if "/" not in n and (n.endswith(".py") or n == "requirements.txt")]
        for name in copied:
            (work / name).write_bytes(z.read(name))
    env = {**os.environ, **extra_env, "AGENT_TIME_LIMIT_SEC": str(time_limit)}
    started = time.time()
    with open(work / "agent.log", "w", encoding="utf-8") as log:
        try:
            code = subprocess.run([sys.executable, "solution.py", "task.md"], cwd=work, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, timeout=time_limit + 120).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    log_text = (work / "agent.log").read_text(encoding="utf-8", errors="replace")
    return {"task": task_src.name, "exit": code, "seconds": round(time.time() - started),
            "errors": check_submission(work), "v2": "v2 submission written" in log_text,
            "llm": "llm=on" in log_text, "dir": str(work)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip")
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--env-file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    extra_env = read_env_file(Path(args.env_file)) if args.env_file else {}
    for item in args.set:
        key, _, value = item.partition("=")
        extra_env[key.strip()] = value.strip()
    ok = True
    for task in args.tasks:
        res = run(Path(args.zip), Path(task), args.time_limit, extra_env)
        verdict = "OK" if res["exit"] == 0 and not res["errors"] else "FAIL"
        ok &= verdict == "OK"
        print(f"{verdict} {res['task']}: exit={res['exit']} {res['seconds']}s llm={res['llm']} v2={res['v2']} "
              f"errors={res['errors']} log={res['dir']}\\agent.log", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
