"""Проверяет сабмит агента на задаче полигона и считает балл как на платформе.

Запуск: .venv/Scripts/python.exe -I polygon/score.py <task_name> <submission.csv>
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent


def check_format(sub: pd.DataFrame, answers: pd.DataFrame, target: str) -> list[str]:
    errors = []
    if list(sub.columns) != ["id", target]:
        errors.append(f"колонки {list(sub.columns)}, ожидались ['id', '{target}']")
        return errors
    if len(sub) != len(answers):
        errors.append(f"{len(sub)} строк, ожидалось {len(answers)}")
    if sub["id"].duplicated().any():
        errors.append("повторяющиеся id")
    if set(sub["id"]) != set(answers["id"]):
        errors.append("набор id не совпадает с test.csv")
    if sub[target].isna().any():
        errors.append(f"пропуски в {target}: {int(sub[target].isna().sum())}")
    return errors


def score(task: str, submission: Path) -> dict:
    private = ROOT / "tasks" / task / "private"
    meta = json.loads((private / "meta.json").read_text(encoding="utf-8"))
    answers = pd.read_csv(private / "answers.csv")
    sub = pd.read_csv(submission)
    target = meta["target"]

    errors = check_format(sub, answers, target)
    if errors:
        return {"task": task, "valid": False, "errors": errors, "score": 0.0}

    merged = answers.merge(sub, on="id", suffixes=("_true", "_pred"))
    value = metrics.compute(meta["metric"], merged[f"{target}_true"], merged[f"{target}_pred"])
    return {
        "task": task,
        "valid": True,
        "metric": meta["metric"],
        "value": value,
        "baseline_value": meta["baseline_value"],
        "score": metrics.to_score(meta["metric"], value, meta["baseline_value"]),
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) != 3:
        sys.exit("Использование: score.py <task_name> <submission.csv>")
    print(json.dumps(score(sys.argv[1], Path(sys.argv[2])), ensure_ascii=False, indent=2))
