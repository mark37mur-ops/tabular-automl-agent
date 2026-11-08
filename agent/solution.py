"""Platform entry point: python solution.py task.md

Stage 1 (v1, no LLM): rules -> constant fallback -> 5-fold LightGBM -> submission.
Stage 2 (v2, automl/pipeline.py): analyst -> research loop -> model blend -> better submission.
Every stage that can fail falls back to the previous answer; the process always exits 0.
AGENT_MODE=v1 stops after stage 1.
"""
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm  # noqa: E402
import pandas as pd  # noqa: E402

from automl.model import train_and_predict  # noqa: E402
from automl.submission import fallback_predictions, write_submission  # noqa: E402
from automl.task import build_spec, resolve_output_path  # noqa: E402


def _env_seconds(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


TIME_LIMIT_SEC = _env_seconds("AGENT_TIME_LIMIT_SEC", 3 * 3600)
V1_BUDGET_SEC = _env_seconds("AGENT_V1_BUDGET_SEC", 120.0)
SAFETY_MARGIN_SEC = min(1200.0, 0.15 * TIME_LIMIT_SEC)  # no new fold after limit - margin
HARD_MARGIN_SEC = min(480.0, 0.05 * TIME_LIMIT_SEC)  # boosting is cut at limit - hard margin


def log(message: str) -> None:
    print(f"[agent {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _copy_sample_as_last_resort(task_path: Path) -> None:
    try:
        samples = sorted(task_path.parent.rglob("sample_submission.csv"), key=lambda p: len(p.parts))
        if not samples:
            log("no sample_submission.csv found: nothing to submit")
            return
        output = resolve_output_path(task_path)
        shutil.copyfile(samples[0], output)
        log(f"copied {samples[0]} to {output} as the answer")
    except Exception:
        log("last resort failed:\n" + traceback.format_exc())


def main(argv: list[str]) -> int:
    started = time.time()
    deadline = started + TIME_LIMIT_SEC - SAFETY_MARGIN_SEC
    task_path = Path(argv[1] if len(argv) > 1 else "task.md").resolve()
    log(f"start: python {platform.python_version()}, pandas {pd.__version__}, "
        f"lightgbm {lightgbm.__version__}, cpu {os.cpu_count()}, cwd {os.getcwd()}, task {task_path}")

    try:
        spec = build_spec(task_path)
    except Exception:
        log("task parsing failed:\n" + traceback.format_exc())
        _copy_sample_as_last_resort(task_path)
        return 0
    log(f"task: {spec.task_type}, target={spec.target}, answer_col={spec.answer_col}, "
        f"id={spec.id_col}, metric={spec.metric}, format={spec.pred_format}")
    log(f"files: train={spec.train_path}, test={spec.test_path}, output={spec.output_path}")
    for note in spec.notes:
        log(f"note: {note}")

    try:
        train = pd.read_csv(spec.train_path)
        test = pd.read_csv(spec.test_path)
        log(f"train {train.shape}, test {test.shape}")
        write_submission(test, fallback_predictions(train, len(test), spec), spec)
        log("fallback submission written")
    except Exception:
        log("fallback failed:\n" + traceback.format_exc())
        _copy_sample_as_last_resort(task_path)
        return 0

    hard_deadline = started + TIME_LIMIT_SEC - HARD_MARGIN_SEC
    try:
        # Keep the fallback stage bounded so one expensive v1 fit cannot consume v2's budget.
        v1_end = started + max(1.0, min(V1_BUDGET_SEC, .1 * TIME_LIMIT_SEC))
        result = train_and_predict(train, test, spec, min(deadline, v1_end),
                                   hard_deadline=min(hard_deadline, v1_end), log=log)
        write_submission(test, result["pred"], spec)
        log(f"v1 submission written in {time.time() - started:.0f}s")
    except Exception as error:
        log(f"v1 model failed ({error!r}), keeping fallback submission:\n" + traceback.format_exc())

    if os.environ.get("AGENT_MODE", "v2") == "v1":
        return 0
    try:
        # Imported here: a broken v2 module must never cost the v1 answer already on disk.
        from automl.pipeline import run_v2
        pred = run_v2(train, test, spec, deadline, hard_deadline, log=log)
        write_submission(test, pred, spec)
        log(f"v2 submission written in {time.time() - started:.0f}s")
    except Exception as error:
        log(f"v2 failed ({error!r}), keeping previous submission:\n" + traceback.format_exc())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
