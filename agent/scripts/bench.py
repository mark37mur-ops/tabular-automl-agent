"""Run the agent on polygon tasks like the platform does and score the answers.

Usage (from repo root):
  python agent/scripts/bench.py [--agent agent] [--tasks amazon_access bike_sharing]
                                [--time-limit 1800] [--mode v2|v1] [--out bench.json]

Each task's public/ folder is copied to a fresh work dir, `python solution.py task.md`
runs there with AGENT_TIME_LIMIT_SEC and AGENT_MODE, then polygon/score.py scores
submission.csv. Results are written after every task, so a crash keeps earlier ones.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "polygon"))
import score as polygon_score  # noqa: E402


def read_env_file(path: Path) -> dict:
    """KEY=VALUE lines (comments and blanks skipped). Values are passed to the agent only,
    never printed: the file may hold an API key."""
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def run_task(agent_dir: Path, task: str, time_limit: float, mode: str, work_root: Path,
             extra_env: dict | None = None) -> dict:
    work = work_root / task
    shutil.copytree(REPO / "polygon" / "tasks" / task / "public", work)
    env = {**os.environ, **(extra_env or {}), "AGENT_TIME_LIMIT_SEC": str(time_limit), "AGENT_MODE": mode}
    started = time.time()
    with open(work / "agent.log", "w", encoding="utf-8") as log:
        try:
            code = subprocess.run([sys.executable, "solution.py", str(work / "task.md")], cwd=agent_dir,
                                  env=env, stdout=log, stderr=subprocess.STDOUT,
                                  timeout=time_limit + 120).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"  # like the platform: whatever submission.csv exists now is scored
    elapsed = time.time() - started
    result = polygon_score.score(task, work / "submission.csv") if (work / "submission.csv").exists() \
        else {"task": task, "valid": False, "score": 0.0, "errors": ["no submission.csv"]}
    result.update(exit_code=code, seconds=round(elapsed, 1), log=str(work / "agent.log"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default=str(REPO / "agent"))
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--mode", default="v2", choices=["v1", "v2"])
    parser.add_argument("--out")
    parser.add_argument("--env-file", help="KEY=VALUE file for the agent, e.g. .env with OPENAI_* for an LLM")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="extra agent env var, overrides --env-file (repeatable)")
    parser.add_argument("--record", help="folder for a result record (experiments/results in main)")
    parser.add_argument("--approach", default="v3-rd", help="approach name for the results table")
    parser.add_argument("--notes", default="", help="one line for the results table")
    args = parser.parse_args()
    extra_env = read_env_file(Path(args.env_file)) if args.env_file else {}
    for item in args.set:
        key, _, value = item.partition("=")
        extra_env[key.strip()] = value.strip()
    if extra_env:
        has_key = bool(extra_env.get("OPENAI_API_KEY"))
        print(f"env file: {sorted(extra_env)}; OPENAI_API_KEY {'set' if has_key else 'EMPTY'}; "
              f"model {extra_env.get('OPENAI_MODEL', '-')}", flush=True)
    tasks = args.tasks or sorted(p.name for p in (REPO / "polygon" / "tasks").iterdir() if p.is_dir())
    work_root = Path(tempfile.mkdtemp(prefix="alfa-bench-"))
    results = []
    for task in tasks:
        res = run_task(Path(args.agent).resolve(), task, args.time_limit, args.mode, work_root, extra_env)
        results.append(res)
        print(f"{task:15s} score={res['score']:6.2f} value={res.get('value', float('nan')):.5f} "
              f"time={res['seconds']:7.1f}s valid={res['valid']}", flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MEAN score = {sum(r['score'] for r in results) / len(results):.2f}")
    if args.record:
        path = record(Path(args.record), results, args, extra_env)
        print(f"result record: {path}")


def _git(*cmd: str) -> str:
    try:
        return subprocess.run(["git", *cmd], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


TOGGLES = {"AGENT_SCREEN": "scr", "AGENT_DEVELOPER": "dev", "AGENT_COMPONENTS": "cmp",  # env -> tag
           "AGENT_HISTORY": "his", "AGENT_LABEL_SHIFT": "lsh"}
TOGGLES_OFF_BY_DEFAULT = {"AGENT_LABEL_SHIFT"}  # as in automl/research/rd.py Toggles.from_env
SAFE_SETTINGS = ("AGENT_",)  # only agent knobs go into the record; never keys or URLs


def _flag(value: str | None, default: str = "1") -> int:
    return int(str(value if value is not None else default).strip().lower() not in ("0", "false", "off", "no", ""))


def record(folder: Path, results: list[dict], args, extra_env: dict) -> Path:
    """One JSON per run in the format of experiments/results in main (see docs/experiments.md there),
    plus agent.log / agent_journal.jsonl / agent_llm.jsonl per task in a folder named like the record."""
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    env = {**os.environ, **extra_env}
    llm_on = bool(env.get("OPENAI_API_KEY"))
    model = env.get("OPENAI_MODEL") or "?"
    base_url = env.get("OPENAI_BASE_URL") or ""
    host = urlparse(base_url).hostname or "" if base_url else ""
    # the record keeps only the provider kind, never the endpoint address
    llm = f"{model} ({'Groq' if 'groq' in host else 'custom endpoint' if host else 'OpenAI API'})" if llm_on else None
    toggles = {name: _flag(env.get(name), "0" if name in TOGGLES_OFF_BY_DEFAULT else "1") for name in TOGGLES}
    tag = "_".join(f"{short}{toggles[name]}" for name, short in TOGGLES.items())
    settings = {k: v for k, v in env.items() if k.startswith(SAFE_SETTINGS) and k != "AGENT_TIME_LIMIT_SEC"}
    stamp = time.strftime("%Y-%m-%dT%H:%M")
    rid = (f"{stamp[:10]}_{args.approach}_{commit}_{'llm' if llm_on else 'rules'}_"
           f"{int(args.time_limit // 60)}m_{tag}_{stamp[11:].replace(':', '')}")
    rec = {"id": rid, "datetime": stamp, "date": stamp[:10], "approach": args.approach, "branch": branch,
           "commit": commit, "llm": llm, "time_limit_sec": args.time_limit, "toggles": toggles,
           "settings": settings, "notes": args.notes,
           "tasks": {r["task"]: ({"metric": r.get("metric"), "value": round(r["value"], 5),
                                  "score": round(r["score"], 2), "seconds": r["seconds"]} if r.get("valid")
                                 else {"valid": False, "score": 0.0, "errors": r.get("errors"),
                                       "seconds": r["seconds"]})
                     for r in results}}
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{rid}.json"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in results:  # evidence next to the record: logs and journals per task
        work = Path(r["log"]).parent
        dest = folder / rid / r["task"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("agent.log", "agent_journal.jsonl", "agent_llm.jsonl"):
            if (work / name).exists():
                shutil.copy2(work / name, dest / name)
    return path


if __name__ == "__main__":
    main()
