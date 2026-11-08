"""Rebuild the results tables in docs/results.md from experiments/results/*.json.

Usage (from repo root): python experiments/report.py
Only the block between the RESULTS markers is rewritten; the rest of the document is manual.
Standard library only, so it runs anywhere.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
DOC = REPO / "docs" / "results.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def load() -> list[dict]:
    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as error:
            raise SystemExit(f"{path.name}: invalid JSON ({error})")
    return sorted(runs, key=lambda r: (r.get("datetime") or r.get("date", ""), r.get("id", "")))


def render(runs: list[dict]) -> str:
    tasks = sorted({t for r in runs for t in r.get("tasks", {})})
    lines = [f"_Сгенерировано `experiments/report.py` из {len(runs)} прогонов. Не править вручную._", "",
             "### Все прогоны", "",
             "| Дата | Подход | Ветка @ коммит | LLM | Лимит | Переключатели | " + " | ".join(tasks) + " | Среднее |",
             "|---|---|---|---|---|---|" + "---|" * len(tasks) + "---|"]
    best = {t: None for t in tasks}
    for r in runs:
        cells = []
        for t in tasks:
            res = r.get("tasks", {}).get(t)
            if res is None:
                cells.append("—")
            elif res.get("valid") is False:
                cells.append("упал (0)")
            else:
                cells.append(f"{res['score']:.2f}")
                if best[t] is None or res["score"] > best[t][0]:
                    best[t] = (res["score"], r)
        scores = [r["tasks"][t].get("score", 0.0) for t in tasks if t in r.get("tasks", {})]
        mean = f"**{sum(scores) / len(scores):.2f}**" if len(scores) == len(tasks) else "—"
        llm = r.get("llm") or "нет"
        limit = f"{int(r.get('time_limit_sec', 0)) // 60} мин"
        toggles = r.get("toggles")
        switch = ", ".join(f"{k.replace('AGENT_', '').lower()}={v}" for k, v in toggles.items()) if toggles else "—"
        lines.append(f"| {r.get('date', '')} | {r.get('approach', '')} | `{r.get('branch', '')}` @ "
                     f"`{r.get('commit', '')}` | {llm} | {limit} | {switch} | " + " | ".join(cells) + f" | {mean} |")
    lines += ["", "Среднее считается только для прогонов по всем задачам. «—» — задача не запускалась. "
              "Логи и журналы агента лежат в папке с тем же именем, что и файл прогона.", "",
              "### Лучший результат по каждой задаче", "", "| Задача | Балл | Метрика | Подход | Коммит |",
              "|---|---|---|---|---|"]
    for t in tasks:
        if best[t]:
            score, r = best[t]
            res = r["tasks"][t]
            lines.append(f"| {t} | {score:.2f} | {res['metric']} = {res['value']} | {r['approach']} | "
                         f"`{r['commit']}` |")
    lines += ["", "### Заметки к прогонам", ""]
    lines += [f"- `{r['id']}`: {r.get('notes', '')}" for r in runs if r.get("notes")]
    return "\n".join(lines)


def main() -> None:
    doc = DOC.read_text(encoding="utf-8")
    if START not in doc or END not in doc:
        raise SystemExit(f"markers {START} / {END} not found in {DOC}")
    head, rest = doc.split(START, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(f"{head}{START}\n{render(load())}\n{END}{tail}", encoding="utf-8")
    print(f"{DOC.relative_to(REPO)} updated")


if __name__ == "__main__":
    main()
