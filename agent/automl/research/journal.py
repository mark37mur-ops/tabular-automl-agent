"""Experiment journal: one line per attempt, written to JSONL and summarised for prompts."""
import json
import time
from pathlib import Path


def signature(card: dict) -> str:
    """Identity of an idea, used to avoid repeating attempts."""
    body = {k: card.get(k) for k in ("kind", "ops", "params", "value", "hypothesis")}
    return json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)


class Journal:
    def __init__(self, path: Path | None = None, log=print):
        self.path = path
        self.log = log
        self.entries: list[dict] = []
        self.tried: set[str] = set()

    def add(self, card: dict, score: float | None, gain: float | None, verdict: str,
            seconds: float, source: str, note: str = "", extra: dict | None = None) -> dict:
        entry = {"n": len(self.entries) + 1, "time": time.strftime("%H:%M:%S"), "source": source,
                 "title": card.get("title", ""), "kind": card.get("kind"), "ops": card.get("ops"),
                 "params": card.get("params"), "hypothesis": card.get("hypothesis"),
                 "score": score, "gain": gain, "verdict": verdict,
                 "seconds": round(seconds, 1), "note": note, **(extra or {})}
        self.entries.append(entry)
        self.tried.add(signature(card))
        if self.path is not None:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except OSError as error:
                self.log(f"journal file not written ({error!r}); keeping it in memory only")
                self.path = None
        return entry

    def summary(self, limit: int = 25) -> str:
        """Compact text for the LLM: accepted ideas first, then the latest rejections."""
        if not self.entries:
            return "No experiments yet."
        kept = [e for e in self.entries if e["verdict"] == "kept"]
        other = [e for e in self.entries if e["verdict"] != "kept"][-limit:]
        lines = []
        for e in kept + other:
            gain = "n/a" if e["gain"] is None else f"{e['gain']:+.5f}"
            lines.append(f"- [{e['verdict']}] {e['title']} (gain {gain}){': ' + e['note'] if e['note'] else ''}")
        return "\n".join(lines)
