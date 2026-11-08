"""Ideas borrowed from RD-Agent (Microsoft, arXiv 2505.14738), adapted to our guarded loop.

- choose_component: each research round targets ONE component of the solution
  (features via ops, developer-written feature code, or the model settings), picked by UCB
  over what the journal says worked so far.
- developer: the LLM writes code for a feature block from a plain-language hypothesis and
  fixes it from the error message, up to DEV_ATTEMPTS times, on a small sample, before the
  idea is ever evaluated (RD-Agent's Developer + debug-on-sample).

Switches (env; 1 switches on, 0/false/off/no switch off), read once per run into Toggles. Default on
except AGENT_LABEL_SHIFT:
  AGENT_SCREEN      screen ideas on a sample before the full evaluation (pipeline.calibrate/_try_card)
  AGENT_DEVELOPER   LLM developer writes feature-block code from plain-language hypotheses
  AGENT_COMPONENTS  each research round targets one component chosen by UCB
  AGENT_HISTORY     object's past target as features on temporal tasks (automl/history.py)
  AGENT_LABEL_SHIFT EM correction of test class priors for label answers (automl/shift.py). Off by
                    default: on the public UNSW task (test drawn differently, not only other class
                    shares) EM put DoS at 0% of the test, about 5% in fact; polygon gain was +0.03.
"""
import math
import os
import re
import time
from dataclasses import dataclass

import pandas as pd

from automl.features import OPS, apply_op

COMPONENTS = ("features", "develop", "model")
DEV_ATTEMPTS = 3
DEV_SAMPLE_ROWS = 2000
SYSTEM_DEV = ("You are a careful Python developer inside an AutoML agent. You write short, vectorised "
              "pandas/numpy code. Reply with ONE ```python code block and nothing else.")


def enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "off", "no", "")


@dataclass(frozen=True)
class Toggles:
    screen: bool = True
    developer: bool = True
    components: bool = True
    history: bool = True       # automl/history.py: object's past target on temporal tasks
    label_shift: bool = False  # automl/shift.py: EM re-estimation of test class priors (opt-in)

    @classmethod
    def from_env(cls) -> "Toggles":
        return cls(enabled("AGENT_SCREEN"), enabled("AGENT_DEVELOPER"), enabled("AGENT_COMPONENTS"),
                   enabled("AGENT_HISTORY"), enabled("AGENT_LABEL_SHIFT", default="0"))


def component_of(entry: dict) -> str | None:
    kind = entry.get("kind")
    if kind == "develop":  # developer attempts that failed before an evaluation
        return "develop"
    if kind == "features":
        return "develop" if entry.get("source") == "llm-dev" else "features"
    if kind in ("params", "target_log", "class_weight"):
        return "model"
    return None


def choose_component(journal_entries: list[dict], available: tuple[str, ...]) -> str:
    """UCB1 over components: reward 1 for a kept idea, 0 otherwise; untried components first."""
    stats = {c: [0, 0] for c in available}  # tries, kept
    for entry in journal_entries:
        comp = component_of(entry)
        if comp in stats:
            stats[comp][0] += 1
            stats[comp][1] += entry.get("verdict") == "kept"
    for comp in available:
        if stats[comp][0] == 0:
            return comp
    total = sum(t for t, _ in stats.values())
    return max(available, key=lambda c: stats[c][1] / stats[c][0] + math.sqrt(2 * math.log(total) / stats[c][0]))


def focus_text(component: str, target: str) -> str:
    """Instruction appended to the researcher prompt for this round."""
    if component == "develop":
        return ('THIS ROUND: propose ideas for the DEVELOPER only: cards with "kind": "develop" and a '
                '"hypothesis" field describing in plain words a block of new features that needs custom '
                f"pandas code (combinations, groupings, domain formulas). Never use the target {target!r}.")
    if component == "model":
        return ('THIS ROUND: propose ideas about the MODEL only: cards with "kind" "params", "target_log" '
                'or "class_weight".')
    return ('THIS ROUND: propose FEATURE ideas only: cards with "kind": "features" built from the '
            f"available ops (no free code). Never use the target {target!r}.")


def extract_code(text: str) -> str | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL)
    code = blocks[0] if blocks else (text if "def features" in text else None)
    return code.strip() if code else None


def _slug(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", text.lower()).strip("_")[:30] or "block"


DEV_MIN_ATTEMPT_SECONDS = 30  # first-attempt estimate; later attempts use the measured duration


def developer(llm, card: dict, frame: pd.DataFrame, numeric: set[str], target: str, log=print,
              deadline: float = float("inf"), stats: dict | None = None) -> dict | None:
    """Turn a 'develop' hypothesis into a checked code_block op, or None after DEV_ATTEMPTS.
    stats (optional) receives {"attempts": n, "outcome": ok|failed|skipped}."""
    stats = stats if stats is not None else {}
    stats.update(attempts=0, outcome="skipped")
    if llm is None or not llm.available:
        return None
    sample = frame.head(DEV_SAMPLE_ROWS)
    numeric_cols = [c for c in frame.columns if c in numeric]
    categorical_cols = [c for c in frame.columns if c not in numeric]
    preview = sample.head(3).to_string(max_cols=40, max_colwidth=20)[:2500]
    base_prompt = (
        f"Hypothesis to implement: {card.get('hypothesis') or card.get('title')}\n"
        f"Why: {card.get('why', '')}\n\n"
        f"Numeric columns: {', '.join(numeric_cols[:150])}\n"
        f"Categorical columns (pandas category dtype): {', '.join(categorical_cols[:80]) or '-'}\n"
        f"First rows:\n{preview}\n\n"
        "Write `def features(df):` that returns a pandas DataFrame with 1-20 NEW columns, same row order "
        "and length as df. Rules: use only pd and np (already imported), no imports, no loops (for/while), "
        "no file or network access, vectorised operations only, handle missing values and division by "
        f"zero. The target column {target!r} is NOT in df. Example:\n"
        "```python\ndef features(df):\n    out = pd.DataFrame(index=df.index)\n"
        "    out['a_per_b'] = df['a'] / df['b'].replace(0, np.nan)\n    return out\n```")
    prompt, last_error = base_prompt, None
    stats["outcome"] = "failed"
    estimate = DEV_MIN_ATTEMPT_SECONDS
    for attempt in range(1, DEV_ATTEMPTS + 1):
        if time.time() + estimate > deadline:
            log(f"developer: {card.get('title')!r} stopped before attempt {attempt}: research time is over")
            if attempt == 1:
                stats["outcome"] = "skipped"
            break
        stats["attempts"] = attempt
        attempt_started = time.time()
        reply = llm.ask_text("developer", SYSTEM_DEV, prompt)
        code = extract_code(reply or "")
        if code is None:
            last_error = "no ```python code block in the reply"
        else:
            op = {"op": "code_block", "name": _slug(card.get("title", "block")), "code": code}
            try:
                apply_op(sample, op)  # guard + run on the sample: RD-Agent's debug-on-sample
                log(f"developer: {card.get('title')!r} ok on attempt {attempt}")
                stats["outcome"] = "ok"
                return op
            except Exception as error:  # guard rejections and runtime errors both go back to the model
                last_error = repr(error)[:400]
        log(f"developer: {card.get('title')!r} attempt {attempt} failed: {last_error}")
        estimate = max(DEV_MIN_ATTEMPT_SECONDS, time.time() - attempt_started)
        if not llm.available:
            break
        prompt = (f"{base_prompt}\n\nYour previous code:\n```python\n{code or ''}\n```\n"
                  f"It failed with: {last_error}\nFix it and reply with the corrected code block only.")
    return None


assert "code_block" in OPS
