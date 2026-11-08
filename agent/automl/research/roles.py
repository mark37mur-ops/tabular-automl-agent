"""The three LLM roles (analyst, researcher, critic) with rule-based fallbacks.

Every role returns validated, plain-Python structures. If the LLM is unavailable or answers
garbage, the rule-based version is used, so the research loop always has ideas to test.

Idea card: {"title", "why", "kind": features|params|target_log|class_weight,
            "ops": [feature ops], "params": {...}, "value": bool}
"""
import json
import os

import numpy as np
import pandas as pd

from automl.features import OPS, normalize_op, numeric_columns, validate_op
from automl.research.journal import signature

SYSTEM = ("You are a senior data scientist inside an autonomous AutoML agent for tabular data. "
          "Answer with ONE JSON value only, no prose outside JSON.")
PARAM_LIMITS = {"num_leaves": (7, 255), "min_child_samples": (5, 500), "learning_rate": (0.01, 0.2),
                "colsample_bytree": (0.2, 1.0), "subsample": (0.4, 1.0), "reg_lambda": (0.0, 50.0),
                "reg_alpha": (0.0, 50.0), "max_depth": (-1, 16), "min_split_gain": (0.0, 1.0)}


# ---------------------------------------------------------------- data summary

def data_summary(train: pd.DataFrame, spec, dictionary: pd.DataFrame | None, max_cols: int | None = None) -> str:
    """One line per column for prompts; AGENT_LLM_SUMMARY_COLS (default 120) caps prompt size."""
    if max_cols is None:
        try:
            max_cols = int(os.environ.get("AGENT_LLM_SUMMARY_COLS", 120))
        except ValueError:
            max_cols = 120
    desc = {}
    if dictionary is not None and {"feature", "description_ru"} <= set(dictionary.columns):
        desc = dict(zip(dictionary["feature"].astype(str), dictionary["description_ru"].astype(str)))
    lines = [f"rows: {len(train)}, target: {spec.target} ({spec.task_type}), metric: {spec.metric}"]
    y = train[spec.target]
    if spec.task_type == "regression":
        q = y.quantile([0, 0.5, 0.99, 1]).round(3).tolist()
        lines.append(f"target min/median/p99/max: {q}, skew: {y.skew():.2f}")
    else:
        lines.append(f"target classes: {y.value_counts().head(15).to_dict()}")
    cols = [c for c in train.columns if c not in (spec.id_col, spec.target)]
    for col in cols[:max_cols]:
        s = train[col]
        kind = "num" if pd.api.types.is_numeric_dtype(s) else "str"
        examples = ", ".join(map(str, s.dropna().unique()[:3]))[:60]
        stats = f"nunique={s.nunique()}, missing={s.isna().mean():.0%}"
        if kind == "num" and s.notna().any():
            stats += f", min={s.min():.4g}, median={s.median():.4g}, max={s.max():.4g}"
        lines.append(f"- {col} [{kind}] {stats}; e.g. {examples}; {desc.get(col, '')[:120]}")
    if len(cols) > max_cols:
        lines.append(f"... and {len(cols) - max_cols} more columns")
    return "\n".join(lines)


# ---------------------------------------------------------------- analyst

def analyst(llm, task_md: str, summary: str, columns: list[str], log=print) -> dict:
    """Task passport: semantic hints the rules cannot see. Unknown columns are dropped."""
    passport = {"categorical_columns": [], "drop_columns": [], "leak_suspects": [], "entity_key_columns": [],
                "notes": "",
                "feature_ideas": []}
    if llm is None or not llm.available:
        return passport
    user = (
        f"Task description:\n{task_md[:6000]}\n\nData summary (train):\n{summary}\n\n"
        "Return JSON with keys:\n"
        '"categorical_columns": numeric columns that are really IDs/codes/categories,\n'
        '"drop_columns": columns that are row identifiers or pure noise (be conservative; '
        "they are only removed if validation improves),\n"
        '"leak_suspects": columns that may directly reveal the target,\n'
        '"entity_key_columns": columns that together identify the same object (building, client, '
        "device) across time periods: static attributes only, never the time column or values that "
        'change every period; [] if rows are not repeated objects,\n'
        '"notes": 2-4 sentences about the domain and what drives the target,\n'
        '"feature_ideas": up to 6 short ideas for new features in plain words.')
    answer = llm.ask_json("analyst", SYSTEM, user)
    if not isinstance(answer, dict):
        log(f"analyst: unusable reply ({type(answer).__name__}), passport from rules only")
        return passport
    known = set(columns)
    for key in ("categorical_columns", "drop_columns", "leak_suspects", "entity_key_columns"):
        values = answer.get(key) or []
        passport[key] = [c for c in values if isinstance(c, str) and c in known] if isinstance(values, list) else []
    passport["notes"] = str(answer.get("notes", ""))[:1000]
    ideas = answer.get("feature_ideas") or []
    passport["feature_ideas"] = [str(i)[:200] for i in ideas][:6] if isinstance(ideas, list) else []
    return passport


# ---------------------------------------------------------------- card validation

INT_PARAMS = {"num_leaves", "min_child_samples", "max_depth"}
KINDS = ("features", "params", "target_log", "class_weight", "develop")


def clean_card(card, columns: set[str], problem, reasons: list | None = None,
               numeric: set[str] | None = None, develop_enabled: bool = False) -> dict | None:
    """Validate an idea card from the LLM; return a normalised card or None (reason appended
    to `reasons`). Never raises on malformed input. `develop` cards pass only if the developer
    role is on; `code_block` ops never come from here (the developer writes them)."""
    def reject(reason):
        if reasons is not None:
            reasons.append(reason[:160])
        return None

    if not isinstance(card, dict):
        return reject(f"card is {type(card).__name__}, not an object")
    kind = card.get("kind", "features")
    if kind not in KINDS:
        return reject(f"unknown kind {kind!r}")
    out = {"title": str(card.get("title", ""))[:120] or kind, "why": str(card.get("why", ""))[:300],
           "kind": kind}
    if kind == "develop" and not develop_enabled:
        return reject("developer role is off")
    if kind == "features":
        ops = card.get("ops") or []
        if not isinstance(ops, list) or not ops:
            return reject("features card without ops")
        if any(isinstance(op, dict) and op.get("op") in RESEARCHER_HIDDEN_OPS for op in ops):
            return reject("code_block ops are written by the developer role only")
        good = []
        for op in ops[:6]:
            norm = normalize_op(op)
            reason = validate_op(norm if norm is not None else op, columns, numeric)
            if reason is None:
                good.append(norm)
            elif reasons is not None:
                reasons.append(reason[:160])
        if len(good) != len(ops[:6]):  # a partly broken card would be judged as something else
            return reject(f"{len(ops[:6]) - len(good)} of {len(ops[:6])} ops invalid")
        out["ops"] = good
    elif kind == "params":
        params = card.get("params") or {}
        clean = {}
        for key, value in params.items() if isinstance(params, dict) else []:
            if key in PARAM_LIMITS and isinstance(value, (int, float)) and not isinstance(value, bool):
                low, high = PARAM_LIMITS[key]
                value = min(max(value, low), high)
                clean[key] = int(round(value)) if key in INT_PARAMS else float(value)
        if not clean:
            return reject("params card without known params")
        out["params"] = clean
    elif kind == "target_log":
        if problem.classification or np.nanmin(problem.y_raw.astype(float)) < 0:
            return reject("target_log needs a non-negative regression target")
        out["value"] = bool(card.get("value", True))
    elif kind == "class_weight":
        if not problem.classification:
            return reject("class_weight needs classification")
        out["value"] = bool(card.get("value", True))
    elif kind == "develop":  # plain-language hypothesis; the developer role writes the code later
        hypothesis = card.get("hypothesis") or card.get("why")
        if not isinstance(hypothesis, str) or len(hypothesis.strip()) < 10:
            return reject("develop card without a hypothesis")
        if problem.spec.target in hypothesis.replace("'", " ").replace('"', " ").split():
            return reject("develop hypothesis mentions the target")
        out["hypothesis"] = hypothesis.strip()[:600]
    return out


# ---------------------------------------------------------------- researcher

RESEARCHER_HIDDEN_OPS = {"code_block"}  # written only by the developer role, never proposed directly


def _ops_help() -> str:
    lines = [f'- {name}: {text}' for name, text in OPS.items() if name not in RESEARCHER_HIDDEN_OPS]
    return "\n".join(lines)


def researcher(llm, passport: dict, summary: str, journal_summary: str, importance: pd.Series,
               columns: list[str], problem, n: int | None = None, log=print,
               numeric: set[str] | None = None, focus: str = "", develop_enabled: bool = False) -> list[dict]:
    """n ideas per call (AGENT_LLM_IDEAS, default 4); numeric = numeric feature columns;
    focus = extra instruction naming the component to work on this round (RD-Agent style)."""
    if llm is None or not llm.available:
        return []
    if n is None:
        try:
            n = max(1, int(os.environ.get("AGENT_LLM_IDEAS", 4)))
        except ValueError:
            n = 4
    numeric_list = [c for c in columns if numeric is None or c in numeric]
    categorical_list = [c for c in columns if numeric is not None and c not in numeric]
    top = ", ".join(f"{c}={v:.3f}" for c, v in importance.sort_values(ascending=False).head(20).items())
    user = (
        f"Data summary:\n{summary}\n\nAnalyst notes: {passport.get('notes', '')}\n"
        f"Analyst feature ideas: {json.dumps(passport.get('feature_ideas', []), ensure_ascii=False)}\n"
        f"Numeric feature columns: {', '.join(numeric_list[:200])}\n"
        f"Categorical feature columns: {', '.join(categorical_list[:200]) or '-'}\n"
        f"Feature importance (top 20): {top}\n\n"
        f"Experiments so far:\n{journal_summary}\n\n"
        f"Propose {n} NEW experiments that could improve the validation {problem.spec.metric}. "
        f"The target column {problem.spec.target!r} is NOT available as a feature: never use it. "
        "Arithmetic ops, log1p and group_mean/std/dev need NUMERIC columns in a, b, col; "
        "categorical columns go to `by`, freq, group_count, group_nunique. Keep answers short. "
        "Do not repeat tried ideas. Return a JSON list of cards. Card fields:\n"
        '"title" (short), "why" (one sentence), "kind" one of "features", "params", "target_log", '
        f'"class_weight"{", " + chr(34) + "develop" + chr(34) if develop_enabled else ""}.\n'
        'For "features": "ops" = list of ops. Available ops (fields a, b, col, by=[...], code):\n'
        f"{_ops_help()}\n"
        'For "code": {"op": "code", "name": "...", "code": "def feature(df):\\n    return ..."} '
        "using only pd and np, vectorised, no imports, no loops over rows.\n"
        f'For "params": "params" = LightGBM params among {", ".join(PARAM_LIMITS)}.\n'
        'For "target_log"/"class_weight": "value": true/false.\n'
        + ('For "develop": "hypothesis" = plain-language description of a block of new features that a '
           "developer will implement in code.\n" if develop_enabled else "")
        + (f"\n{focus}" if focus else ""))
    answer = llm.ask_json("researcher", SYSTEM, user)
    if isinstance(answer, dict):
        answer = answer.get("cards") or answer.get("experiments") or [answer]
    if not isinstance(answer, list):
        log(f"researcher: reply is not a list of cards ({type(answer).__name__})")
        return []
    known, reasons = set(columns), []
    cards = [clean_card(c, known, problem, reasons, numeric, develop_enabled) for c in answer[: n + 3]]
    good = [c for c in cards if c is not None]
    log(f"researcher: proposed {len(answer)}, valid {len(good)}"
        + (f"; dropped: {reasons[:5]}" if reasons else ""))
    return good


def rule_ideas(problem, frame: pd.DataFrame, importance: pd.Series, passport: dict) -> list[dict]:
    """Generic, cheap ideas used when there is no LLM (and mixed in when there is)."""
    cards = []
    cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in frame.columns if c not in cats]
    ranked = [c for c in importance.sort_values(ascending=False).index if c in frame.columns]
    top_num = [c for c in ranked if c in nums][:4]
    top_cat = [c for c in ranked if c in cats][:3]

    for col in passport.get("categorical_columns", []):
        if col in nums:
            cards.append({"title": f"treat {col} as category (analyst)", "kind": "features",
                          "ops": [{"op": "as_category", "col": col}], "why": "analyst"})
    drops = [c for c in passport.get("drop_columns", []) if c in frame.columns]
    if drops:  # the analyst's drop advice is tested like any other idea, never applied blindly
        cards.append({"title": "drop columns flagged by analyst", "kind": "features",
                      "ops": [{"op": "drop", "col": c} for c in drops[:6]], "why": "analyst"})
    if not problem.classification and np.nanmin(problem.y_raw.astype(float)) >= 0 \
            and problem.spec.metric in ("rmse", "mse", "mae", "r2", "mape"):
        cards.append({"title": "train on log1p(target)", "kind": "target_log", "value": True,
                      "why": "skewed non-negative target"})
    if cats:
        cards.append({"title": "count encoding of categories", "kind": "features",
                      "ops": [{"op": "freq", "col": c} for c in (top_cat or cats)[:6]], "why": "frequency"})
    if len(top_cat) >= 2:
        cards.append({"title": "counts of category pairs", "kind": "features",
                      "ops": [{"op": "group_count", "by": [a, b]} for i, a in enumerate(top_cat)
                              for b in top_cat[i + 1:]], "why": "interactions"})
    for cat in top_cat[:2]:
        own = [c for c in top_num if cat not in c][:3]  # freq__cat is constant within cat groups
        if own:
            cards.append({"title": f"group stats by {cat}", "kind": "features",
                          "ops": [{"op": "group_dev", "by": [cat], "col": c} for c in own],
                          "why": "deviation from group mean"})
    if len(top_num) >= 2:
        pairs = [(a, b) for i, a in enumerate(top_num) for b in top_num[i + 1:]][:4]
        cards.append({"title": "ratios of top numeric features", "kind": "features",
                      "ops": [{"op": "ratio", "a": a, "b": b} for a, b in pairs], "why": "ratios"})
        cards.append({"title": "differences of top numeric features", "kind": "features",
                      "ops": [{"op": "diff", "a": a, "b": b} for a, b in pairs], "why": "differences"})
    cards += [
        {"title": "smaller trees, stronger regularisation", "kind": "params",
         "params": {"num_leaves": 31, "min_child_samples": 50, "reg_lambda": 5.0}, "why": "less overfit"},
        {"title": "bigger trees", "kind": "params", "params": {"num_leaves": 127, "min_child_samples": 10},
         "why": "more capacity"},
        {"title": "fewer features per tree", "kind": "params", "params": {"colsample_bytree": 0.5},
         "why": "decorrelate trees"},
    ]
    if problem.classification:
        balanced_now = problem.spec.metric in ("f1_macro", "f1", "balanced_accuracy")
        cards.append({"title": f"class weights {'off' if balanced_now else 'on'}", "kind": "class_weight",
                      "value": not balanced_now, "why": "imbalance"})
    low = importance[importance <= importance.quantile(0.15)].index.tolist() if len(importance) > 20 else []
    if low:
        cards.append({"title": "drop least important features", "kind": "features",
                      "ops": [{"op": "drop", "col": c} for c in low[:10]], "why": "noise"})
    known = set(frame.columns)
    numeric = numeric_columns(frame)
    return [c for c in (clean_card(card, known, problem, numeric=numeric) for card in cards) if c is not None]


# ---------------------------------------------------------------- critic

def _uses_columns(card: dict, columns: set[str]) -> bool:
    for op in card.get("ops", []):
        named = {op.get(key) for key in ("a", "b", "col")} | set(op.get("by") or [])
        if named & columns:
            return True
        code = op.get("code") or ""
        if any(f"'{c}'" in code or f'"{c}"' in code for c in columns):
            return True
    return False


def critic(llm, cards: list[dict], journal_summary: str, passport: dict, tried: set[str],
           problem, k: int = 2, log=print) -> list[dict]:
    """Pick up to k cards: rules drop repeats and leak-suspect columns, the LLM ranks the rest."""
    leaks = set(passport.get("leak_suspects", []))
    fresh, seen = [], set(tried)
    for card in cards:
        sig = signature(card)
        if sig in seen or (leaks and _uses_columns(card, leaks)):
            continue
        seen.add(sig)
        fresh.append(card)
    if len(fresh) <= k or llm is None or not llm.available:
        return fresh[:k]
    listing = "\n".join(f"{i}: {json.dumps(c, ensure_ascii=False)[:400]}" for i, c in enumerate(fresh))
    user = (f"Metric: {problem.spec.metric}. Experiments so far:\n{journal_summary}\n\n"
            f"Candidate experiments:\n{listing}\n\n"
            f"Pick the {k} most promising and safe ones (no target leakage, not repeats, cheap). "
            'Return JSON {"selected": [indices], "reasons": "one sentence"}.')
    answer = llm.ask_json("critic", SYSTEM, user)
    picked = []
    if isinstance(answer, dict) and isinstance(answer.get("selected"), list):
        for i in answer["selected"]:
            if isinstance(i, int) and 0 <= i < len(fresh) and fresh[i] not in picked:
                picked.append(fresh[i])
        log(f"critic: picked {[c['title'] for c in picked]}; {str(answer.get('reasons', ''))[:200]}")
    else:
        log("critic: unusable reply, falling back to the proposal order")
    for card in fresh:  # top up with the original order if the LLM picked too few
        if len(picked) >= k:
            break
        if card not in picked:
            picked.append(card)
    return picked[:k]
