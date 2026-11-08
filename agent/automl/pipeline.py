"""Agent v2: analyst -> noise calibration -> research loop -> strong models -> blend -> checks.

Time plan (fractions of the time left when v2 starts, up to the soft deadline):
  research until 50%, final models until 80%, refit on all rows until 100%.
Every stage only replaces the answer with a better-validated one; a failing stage falls back.

Stages are separate functions so that each owner edits their own part:
  research_loop (AI engineer) · fit_models / blend_and_check / refit (ML engineer).
"""
import os
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from automl.cv import (LGBM_FAST, LGBM_PARAMS, N_JOBS, CVResult, Problem, answer_from, fit_full, make_problem,
                       make_subproblem, run_cv)
from automl.history import find_entity_key, history_features
from automl.research.journal import signature
from automl.shift import MIN_TOTAL_VARIATION, em_prior_weights, total_variation
from automl.research.rd import COMPONENTS, Toggles, choose_component, component_of, developer, focus_text
from automl.ensemble import blend, greedy_blend, tune_class_weights
from automl.features import (apply_recipe, base_frame, categorical_hints, numeric_columns, op_name,
                             read_dictionary)
from automl.metrics import CLASS_BALANCED
from automl.research.journal import Journal
from automl.research.llm import LLM
from automl.research.roles import KINDS, analyst, critic, data_summary, researcher, rule_ideas
from automl.task import TaskSpec

RESEARCH_SHARE, MODELS_SHARE = 0.50, 0.80
CANDIDATES_PER_ROUND = 2
MIN_V2_SECONDS = 30  # below this v2 cannot even calibrate; keep the v1 answer
MAX_TUNED_CLASSES = 30  # class-weight search cost grows with classes
MAX_RANDOM_TRIALS = 25  # random LightGBM params once the ideas run out
MAX_EXTRA_SEEDS = 3  # extra seeds of the best model families while model time remains
SCREEN_MIN_ROWS = 15000  # below this a full experiment is cheap enough, no screening
HISTORY_SHARE = 0.05  # at most 5% of the v2 budget for the object-key search
SCREEN_MIN_SAMPLE, SCREEN_FRAC = 8000, 0.25  # screening sample: max(8000, 25% of research rows), 3 folds
SCREEN_MARGIN = 0.0  # an idea goes to the full run only if it beats the best on the sample
RESIDUAL_COLUMN = "hist_last"
SHIFT_AUC = 2.0
SHIFT_SAMPLE = 30000
SCREEN_MIN_FULL_SECONDS = 5.0  # screen only when one full experiment costs at least this (AGENT_SCREEN_MIN_SEC)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class State:
    """What the research loop changes: feature recipe, LightGBM params, target/weight switches."""
    recipe: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    target_log: bool = False
    balanced: bool | str = False
    offset: str | None = None

    def with_card(self, card: dict) -> "State":
        kind = card["kind"]
        if kind == "features":
            return replace(self, recipe=self.recipe + card["ops"])
        if kind == "params":
            return replace(self, params={**self.params, **card["params"]})
        if kind == "target_log":
            return replace(self, target_log=card["value"])
        if kind == "class_weight":
            return replace(self, balanced=card["value"])
        raise ValueError(f"unknown card kind {kind!r}; known: {KINDS}")


@dataclass
class Candidate:
    result: CVResult
    kind: str
    params: dict
    state: State


class Workbench:
    """Shared context of one v2 run: data, problem, base features and a frame cache."""

    def __init__(self, train, test, spec, base, problem, log):
        self.train, self.test, self.spec, self.base, self.problem, self.log = train, test, spec, base, problem, log
        self.n_train = len(train)
        self._cache: dict[str, pd.DataFrame] = {}  # recipe -> frame; LLM code ops are not re-run
        self.screen: Problem | None = None  # sample problem for screening ideas (set by calibrate)
        self.best_screen_score: float | None = None
        self.toggles = Toggles()
        self.oof_prior: np.ndarray | None = None  # mean blended OOF probability (label shift)
        self.shift_auc: float | None = None

    def frames(self, state: State, strict: bool = False):
        key = repr(state.recipe)
        if key not in self._cache:
            if len(self._cache) >= 3:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = apply_recipe(self.base, state.recipe, log=self.log, strict=strict)
        frame = self._cache[key]
        return frame, frame.iloc[:self.n_train].reset_index(drop=True), frame.iloc[self.n_train:].reset_index(drop=True)

    def evaluate(self, state: State, deadline: float, seed: int = 42, strict: bool = False,
                 problem: Problem | None = None) -> CVResult:
        _, X, X_test = self.frames(state, strict=strict)
        return run_cv(problem or self.problem, X, X_test, "lgbm", params={**LGBM_FAST, **state.params},
                      seed=seed, deadline=deadline, target_log=state.target_log, balanced=state.balanced,
                      name="lgbm_fast", log=self.log, offset=state.offset)


def _safe(log, what, func, default):
    """Run a role call; any exception is logged and replaced by `default` (roles are optional)."""
    try:
        return func()
    except Exception:
        log(f"{what} failed, continuing without it:\n{traceback.format_exc()}")
        return default


def run_v2(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec, soft_deadline: float,
           hard_deadline: float, log=print, llm: LLM | None = None) -> np.ndarray:
    """Returns predictions in answer format (test row order). Raises if v2 should not replace v1."""
    t0 = time.time()
    total = soft_deadline - t0
    if total < MIN_V2_SECONDS:
        raise RuntimeError(f"only {total:.0f}s left, keeping the v1 answer")
    research_end, models_end = t0 + RESEARCH_SHARE * total, t0 + MODELS_SHARE * total
    out_dir = spec.output_path.parent
    llm = llm if llm is not None else LLM(log, trace_path=out_dir / "agent_llm.jsonl")
    toggles = Toggles.from_env()
    log(f"v2: llm={'on (' + llm.model + ')' if llm.available else 'off (rules only)'}, "
        f"budget {total:.0f}s: research until +{research_end - t0:.0f}s, models until +{models_end - t0:.0f}s; "
        f"rd toggles {toggles}")

    train = train[train[spec.target].notna()].reset_index(drop=True)
    dictionary = read_dictionary(spec, log=log)
    journal = Journal(out_dir / "agent_journal.jsonl", log=log)

    # 2. Analyst: passport from task text + data summary; rules decide what is certain.
    summary = data_summary(train, spec, dictionary)
    columns = [c for c in train.columns if c not in (spec.id_col, spec.target)]
    empty = {"categorical_columns": [], "drop_columns": [], "leak_suspects": [], "entity_key_columns": [],
             "notes": "", "feature_ideas": []}
    passport = _safe(log, "analyst", lambda: analyst(llm, _read_task_md(spec, log), summary, columns, log=log), empty)
    hints = categorical_hints(spec, train, dictionary)
    log(f"analyst: categorical by dictionary={hints}, by LLM={passport['categorical_columns']}, "
        f"drop advice={passport['drop_columns']}, leak suspects={passport['leak_suspects']}, "
        f"entity key={passport['entity_key_columns']}")
    base, notes = base_frame(train, test, spec, categorical=hints)
    for note in notes:
        log(f"base: {note}")

    problem = make_problem(train, spec, test=test)
    log(f"problem: {len(problem.research_idx)} research rows, {len(problem.holdout_idx)} holdout rows; "
        f"validation={problem.scheme}, metadata={problem.metadata}")
    skip = history_skip_reason(toggles, problem, spec)
    if skip:
        log(f"history: skipped ({skip})")
    else:
        deadline = time.time() + HISTORY_SHARE * total
        base = _safe(log, "history", lambda: _add_history(train, test, spec, problem, base, passport, log,
                                                          deadline), base)
    bench = Workbench(train, test, spec, base, problem, log)
    bench.toggles = toggles
    if problem.classification and spec.metric in CLASS_BALANCED and spec.pred_format == "label":
        bench.shift_auc = _safe(log, "shift check", lambda: adversarial_auc(base, len(train)), None)
        log(f"shift: train/test adversarial AUC = {bench.shift_auc}")
    start_state = State(target_log=spec.metric == "rmsle", balanced=spec.metric in CLASS_BALANCED)

    base_result, threshold = calibrate(bench, start_state, research_end, journal)
    best, best_state = research_loop(bench, llm, passport, summary, journal, base_result, start_state,
                                     threshold, research_end)
    candidates = fit_models(bench, best, best_state, models_end, total)
    final, class_weights = blend_and_check(bench, candidates, base_result, threshold, journal)
    if final is None:  # the v2 base model, not v1: it has the history features and temporal tuning
        log("final: blend lost to the base model on the check rows -> refit the v2 base model")
        final = ([Candidate(base_result, "lgbm", {**LGBM_FAST, **start_state.params}, start_state)], [1.0])
    test_pred = refit(bench, final, soft_deadline, hard_deadline)
    skip = label_shift_skip_reason(toggles, problem, spec, bench.oof_prior)
    if skip:
        log(f"label shift: skipped ({skip})")
    else:
        class_weights = _safe(log, "label shift", lambda: _label_shift(bench, test_pred, class_weights),
                              class_weights)
    return answer_from(problem, test_pred, class_weights)


def history_skip_reason(toggles, problem, spec) -> str | None:
    if not toggles.history:
        return "AGENT_HISTORY=0"
    if problem.scheme != "temporal":
        return f"validation is {problem.scheme}, not a future test period"
    if spec.task_type not in ("regression", "binary"):
        return f"{spec.task_type}: one past target value does not describe several classes"
    return None


def label_shift_skip_reason(toggles, problem, spec, oof_prior) -> str | None:
    if not toggles.label_shift:
        return "AGENT_LABEL_SHIFT=0"
    if not problem.classification:
        return "not a classification task"
    if spec.pred_format != "label":
        return f"answers are {spec.pred_format}, priors only matter for labels"
    if oof_prior is None:
        return "no OOF class prior from the blend"
    return None


def _add_history(train, test, spec, problem, base, passport, log, deadline=float("inf")):
    """Entity history (automl/history.py): leak-free features from the object's earlier periods."""
    started = time.time()
    key, info = find_entity_key(train, test, problem.time_column, {spec.id_col, spec.target},
                                proposed=passport.get("entity_key_columns") or None, log=log, deadline=deadline)
    if key is None:
        log(f"history: no object key ({info.get('reason')}); column overlap {info.get('overlap')} "
            f"in {time.time() - started:.0f}s")
        return base
    feats = history_features(train, test, key, problem.time_column, problem.y)
    base = base.copy()
    for col in feats.columns:
        base[col] = feats[col].to_numpy()
    known = feats["hist_last"].iloc[len(train):].notna().mean()
    log(f"history: key {key}, test coverage {info['coverage']:.2f}, persistence {info['persistence']:.2f}, "
        f"rows/object {info['rows_per_object']:.2f}; test rows with history {known:.0%} "
        f"({time.time() - started:.0f}s)")
    return base


def _label_shift(bench, test_pred, class_weights):
    """EM prior re-estimation on test probabilities (automl/shift.py), combined with tuned weights."""
    weights, test_prior, iterations = em_prior_weights(bench.oof_prior, test_pred)
    tv = total_variation(test_prior, bench.oof_prior / bench.oof_prior.sum())
    shares = ", ".join(f"{c}: {s:.3f}->{t:.3f}" for c, s, t in
                       zip(bench.problem.classes, bench.oof_prior / bench.oof_prior.sum(), test_prior))
    bench.log(f"label shift: total variation {tv:.3f} after {iterations} EM steps; {shares}")
    if tv < MIN_TOTAL_VARIATION:
        bench.log("label shift: too small, not applied")
        return class_weights
    return weights if class_weights is None else weights * class_weights


# ---------------------------------------------------------------- stage 3: calibration

def calibrate(bench: Workbench, state: State, research_end: float, journal: Journal):
    """Measure seed sensitivity as a practical search tolerance, not statistical significance."""
    base_result = bench.evaluate(state, research_end)
    if not _complete_result(base_result, bench.problem):
        raise RuntimeError("base model did not finish all folds: no time for v2")
    noise = 0.0
    if time.time() + base_result.seconds < research_end - 2 * base_result.seconds:
        second = bench.evaluate(state, research_end, seed=7)
        if _complete_result(second, bench.problem):
            noise = abs(second.score - base_result.score)
    threshold = max(noise, 1e-4 * abs(base_result.score), 1e-6)
    bench.log(f"calibration: base {bench.spec.metric}={base_result.score:.5f}, noise={noise:.5f}, "
              f"threshold={threshold:.5f} ({base_result.seconds:.0f}s per experiment)")
    journal.add({"title": "base model", "kind": "base"}, base_result.score, None, "base",
                base_result.seconds, "rules")

    # Screening problem (RD-Agent "debug on a sample"): ideas are first tried on a subset with 3 folds;
    # only those that beat the current best there get the full 5-fold evaluation.
    n_research = len(bench.problem.research_idx)
    min_seconds = _env_float("AGENT_SCREEN_MIN_SEC", SCREEN_MIN_FULL_SECONDS)
    if bench.toggles.screen and getattr(bench.problem, "scheme", "random") != "random":
        # The screening sample uses random folds; on a temporal problem that would mix past and future.
        bench.log(f"screening: off ({bench.problem.scheme} validation)")
    elif bench.toggles.screen and n_research >= SCREEN_MIN_ROWS and base_result.seconds < min_seconds:
        # Cheap experiments gain nothing from screening, and a small sample can hide signal that
        # needs volume (high-cardinality IDs: amazon_access lost 4 points with screening on).
        bench.log(f"screening: off (a full experiment takes {base_result.seconds:.1f}s < {min_seconds:.0f}s)")
    elif bench.toggles.screen and n_research >= SCREEN_MIN_ROWS:
        try:  # screening is optional: any problem here just switches it off
            rows = max(SCREEN_MIN_SAMPLE, int(SCREEN_FRAC * n_research))
            bench.screen = make_subproblem(bench.problem, rows)
            screen_base = bench.evaluate(state, research_end, problem=bench.screen)
            if screen_base.folds == len(bench.screen.folds):
                bench.best_screen_score = screen_base.score
                bench.log(f"screening: {len(bench.screen.research_idx)} rows x {len(bench.screen.folds)} folds, "
                          f"base {screen_base.score:.5f} in {screen_base.seconds:.0f}s "
                          f"(full run {base_result.seconds:.0f}s)")
            else:
                bench.screen = None
                bench.log("screening: off (sample base run cut by the deadline)")
        except Exception as error:
            bench.screen = None
            bench.log(f"screening: off after {error!r}"[:300])
    elif bench.toggles.screen:
        bench.log(f"screening: off ({n_research} research rows < {SCREEN_MIN_ROWS})")
    return base_result, threshold


# ---------------------------------------------------------------- stage 4: research (AI engineer)

def research_loop(bench: Workbench, llm, passport, summary, journal: Journal, best: CVResult,
                  best_state: State, threshold: float, research_end: float):
    log, problem = bench.log, bench.problem
    base_score = best.score
    pool = _safe(log, "rule ideas", lambda: rule_ideas(problem, bench.base, best.importance, passport), [])
    rounds, random_left, rng = 0, MAX_RANDOM_TRIALS, np.random.default_rng(0)
    toggles = bench.toggles
    components = tuple(c for c in COMPONENTS if c != "develop" or toggles.developer)
    while time.time() + 1.3 * best.seconds < research_end:
        rounds += 1
        frame_now = bench.frames(best_state)[0]
        focus, component = "", None
        if toggles.components and llm is not None and llm.available:
            component = choose_component(journal.entries, components)
            focus = focus_text(component, problem.spec.target)
            log(f"research round {rounds}: component = {component}")
        llm_cards = _safe(log, "researcher", lambda: researcher(
            llm, passport, summary, journal.summary(), best.importance, list(frame_now.columns), problem,
            log=log, numeric=numeric_columns(frame_now), focus=focus, develop_enabled=toggles.developer), [])
        for card in llm_cards:
            card["source"] = "llm"
            card["round"], card["component"] = rounds, component
        pool_now = llm_cards + [c for c in pool if c not in llm_cards]
        picked = _safe(log, "critic", lambda: critic(llm, pool_now, journal.summary(), passport, journal.tried,
                                                      problem, k=CANDIDATES_PER_ROUND, log=log), pool_now[:2])
        if not picked:  # second wave: rules on the current best features and importance
            frame_now = bench.frames(best_state)[0]
            pool = _safe(log, "rule ideas", lambda: rule_ideas(problem, frame_now, best.importance, passport), [])
            picked = critic(None, pool, "", passport, journal.tried, problem, k=CANDIDATES_PER_ROUND, log=log)
        if not picked and random_left > 0:  # ideas exhausted: random search over LightGBM params
            random_left -= 1
            picked = [_random_params_card(rng)]
        if not picked:
            log("research: no fresh ideas left")
            break
        for card in picked:
            if card in pool:
                pool.remove(card)
            if time.time() + 1.3 * best.seconds > research_end:
                break
            if card["kind"] == "develop":  # RD-Agent Developer: hypothesis -> code, fixed on a sample
                card = _develop(bench, llm, card, best_state, journal, research_end)
                if card is None:
                    continue
            best, best_state = _try_card(bench, card, best, best_state, threshold, research_end, journal)
    log(f"research done: {rounds} rounds, {len(journal.entries) - 1} experiments, "
        f"best {bench.spec.metric}={best.score:.5f} (base {base_score:.5f}); llm {llm.stats()}")
    log("research summary: " + _research_summary(journal))
    log(f"recipe: {[op_name(op) for op in best_state.recipe]}, params={best_state.params}, "
        f"target_log={best_state.target_log}, balanced={best_state.balanced}")
    return best, best_state


def _random_params_card(rng) -> dict:
    params = {"num_leaves": int(rng.choice([15, 31, 63, 127, 255])),
              "min_child_samples": int(rng.choice([5, 10, 20, 50, 100, 200])),
              "colsample_bytree": float(rng.choice([0.3, 0.5, 0.7, 0.9])),
              "subsample": float(rng.choice([0.6, 0.8, 1.0])),
              "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 20.0]))}
    return {"title": f"random params {params}", "kind": "params", "params": params, "why": "random search",
            "source": "random"}


def _research_summary(journal: Journal) -> str:
    """One line to judge the RD ideas: verdict counts, screening time, developer and component stats."""
    entries = [e for e in journal.entries if e["kind"] not in ("base", "final")]
    verdicts = {}
    for e in entries:
        verdicts[e["verdict"]] = verdicts.get(e["verdict"], 0) + 1
    screened_sec = sum(e["seconds"] for e in entries if e["verdict"] == "screened")
    dev = [e for e in entries if e["source"] == "llm-dev"]
    comps = {}
    for e in entries:
        comp = component_of(e)
        if comp:
            tries, kept = comps.get(comp, (0, 0))
            comps[comp] = (tries + 1, kept + (e["verdict"] == "kept"))
    return (f"verdicts={verdicts}, screened {screened_sec:.0f}s total, "
            f"developer ok={sum(1 for e in dev if e['kind'] == 'features')} "
            f"failed={sum(1 for e in dev if e['kind'] == 'develop' and e['verdict'] == 'failed')} "
            f"kept={sum(1 for e in dev if e['verdict'] == 'kept')}, components (tries, kept)={comps}")


def _develop(bench, llm, card, best_state, journal, deadline):
    """Developer role: returns a features card with the written code_block op, or None."""
    started = time.time()
    journal.tried.add(signature(card))  # the same hypothesis is not developed twice
    frame = bench.frames(best_state)[0]
    stats = {}
    op = _safe(bench.log, "developer", lambda: developer(
        llm, card, frame, numeric_columns(frame), bench.spec.target, log=bench.log, deadline=deadline,
        stats=stats), None)
    extra = {"attempts": stats.get("attempts"), "round": card.get("round"), "component": card.get("component")}
    if op is None:
        verdict = "skipped" if stats.get("outcome") == "skipped" else "failed"
        journal.add(card, None, None, verdict, time.time() - started, "llm-dev",
                    "llm unavailable or no time" if verdict == "skipped" else "developer could not write working code",
                    extra=extra)
        return None
    return {"title": card["title"], "why": card.get("why", ""), "kind": "features", "ops": [op],
            "hypothesis": card.get("hypothesis"), "source": "llm-dev", **extra}


def _try_card(bench, card, best, best_state, threshold, research_end, journal):
    """Evaluate one idea; keep it only if it beats the best by more than the noise threshold.
    With screening on, the idea must first beat the best on the sample problem."""
    source = card.get("source", "rules")
    extra = {k: card.get(k) for k in ("round", "component", "attempts") if card.get(k) is not None}
    started = time.time()
    try:
        candidate_state = best_state.with_card(card)
        screen_result = None
        if bench.screen is not None:
            screen_result = bench.evaluate(candidate_state, research_end, strict=True, problem=bench.screen)
            if screen_result.folds < len(bench.screen.folds):
                journal.add(card, screen_result.score, None, "partial", screen_result.seconds, source,
                            "sample run cut by the deadline", extra=extra)
                return best, best_state
            screen_gain = bench.problem.gain(screen_result.score, bench.best_screen_score)
            extra["screen_gain"] = screen_gain
            if screen_gain <= SCREEN_MARGIN:
                journal.add(card, screen_result.score, screen_gain, "screened", screen_result.seconds, source,
                            "not better than the best on the sample", extra=extra)
                bench.log(f"research [{source}]: {card['title']!r} screened out "
                          f"(sample gain {screen_gain:+.5f}, {screen_result.seconds:.0f}s)")
                return best, best_state
        result = bench.evaluate(candidate_state, research_end, strict=True)
    except Exception as error:
        journal.add(card, None, None, "failed", time.time() - started, source, repr(error)[:200], extra=extra)
        bench.log(f"research: {card.get('title')!r} failed: {error!r}"[:300])
        return best, best_state
    if not _complete_result(result, bench.problem):  # scored on other rows: not comparable
        journal.add(card, result.score, None, "partial", result.seconds, source, "cut by the deadline", extra=extra)
        bench.log(f"research: {card['title']!r} cut by the deadline, not compared")
        return best, best_state
    gain = bench.problem.gain(result.score, best.score)
    verdict = "kept" if gain > threshold else "rejected"
    journal.add(card, result.score, gain, verdict, result.seconds, source, extra=extra)
    sample_note = f", sample gain {extra['screen_gain']:+.5f}" if "screen_gain" in extra else ""
    bench.log(f"research #{len(journal.entries) - 1} [{source}]: {card['title']!r} -> {result.score:.5f} "
              f"(gain {gain:+.5f}{sample_note}) {verdict}")
    if verdict == "kept" and screen_result is not None:
        bench.best_screen_score = screen_result.score
    return (result, candidate_state) if verdict == "kept" else (best, best_state)


# ---------------------------------------------------------------- stage 5-7: models (ML engineer)

def fit_models(bench: Workbench, best: CVResult, best_state: State, models_end: float, total: float):
    """Strong model families on the best recipe; only complete (all-fold) results are kept."""
    log, problem = bench.log, bench.problem
    _, X, X_test = bench.frames(best_state)
    candidates = [Candidate(best, "lgbm", {**LGBM_FAST, **best_state.params}, best_state)]
    variants = [("lgbm", {**LGBM_PARAMS, **best_state.params}, best_state, "lgbm")]
    if problem.classification:
        variants.append(("lgbm", {**LGBM_PARAMS, **best_state.params, "num_leaves": 31,
                                  "min_child_samples": 80, "reg_lambda": 5.0,
                                  "learning_rate": 0.07, "n_estimators": 800},
                         replace(best_state, balanced="sqrt"), "lgbm_regularized_sqrt"))
    else:
        variants.append(("lgbm", {**LGBM_PARAMS, **best_state.params, "num_leaves": 15,
                                  "min_child_samples": 150, "reg_lambda": 10.0},
                         best_state, "lgbm_regularized"))
    variants.extend([("catboost", {}, best_state, "catboost"), ("xgb", {}, best_state, "xgb")])
    if not problem.classification and not best_state.target_log and RESIDUAL_COLUMN in X.columns:
        resid = replace(best_state, offset=RESIDUAL_COLUMN)
        variants.extend([("lgbm", {**LGBM_PARAMS, **best_state.params}, resid, "lgbm_resid"),
                         ("xgb", {}, resid, "xgb_resid")])
    for kind, params, candidate_state, name in variants:
        if time.time() > models_end - 0.05 * total:
            log(f"models: skip {kind}, no time")
            continue
        try:
            res = run_cv(problem, X, X_test, kind, params=params, deadline=models_end,
                         target_log=candidate_state.target_log, balanced=candidate_state.balanced, name=name,
                         log=log, offset=candidate_state.offset)
        except Exception as error:
            log(f"models: {kind} failed: {error!r}"[:300])
            continue
        log(f"models: {kind} {bench.spec.metric}={res.score:.5f} in {res.seconds:.0f}s ({res.folds} folds)")
        if not _complete_result(res, problem):  # partial OOF must not change comparison rows
            log(f"models: {kind} finished only {res.folds} folds, not blended")
            continue
        candidates.append(Candidate(res, kind, params, candidate_state))

    # Spare model time: more seeds of the two best families (seed bagging, a safe small gain).
    ranked = sorted(candidates[1:], key=lambda c: c.result.score, reverse=problem.better(1, 0))[:2]
    for seed in range(1, MAX_EXTRA_SEEDS + 1):
        for c in ranked:
            if time.time() + 1.3 * c.result.seconds > models_end:
                continue
            try:
                res = run_cv(problem, X, X_test, c.kind, params=c.params, seed=42 + seed, deadline=models_end,
                             target_log=c.state.target_log, balanced=c.state.balanced,
                             name=f"{c.result.name}_s{seed}", log=log, offset=c.state.offset)
            except Exception as error:
                log(f"models: {c.kind} seed {seed} failed: {error!r}"[:300])
                continue
            if _complete_result(res, problem):
                log(f"models: {res.name} {bench.spec.metric}={res.score:.5f} in {res.seconds:.0f}s")
                candidates.append(Candidate(res, c.kind, c.params, c.state))
    return candidates


def blend_and_check(bench: Workbench, candidates, base_result: CVResult, threshold: float, journal: Journal):
    """Blend on OOF, tune class weights, then verify against the base model on rows the
    choices never saw (holdout), or on OOF when the data is too small for a holdout.
    Returns ((candidates, weights), class_weights), or (None, None) to keep the base model."""
    log, problem, spec = bench.log, bench.problem, bench.spec
    rows_mask = np.ones(len(problem.research_idx), dtype=bool)
    for c in candidates + [Candidate(base_result, "lgbm", {}, State())]:
        oof = c.result.oof
        rows_mask &= ~np.isnan(oof if oof.ndim == 1 else oof[:, 0])
    rows = problem.research_idx[rows_mask]
    oofs = [c.result.oof[rows_mask] for c in candidates]
    weights = greedy_blend(problem, oofs, rows)
    shifted = _env_float("AGENT_HEDGE", float(bench.shift_auc is not None and bench.shift_auc >= SHIFT_AUC)) > 0
    if shifted:
        weights = _hedge_balanced(bench, candidates, oofs, rows, weights)
    blended_oof = blend(oofs, weights)
    if problem.classification:
        bench.oof_prior = np.asarray(blended_oof, float).mean(axis=0)
    class_weights = None
    if problem.n_out <= MAX_TUNED_CLASSES and not shifted:
        class_weights = tune_class_weights(problem, blended_oof, rows)
    log("blend: " + ", ".join(f"{c.result.name}={w:.2f}" for c, w in zip(candidates, weights) if w > 0)
        + f"; OOF {spec.metric}={problem.score_pred(blended_oof, rows):.5f}"
        + (f", with class weights {problem.score_pred(blended_oof, rows, class_weights):.5f}"
           if class_weights is not None else ""))

    if len(problem.holdout_idx):
        hold = problem.holdout_idx
        blended = blend([c.result.holdout for c in candidates], weights)
        reference = base_result.holdout
        where = "holdout"
    else:  # small data: the blend was fit on these OOF rows, so this check is optimistic
        hold = rows
        blended = blended_oof
        reference = base_result.oof[rows_mask]
        where = "OOF (no holdout)"
    s_blend = problem.score_pred(blended, hold)
    s_tuned = problem.score_pred(blended, hold, class_weights) if class_weights is not None else None
    s_base = problem.score_pred(reference, hold)
    log(f"{where}: base={s_base:.5f}, blend={s_blend:.5f}"
        + (f", blend+class weights={s_tuned:.5f}" if s_tuned is not None else ""))
    if s_tuned is not None and not problem.better(s_tuned, s_blend):
        class_weights = None
        log(f"{where}: class weights do not help, dropped")
    final_score = s_tuned if class_weights is not None else s_blend
    keep_base = problem.gain(final_score, s_base) < -threshold
    journal.add({"title": "final", "kind": "final"}, final_score, problem.gain(final_score, s_base),
                "base" if keep_base else "blend", 0, "rules",
                f"{where}; weights={[round(float(w), 2) for w in weights]}")
    if keep_base:
        log(f"{where}: v2 blend is worse than the base model -> keep base model")
        return None, None
    return (candidates, weights), class_weights


def adversarial_auc(base: pd.DataFrame, n_train: int, seed: int = 0) -> float:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    rng = np.random.default_rng(seed)
    tr = np.arange(n_train)
    te = np.arange(n_train, len(base))
    tr = rng.choice(tr, min(len(tr), SHIFT_SAMPLE), replace=False)
    te = rng.choice(te, min(len(te), SHIFT_SAMPLE), replace=False)
    X = base.iloc[np.r_[tr, te]].reset_index(drop=True)
    X = X.rename(columns={c: f"f{i}" for i, c in enumerate(X.columns)})
    z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31, n_jobs=N_JOBS, verbose=-1)
    p = cross_val_predict(model, X, z, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return round(float(roc_auc_score(z, p)), 4)


def _hedge_balanced(bench, candidates, oofs, rows, weights):
    problem = bench.problem
    if not problem.classification or bench.spec.metric not in CLASS_BALANCED:
        return weights
    balanced = [i for i, c in enumerate(candidates) if c.state.balanced is True]
    if not balanced:
        return weights
    best = max(balanced, key=lambda i: problem.score_pred(oofs[i], rows) * (1 if problem.better(1, 0) else -1))
    hedged = 0.5 * np.asarray(weights, float)
    hedged[best] += 0.5
    bench.log(f"hedge: half of the blend to {candidates[best].result.name}")
    return hedged


def refit(bench: Workbench, final, soft_deadline: float, hard_deadline: float) -> np.ndarray:
    """Retrain blended models on all rows (research + holdout) while time allows; models that
    do not fit in time keep their research-time test predictions."""
    candidates, weights = final
    log, spec = bench.log, bench.spec
    tests = [c.result.test for c in candidates]
    # fit_full consumes the complete y/X, regardless of the research/holdout indices.
    full = bench.problem
    order = [i for i in sorted(range(len(candidates)), key=lambda i: -weights[i]) if weights[i] > 0]
    for k, i in enumerate(order):
        c = candidates[i]
        # extra seeds of one model must not take the time the next models need for their refit
        bag_deadline = time.time() + (soft_deadline - time.time()) / (len(order) - k)
        estimate = c.result.seconds / max(c.result.folds, 1) * 1.5
        if time.time() + estimate > soft_deadline:
            log(f"refit: skip {c.result.name}, needs ~{estimate:.0f}s")
            continue
        try:
            _, X, X_test = bench.frames(c.state)
            iterations = refit_iterations(c.result.iterations, bench.problem.scheme)
            tests[i], used = bag_seeds(
                lambda seed: fit_full(full, X, X_test, c.kind, params=c.params, iterations=iterations,
                                      deadline=hard_deadline, target_log=c.state.target_log,
                                      balanced=c.state.balanced, seed=seed, log=log, offset=c.state.offset),
                base_seed=42, n_seeds=REFIT_SEEDS, soft_deadline=bag_deadline)
            log(f"refit: {c.result.name} on all {len(bench.train)} rows, iterations={iterations}, "
                f"mean of {used} seeds")
        except Exception as error:
            log(f"refit: {c.result.name} failed: {error!r}"[:300])
    return blend(tests, weights)


REFIT_SEEDS = 10  # refits are cheap (tens of trees on temporal tasks); seeds average out their noise


def bag_seeds(fit, base_seed: int, n_seeds: int, soft_deadline: float):
    """Average fit(seed) over up to n_seeds seeds while the next one still fits before soft_deadline.
    On wids_future one refit scored anywhere from 1.9 to 19.1 depending on the seed; the mean of
    10 seeds scored 12.2 against 10.9 for a single seed on average (wids_seen: 32.3 against 31.1).
    The first fit always runs and its errors propagate; later failures keep the mean so far."""
    started = time.time()
    total = np.asarray(fit(base_seed), dtype=float)
    used, per_fit = 1, time.time() - started
    while used < n_seeds and time.time() + 1.3 * per_fit < soft_deadline:
        try:
            total = total + np.asarray(fit(base_seed + 1000 * used), dtype=float)
        except Exception:
            break
        used += 1
    return total / used, used


def refit_iterations(counts: list[int] | None, scheme: str) -> int | None:
    """Tree count for the full refit. Temporal folds grow (period 1 -> 2, periods 1-2 -> 3, ...):
    early folds see little data and stop early, so the latest fold is the closest to the refit.
    Random folds are alike, so their median is used."""
    if not counts:
        return None
    return int(counts[-1]) if scheme == "temporal" else int(np.median(counts))


def _complete_result(result: CVResult, problem: Problem) -> bool:
    """Completed candidates must cover exactly the predeclared OOF validation rows."""
    finite = np.isfinite(result.oof) if result.oof.ndim == 1 else np.isfinite(result.oof).all(axis=1)
    expected = getattr(problem, "oof_mask", None)
    if expected is None:
        expected = np.zeros(len(problem.research_idx), dtype=bool)
        for _, valid in problem.folds:
            expected[valid] = True
    return result.folds == len(problem.folds) and np.array_equal(finite, expected)


def _read_task_md(spec: TaskSpec, log) -> str:
    try:
        return Path(spec.task_path).read_text(encoding="utf-8", errors="replace") if spec.task_path else ""
    except OSError as error:
        log(f"task.md unreadable for the analyst: {error!r}")
        return ""
