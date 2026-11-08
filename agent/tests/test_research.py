import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from automl.cv import make_problem
from automl.ensemble import greedy_blend, tune_class_weights
from automl.pipeline import run_v2
from automl.research.llm import DEFAULT_MODEL, LLM, LLM_DEFAULT_MAX_TOKENS, extract_json
from automl.research.roles import analyst, clean_card, critic, researcher, rule_ideas
from automl.task import build_spec
from tests.helpers import make_task


class FakeLLM:
    """Scripted LLM: answers per role; records calls."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.model = "fake"
        self.calls = []

    @property
    def available(self):
        return True

    def ask_json(self, role, system, user):
        self.calls.append(role)
        answer = self.answers.get(role)
        return answer(user) if callable(answer) else answer

    def stats(self):
        return {"calls": len(self.calls)}


class LLMClientTests(unittest.TestCase):
    def test_extract_json_variants(self):
        self.assertEqual(extract_json('<think>hmm {"a": 0}</think>\n```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json('Sure! [1, 2] done'), [1, 2])
        self.assertIsNone(extract_json("no json here"))

    def test_min_interval_and_consecutive_failures(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "AGENT_LLM_MIN_INTERVAL_SEC": "1"}):
            llm = LLM(log=lambda m: None)
        llm._client = object()  # pretend credentials exist
        replies = iter(['{"ok": 1}', "garbage", "garbage", "garbage", "garbage", '{"ok": 2}'])
        llm._complete = lambda system, user: next(replies)
        started = time.time()
        self.assertEqual(llm.ask_json("a", "s", "u"), {"ok": 1})
        self.assertIsNone(llm.ask_json("a", "s", "u"))
        self.assertGreaterEqual(time.time() - started, 1.0)  # second call waited for the interval
        for _ in range(3):
            llm.ask_json("a", "s", "u")
        self.assertFalse(llm.available)  # 4 failures in a row switch the LLM off

    def test_unavailable_without_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "OPENAI_MODEL": ""}):
            llm = LLM(log=lambda m: None)
        self.assertFalse(llm.available)
        self.assertIsNone(llm.ask_json("analyst", "s", "u"))

    def test_reply_cut_by_reasoning_is_retried_with_a_larger_limit(self):
        # Qwen3 spends the limit on hidden reasoning and returns empty content with finish_reason=length.
        from types import SimpleNamespace as NS
        calls = []

        def create(**kwargs):
            calls.append(kwargs["max_tokens"])
            cut = len(calls) == 1
            return NS(choices=[NS(message=NS(content="" if cut else '{"ok": 1}'),
                                  finish_reason="length" if cut else "stop")],
                      usage=NS(prompt_tokens=10, completion_tokens=kwargs["max_tokens"] if cut else 50))

        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "AGENT_LLM_MAX_TOKENS": "1000"}):
            llm = LLM(log=lambda m: None)
        llm._client = NS(chat=NS(completions=NS(create=create)))
        self.assertEqual(llm.ask_json("analyst", "s", "u"), {"ok": 1})
        self.assertEqual(calls, [1000, 2000])
        self.assertGreaterEqual(LLM_DEFAULT_MAX_TOKENS, 8000)

    def test_key_without_model_name_still_enables_llm(self):
        # The platform proxy substitutes the task's model, so OPENAI_MODEL may be absent.
        env = {"OPENAI_API_KEY": "test-key", "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
        with patch.dict(os.environ, env):
            os.environ.pop("OPENAI_MODEL", None)
            llm = LLM(log=lambda m: None)
        self.assertTrue(llm.available)
        self.assertEqual(llm.model, DEFAULT_MODEL)


class RolesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = build_spec(make_task(Path(self.tmp.name), "regression"))
        self.train = pd.read_csv(self.spec.train_path)
        self.problem = make_problem(self.train, self.spec)

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyst_drops_unknown_columns(self):
        llm = FakeLLM({"analyst": {"categorical_columns": ["cat", "ghost"], "leak_suspects": ["x1"],
                                   "notes": "n", "feature_ideas": ["a"]}})
        passport = analyst(llm, "task", "summary", ["x1", "x2", "cat"])
        self.assertEqual(passport["categorical_columns"], ["cat"])
        self.assertEqual(passport["leak_suspects"], ["x1"])

    def test_researcher_validates_cards(self):
        cards = [{"title": "ok", "kind": "features", "ops": [{"op": "ratio", "a": "x1", "b": "x2"}]},
                 {"title": "bad column", "kind": "features", "ops": [{"op": "ratio", "a": "x1", "b": "zz"}]},
                 {"title": "params", "kind": "params", "params": {"num_leaves": 10000, "evil": 1}},
                 {"title": "log", "kind": "target_log", "value": True},
                 "garbage"]
        llm = FakeLLM({"researcher": cards})
        out = researcher(llm, {}, "s", "j", pd.Series({"x1": 1.0}), ["x1", "x2", "cat"], self.problem)
        self.assertEqual([c["title"] for c in out], ["ok", "params", "log"])
        self.assertEqual(out[1]["params"], {"num_leaves": 255})

    def test_critic_skips_tried_and_leaky_and_survives_garbage(self):
        cards = [clean_card({"title": t, "kind": "features", "ops": [{"op": "log1p", "col": c}]},
                            {"x1", "x2"}, self.problem) for t, c in (("a", "x1"), ("b", "x2"))]
        picked = critic(FakeLLM({"critic": "nonsense"}), cards, "", {"leak_suspects": ["x1"]}, set(),
                        self.problem, k=2)
        self.assertEqual([c["title"] for c in picked], ["b"])

    def test_researcher_prompt_forbids_target_and_drops_target_ops(self):
        prompts = []
        cards = [{"title": "leak", "kind": "features", "ops": [{"op": "freq", "col": "price"}]},
                 {"title": "ok", "kind": "features", "ops": [{"op": "freq", "col": "cat"}]}]
        llm = FakeLLM({"researcher": lambda user: prompts.append(user) or cards})
        out = researcher(llm, {}, "s", "j", pd.Series({"x1": 1.0}), ["x1", "x2", "cat"], self.problem,
                         n=2, numeric={"x1", "x2"}, log=lambda m: None)
        self.assertEqual([c["title"] for c in out], ["ok"])
        self.assertIn("'price' is NOT available", prompts[0])
        self.assertIn("Categorical feature columns: cat", prompts[0])

    def test_rule_ideas_are_valid(self):
        from automl.features import base_frame
        frame, _ = base_frame(self.train, pd.read_csv(self.spec.test_path), self.spec)
        importance = pd.Series(1.0, index=frame.columns)
        cards = rule_ideas(self.problem, frame, importance, {"categorical_columns": []})
        self.assertGreaterEqual(len(cards), 4)
        self.assertTrue(all(c["kind"] in {"features", "params", "target_log", "class_weight"} for c in cards))


class EnsembleTests(unittest.TestCase):
    def test_greedy_blend_prefers_better_model(self):
        tmp = tempfile.TemporaryDirectory()
        spec = build_spec(make_task(Path(tmp.name), "regression"))
        train = pd.read_csv(spec.train_path)
        problem = make_problem(train, spec)
        rows = problem.research_idx
        y = problem.y_raw[rows].astype(float)
        rng = np.random.default_rng(0)
        good, bad = y + rng.normal(0, 0.5, len(y)), y + rng.normal(0, 5, len(y))
        weights = greedy_blend(problem, [bad, good], rows)
        self.assertGreater(weights[1], weights[0])
        tmp.cleanup()

    def test_class_weights_improve_macro_f1(self):
        tmp = tempfile.TemporaryDirectory()
        spec = build_spec(make_task(Path(tmp.name), "multiclass", rare_rows=40))
        train = pd.read_csv(spec.train_path)
        problem = make_problem(train, spec)
        rows = problem.research_idx
        k = problem.n_out
        proba = np.full((len(rows), k), 0.1)
        proba[np.arange(len(rows)), problem.y[rows]] += 0.3
        rare = problem.classes.index("rare")
        proba[:, rare] *= 0.2  # the model never predicts the rare class
        before = problem.score_pred(proba, rows)
        weights = tune_class_weights(problem, proba, rows)
        self.assertGreater(problem.score_pred(proba, rows, weights), before)
        tmp.cleanup()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, kind, llm, budget=45, **kwargs):
        spec = build_spec(make_task(self.root, kind, n_train=4000, n_test=500, **kwargs))
        train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
        now = time.time()
        pred = run_v2(train, test, spec, now + budget, now + budget + 10, log=lambda m: None, llm=llm)
        self.assertEqual(len(pred), len(test))
        self.assertLess(time.time() - now, budget + 30)
        return spec, pred

    def test_rules_only_regression(self):
        spec, pred = self.run_pipeline("regression", llm=FakeLLM({}))
        self.assertTrue(np.isfinite(np.asarray(pred, float)).all())
        journal = (self.root / "agent_journal.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertGreaterEqual(len(journal), 3)

    def test_llm_cards_are_tried(self):
        card = {"title": "llm idea", "kind": "features",
                "ops": [{"op": "code", "name": "x1sq", "code": "def feature(df):\n    return df['x1'] ** 2"}]}
        llm = FakeLLM({"analyst": {"categorical_columns": [], "notes": "synthetic"},
                       "researcher": [card], "critic": {"selected": [0]}})
        self.run_pipeline("multiclass", llm=llm, budget=120)
        self.assertIn("researcher", llm.calls)
        journal = (self.root / "agent_journal.jsonl").read_text(encoding="utf-8")
        self.assertIn("llm idea", journal)

    def test_broken_llm_ideas_are_failed_or_dropped_not_fatal(self):
        cards = [
            {"title": "runtime error", "kind": "features",
             "ops": [{"op": "code", "name": "boom", "code": "def feature(df):\n    return df['nope']"}]},
            {"title": "import", "kind": "features",
             "ops": [{"op": "code", "name": "evil", "code": "import os\ndef feature(df):\n    return df['x1']"}]},
            {"title": "garbage types", "kind": "features", "ops": [{"op": "ratio", "a": ["x1"], "b": 5}]},
            {"title": "unknown kind", "kind": "teleport"},
        ]
        llm = FakeLLM({"analyst": "not json", "researcher": cards, "critic": {"selected": [0, 99, "x"]}})
        self.run_pipeline("regression", llm=llm, budget=90)
        journal = (self.root / "agent_journal.jsonl").read_text(encoding="utf-8")
        self.assertIn('"title": "runtime error"', journal)
        self.assertIn('"verdict": "failed"', journal)
        for dropped in ("evil", "garbage types", "unknown kind"):
            self.assertNotIn(dropped, journal)

    def test_small_data_without_holdout(self):
        spec = build_spec(make_task(self.root, "binary", n_train=800, n_test=200))
        train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
        now = time.time()
        pred = run_v2(train, test, spec, now + 60, now + 70, log=lambda m: None, llm=FakeLLM({}))
        self.assertEqual(len(pred), len(test))

    def test_too_little_time_refuses(self):
        spec = build_spec(make_task(self.root, "binary"))
        train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
        now = time.time()
        with self.assertRaises(RuntimeError):
            run_v2(train, test, spec, now + 10, now + 20, log=lambda m: None, llm=FakeLLM({}))

    def test_binary_probabilities(self):
        _, pred = self.run_pipeline("binary", llm=FakeLLM({}))
        pred = np.asarray(pred, float)
        self.assertTrue(((pred >= 0) & (pred <= 1)).all())


if __name__ == "__main__":
    unittest.main()
