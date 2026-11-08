"""Tests for the RD-Agent-style additions: screening sample, developer role, component choice."""
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from automl.cv import make_problem, make_subproblem
from automl.features import apply_op, validate_op
from automl.guard import GuardError, run_block_code
from automl.pipeline import run_v2
from automl.research.rd import choose_component, developer, extract_code
from automl.task import build_spec
from tests.helpers import make_task
from tests.test_research import FakeLLM

FRAME = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 0, 1, 4], "c": pd.Categorical(["x", "y", "x", "x"])})
GOOD_BLOCK = ("def features(df):\n    out = pd.DataFrame(index=df.index)\n"
              "    out['a_per_b'] = df['a'] / df['b'].replace(0, np.nan)\n"
              "    out['a_sq'] = df['a'] ** 2\n    return out")


class ScriptedLLM(FakeLLM):
    """FakeLLM whose roles can answer with a list of successive replies; supports ask_text."""

    def _next(self, role, user):
        self.calls.append(role)
        answer = self.answers.get(role)
        if isinstance(answer, list) and answer and isinstance(answer[0], str):
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return answer(user) if callable(answer) else answer

    def ask_json(self, role, system, user):
        return self._next(role, user)

    def ask_text(self, role, system, user):
        return self._next(role, user)


class BlockCodeTests(unittest.TestCase):
    def test_block_adds_prefixed_columns(self):
        out = apply_op(FRAME, {"op": "code_block", "name": "ratios", "code": GOOD_BLOCK})
        self.assertIn("block_ratios__a_per_b", out.columns)
        self.assertIn("block_ratios__a_sq", out.columns)
        self.assertTrue(np.isnan(out["block_ratios__a_per_b"][1]))

    def test_block_rules(self):
        with self.assertRaises(GuardError):
            run_block_code("def features(df):\n    return df['a']" + "\n" + "import os", FRAME)
        with self.assertRaises(GuardError):  # too many columns
            run_block_code("def features(df):\n    return pd.DataFrame({str(i): df['a'] + i "
                           "for i in range(25)})", FRAME)
        self.assertIsNotNone(validate_op({"op": "code_block", "name": "x", "code": "def feature(df): pass"},
                                         {"a"}))  # wrong function name


class SubproblemTests(unittest.TestCase):
    def test_stratified_sample_keeps_rare_class_and_has_three_folds(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_spec(make_task(Path(tmp), "multiclass", n_train=4000, rare_rows=10))
            problem = make_problem(pd.read_csv(spec.train_path), spec)
            sub = make_subproblem(problem, 1000)
            self.assertLess(len(sub.research_idx), 1200)
            self.assertTrue(set(sub.research_idx) <= set(problem.research_idx))
            self.assertEqual(len(sub.folds), 3)
            self.assertEqual(len(sub.holdout_idx), 0)
            rare = problem.classes.index("rare")
            self.assertEqual((problem.y[sub.research_idx] == rare).sum(), (problem.y[problem.research_idx] == rare).sum())

    def test_regression_and_oversized_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = build_spec(make_task(Path(tmp), "regression", n_train=4000))
            problem = make_problem(pd.read_csv(spec.train_path), spec)
            sub = make_subproblem(problem, 1000)
            self.assertEqual(len(sub.research_idx), 1000)
            self.assertEqual(len(make_subproblem(problem, 10 ** 6).research_idx), len(problem.research_idx))


class ComponentChoiceTests(unittest.TestCase):
    def test_untried_first_then_ucb(self):
        comps = ("features", "develop", "model")
        self.assertEqual(choose_component([], comps), "features")
        entries = [{"kind": "features", "source": "llm", "verdict": "rejected"},
                   {"kind": "features", "source": "llm-dev", "verdict": "kept"}]
        self.assertEqual(choose_component(entries, comps), "model")  # never tried
        entries += [{"kind": "params", "source": "llm", "verdict": "rejected"}] * 3
        self.assertEqual(choose_component(entries, comps), "develop")  # best success rate

    def test_failed_developer_attempts_count_as_tries(self):
        comps = ("features", "develop", "model")
        entries = [{"kind": "features", "source": "llm", "verdict": "rejected"},
                   {"kind": "develop", "source": "llm-dev", "verdict": "failed"}]
        self.assertEqual(choose_component(entries, comps), "model")  # develop was tried, model was not

    def test_toggles_parse_common_spellings(self):
        from unittest.mock import patch
        import os
        from automl.research.rd import Toggles
        with patch.dict(os.environ, {"AGENT_SCREEN": "off", "AGENT_DEVELOPER": "False", "AGENT_COMPONENTS": "1"}):
            self.assertEqual(Toggles.from_env(), Toggles(screen=False, developer=False, components=True))


class DeveloperTests(unittest.TestCase):
    def test_debug_loop_fixes_code_from_error(self):
        broken = "```python\ndef features(df):\n    return pd.DataFrame({'x': df['missing']})\n```"
        fixed = f"```python\n{GOOD_BLOCK}\n```"
        llm = ScriptedLLM({"developer": [broken, fixed]})
        logs = []
        op = developer(llm, {"title": "Ratios", "hypothesis": "ratio of a and b"}, FRAME, {"a", "b"}, "y",
                       log=logs.append)
        self.assertEqual(op["op"], "code_block")
        self.assertEqual(llm.calls, ["developer", "developer"])
        self.assertTrue(any("attempt 1 failed" in m for m in logs))

    def test_gives_up_after_three_attempts(self):
        llm = ScriptedLLM({"developer": ["no code here"]})
        self.assertIsNone(developer(llm, {"title": "x", "hypothesis": "something long"}, FRAME, {"a"}, "y",
                                    log=lambda m: None))
        self.assertEqual(len(llm.calls), 3)

    def test_extract_code(self):
        self.assertEqual(extract_code("<think>x</think>```python\ndef features(df):\n    return df\n```"),
                         "def features(df):\n    return df")


class PipelineRDTests(unittest.TestCase):
    def test_develop_card_goes_through_developer_and_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = build_spec(make_task(root, "regression", n_train=4000, n_test=500))
            train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
            code = ("```python\ndef features(df):\n    out = pd.DataFrame(index=df.index)\n"
                    "    out['x1_x2'] = df['x1'] * df['x2']\n    return out\n```")
            llm = ScriptedLLM({"analyst": {"notes": "synthetic"},
                               "researcher": [{"title": "interaction block", "kind": "develop",
                                               "hypothesis": "product of x1 and x2 captures interaction"}],
                               "critic": {"selected": [0]}, "developer": [code]})
            now = time.time()
            run_v2(train, test, spec, now + 120, now + 130, log=lambda m: None, llm=llm)
            entries = [json.loads(line) for line in
                       (root / "agent_journal.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("developer", llm.calls)
            developed = [e for e in entries if e["source"] == "llm-dev"]
            self.assertTrue(developed, "developer idea must reach the journal")
            first = developed[0]
            self.assertEqual(first["kind"], "features")  # converted into a real, evaluated idea
            self.assertEqual(first["ops"][0]["op"], "code_block")
            self.assertIn(first["verdict"], {"kept", "rejected", "screened"})
            self.assertEqual(llm.calls.count("developer"), 1, "the same hypothesis is developed once")

    def test_developer_off_means_no_developer_calls(self):
        from unittest.mock import patch
        import os
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"AGENT_DEVELOPER": "0"}):
            root = Path(tmp)
            spec = build_spec(make_task(root, "regression", n_train=4000, n_test=500))
            train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
            prompts = []
            llm = ScriptedLLM({"analyst": {"notes": "x"},
                               "researcher": lambda user: prompts.append(user) or
                               [{"title": "dev", "kind": "develop", "hypothesis": "product of x1 and x2 block"}],
                               "critic": {"selected": [0]}, "developer": ["```python\nbad\n```"]})
            now = time.time()
            run_v2(train, test, spec, now + 90, now + 100, log=lambda m: None, llm=llm)
            self.assertNotIn("developer", llm.calls)
            self.assertTrue(prompts and all('"develop"' not in p and "code_block" not in p for p in prompts))

    def test_screening_on_and_off_on_large_data(self):
        from unittest.mock import patch
        import os
        for flag in ("1", "0"):
            with self.subTest(AGENT_SCREEN=flag), tempfile.TemporaryDirectory() as tmp, \
                    patch.dict(os.environ, {"AGENT_SCREEN": flag, "AGENT_SCREEN_MIN_SEC": "0"}):
                root = Path(tmp)
                spec = build_spec(make_task(root, "binary", n_train=20000, n_test=500))
                train, test = pd.read_csv(spec.train_path), pd.read_csv(spec.test_path)
                logs = []
                now = time.time()
                run_v2(train, test, spec, now + 150, now + 160, log=logs.append, llm=FakeLLM({}))
                screening_on = any(m.startswith("screening:") and "rows x" in m for m in logs)
                self.assertEqual(screening_on, flag == "1")
                entries = [json.loads(line) for line in
                           (root / "agent_journal.jsonl").read_text(encoding="utf-8").splitlines()]
                has_screen_data = any(e["verdict"] == "screened" or "screen_gain" in e for e in entries)
                self.assertEqual(has_screen_data, flag == "1")
                self.assertTrue(any(m.startswith("research summary:") for m in logs))


if __name__ == "__main__":
    unittest.main()
