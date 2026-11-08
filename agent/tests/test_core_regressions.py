"""Regression checks for failed-fold isolation and honest all-row final refits."""
import time
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from automl import model, pipeline
from automl.cv import CVResult, Problem
from automl.metrics import score
import solution
from tests.helpers import make_task


class CoreRegressionTests(unittest.TestCase):
    def test_v1_training_budget_reserves_time_for_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_path = make_task(Path(tmp), "regression")
            seen = {}

            def fake_train(train, test, spec, deadline, **kwargs):
                seen.update(soft=deadline, hard=kwargs["hard_deadline"])
                return {"pred": np.full(len(test), train[spec.target].mean())}

            before = time.time()
            with patch.object(solution, "TIME_LIMIT_SEC", 1000.), \
                    patch.object(solution, "V1_BUDGET_SEC", 12.), \
                    patch.object(solution, "train_and_predict", side_effect=fake_train), \
                    patch.dict("os.environ", {"AGENT_MODE": "v1"}):
                self.assertEqual(solution.main(["solution.py", str(task_path)]), 0)
            self.assertLessEqual(seen["soft"], before+15)
            self.assertLessEqual(seen["hard"], before+15)

    def test_failed_test_prediction_does_not_commit_validation_oof(self):
        train = pd.DataFrame({"id": [0, 1, 2, 3], "x": [0., 1., 2., 3.], "y": [0., 1., 2., 3.]})
        test = pd.DataFrame({"id": [4, 5, 6], "x": [4., 5., 6.]})
        spec = SimpleNamespace(id_col="id", target="y", task_type="regression", metric="rmse")
        instances = []

        class FakeModel:
            best_iteration_ = 1

            def __init__(self):
                self.number = len(instances)
                instances.append(self)

            def fit(self, *args, **kwargs):
                return self

            def predict(self, frame):
                if self.number == 0 and len(frame) == 3:
                    raise RuntimeError("test prediction failed")
                return np.ones(len(frame))

        folds = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
        with patch.object(model, "_folds", return_value=folds), \
                patch.object(model, "_model", side_effect=lambda *args: FakeModel()):
            result = model.train_and_predict(train, test, spec, time.time()+100, log=lambda *_: None)
        self.assertEqual(result["folds"], 1)
        self.assertTrue(np.isnan(result["oof"][:2]).all())
        self.assertTrue(np.isfinite(result["oof"][2:]).all())

    def test_micro_f1_retains_its_averaging_definition(self):
        y = np.array([0, 0, 0, 1])
        pred = np.array([0, 0, 0, 0])
        self.assertEqual(score("f1_micro", y, pred), .75)
        self.assertAlmostEqual(score("f1_macro", y, pred), 3/7)

    def test_complete_candidate_requires_identical_oof_rows(self):
        problem = Problem(SimpleNamespace(), np.zeros(4), np.zeros(4), None,
                          np.arange(4), np.array([], dtype=int),
                          folds=[(np.array([0, 1]), np.array([2, 3]))],
                          oof_mask=np.array([False, False, True, True]))
        good = CVResult("good", np.array([np.nan, np.nan, 1., 1.]), np.ones(2), None, 1., .1, 1)
        wrong = CVResult("wrong", np.array([1., 1., np.nan, np.nan]), np.ones(2), None, 1., .1, 1)
        self.assertTrue(pipeline._complete_result(good, problem))
        self.assertFalse(pipeline._complete_result(wrong, problem))

    def test_refit_uses_all_rows_and_keeps_previous_prediction_on_failure(self):
        spec = SimpleNamespace(task_type="regression", metric="rmse", target="y")
        train = pd.DataFrame({"x": np.arange(10), "y": np.arange(10, dtype=float)})
        test = pd.DataFrame({"x": [20, 21]})
        problem = Problem(spec, train.y.to_numpy(), train.y.to_numpy(), None,
                          np.arange(8), np.array([8, 9]),
                          folds=[(np.arange(4), np.arange(4, 8))])
        bench = SimpleNamespace(train=train, spec=spec, problem=problem, log=lambda *_: None,
                                frames=lambda state: (None, train[["x"]], test))
        result = CVResult("base", np.ones(8), np.array([2., 3.]), np.ones(2), 1., .01, 1,
                          iterations=[5, 7, 9])
        candidate = pipeline.Candidate(result, "lgbm", {}, pipeline.State())
        final = ([candidate], np.array([1.]))

        def full_fit(p, x, xt, kind, **kwargs):
            self.assertEqual(len(x), 10)
            self.assertEqual(len(p.y), 10)
            self.assertEqual(kwargs["iterations"], 7)
            return np.array([4., 5.])

        with patch.object(pipeline, "fit_full", side_effect=full_fit):
            actual = pipeline.refit(bench, final, time.time()+100, time.time()+120)
        np.testing.assert_array_equal(actual, [4., 5.])
        with patch.object(pipeline, "fit_full", side_effect=TimeoutError("budget")):
            actual = pipeline.refit(bench, final, time.time()+100, time.time()+120)
        np.testing.assert_array_equal(actual, [2., 3.])


if __name__ == "__main__":
    unittest.main()
