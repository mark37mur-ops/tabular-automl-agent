"""Validation policy and full-data refit contracts; no expensive model training."""
import sys
import time
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from automl.cv import (_iteration_count, fit_full, make_problem, run_cv)
from automl.task import TaskSpec


def regression_spec(dictionary=None):
    return TaskSpec(task_dir=Path("."), train_path=Path("train.csv"), test_path=Path("test.csv"),
                    sample_path=Path("sample.csv"), dict_path=dictionary, output_path=Path("answer.csv"),
                    id_col="id", target="target", answer_col="target", task_type="regression",
                    metric="rmse", pred_format="value")


def future_data(column="observation_period", periods=5, rows_per_period=800):
    values = np.repeat(np.arange(1, periods + 1), rows_per_period)
    train = pd.DataFrame({"id": np.arange(len(values)), column: values,
                          "signal": values * 1.0, "target": values * 2.0})
    # Exercise positional indices rather than relying on a sorted/RangeIndex source.
    train = train.sample(frac=1, random_state=13)
    test = pd.DataFrame({"id": np.arange(100), column: periods + 1, "signal": 1.0})
    return train, test


class TemporalPolicyTests(unittest.TestCase):
    def test_future_period_holds_out_latest_and_expands_strictly_into_future(self):
        train, test = future_data("Year_Factor")
        problem = make_problem(train, regression_spec(), test=test)
        self.assertEqual(problem.scheme, "temporal")
        self.assertEqual(problem.time_column, "Year_Factor")
        periods = train.Year_Factor.to_numpy()
        np.testing.assert_array_equal(np.unique(periods[problem.holdout_idx]), [5])
        np.testing.assert_array_equal(np.unique(periods[problem.research_idx]), [1, 2, 3, 4])
        self.assertEqual(len(problem.folds), 3)
        observed_mask = np.zeros(len(problem.research_idx), dtype=bool)
        for tr, va in problem.folds:
            past = periods[problem.research_idx[tr]]
            future = periods[problem.research_idx[va]]
            self.assertLess(past.max(), future.min())
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            self.assertEqual(len(np.intersect1d(problem.research_idx[tr], problem.holdout_idx)), 0)
            observed_mask[va] = True
        np.testing.assert_array_equal(problem.oof_mask, observed_mask)
        self.assertEqual(problem.metadata["oof_coverage"], 0.75)
        self.assertEqual(problem.metadata["holdout_period"], 5)

    def test_other_semantic_name_and_dictionary_hint_work(self):
        train, test = future_data("FiscalYear")
        self.assertEqual(make_problem(train, regression_spec(), test=test).scheme, "temporal")
        train, test = future_data("collection_code")
        dictionary = pd.DataFrame({"feature": ["collection_code"], "role": ["feature"],
                                   "description_ru": ["Период наблюдения, упорядоченный по времени"]})
        with patch("automl.cv.pd.read_csv", return_value=dictionary):
            problem = make_problem(train, regression_spec(Path("dictionary.csv")), test=test)
        self.assertEqual(problem.time_column, "collection_code")

    def test_requires_explicit_future_test_and_does_not_guess_from_row_ids(self):
        train, test = future_data()
        for future in (None, test.assign(observation_period=3)):
            problem = make_problem(train, regression_spec(), test=future)
            self.assertEqual(problem.scheme, "random")
        train, test = future_data("record_number")
        self.assertEqual(make_problem(train, regression_spec(), test=test).scheme, "random")

    def test_small_ambiguous_or_incomplete_time_features_keep_random_cv(self):
        cases = [future_data(rows_per_period=30), future_data(periods=3, rows_per_period=1200),
                 future_data("year_built")]
        for train, test in cases:
            self.assertEqual(make_problem(train, regression_spec(), test=test).scheme, "random")
        train, test = future_data()
        train.iloc[0, train.columns.get_loc("observation_period")] = np.nan
        self.assertEqual(make_problem(train, regression_spec(), test=test).scheme, "random")

    def test_holdout_false_preserves_all_rows_but_cv_warmup_is_not_full_fit(self):
        train, test = future_data()
        problem = make_problem(train, regression_spec(), holdout=False, test=test)
        self.assertEqual(problem.scheme, "temporal")
        self.assertEqual(len(problem.holdout_idx), 0)
        self.assertEqual(len(problem.research_idx), len(train))
        self.assertLess(problem.oof_mask.sum(), len(train))
        for tr, va in problem.folds:
            times = train.observation_period.to_numpy()[problem.research_idx]
            self.assertLess(times[tr].max(), times[va].min())

    def test_random_split_remains_reproducible_and_fixed(self):
        train, _ = future_data()
        first = make_problem(train, regression_spec())
        second = make_problem(train, regression_spec())
        self.assertEqual(first.scheme, "random")
        np.testing.assert_array_equal(first.research_idx, second.research_idx)
        np.testing.assert_array_equal(first.holdout_idx, second.holdout_idx)
        self.assertTrue(first.oof_mask.all())
        for a, b in zip(first.folds, second.folds):
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])


class IterationAndRefitTests(unittest.TestCase):
    def test_family_iteration_indices_become_tree_counts(self):
        self.assertEqual(_iteration_count(types.SimpleNamespace(best_iteration_=17), "lgbm"), 17)
        self.assertEqual(_iteration_count(types.SimpleNamespace(get_best_iteration=lambda: 16), "catboost"), 17)
        self.assertEqual(_iteration_count(types.SimpleNamespace(best_iteration=16), "xgb"), 17)
        self.assertEqual(_iteration_count(types.SimpleNamespace(get_best_iteration=lambda: -1, tree_count_=8),
                                         "catboost"), 8)

    def test_cv_records_iterations_and_scores_only_fixed_temporal_validation_rows(self):
        train, test = future_data()
        problem = make_problem(train, regression_spec(), test=test)
        X, Xt = train[["signal"]], test[["signal"]]
        model = types.SimpleNamespace(predict=lambda frame: frame.iloc[:, 0].to_numpy() * 2,
                                      best_iteration_=7)
        with patch.dict("automl.cv.FITTERS", {"lgbm": lambda *args: model}):
            result = run_cv(problem, X, Xt, log=lambda message: None)
        np.testing.assert_array_equal(np.isfinite(result.oof), problem.oof_mask)
        self.assertEqual(result.iterations, [7, 7, 7])
        self.assertEqual(result.score, 0.0)

    def test_full_fit_uses_all_rows_without_eval_set_or_early_stopping(self):
        train, test = future_data()
        problem = make_problem(train, regression_spec(), test=test)
        captured = {}

        class FakeRegressor:
            def __init__(self, **kwargs):
                captured["params"] = kwargs

            def fit(self, frame, y, **kwargs):
                captured["rows"] = len(frame)
                captured["y"] = y.copy()
                captured["fit_kwargs"] = kwargs
                return self

            def predict(self, frame):
                return np.full(len(frame), 5.0)

        fake = types.SimpleNamespace(LGBMRegressor=FakeRegressor)
        with patch.dict(sys.modules, {"lightgbm": fake}):
            pred = fit_full(problem, train[["signal"]], test[["signal"]], iterations=11,
                            params={"early_stopping_rounds": 10}, log=lambda message: None)
        self.assertEqual(captured["rows"], len(train))
        self.assertEqual(captured["params"]["n_estimators"], 11)
        self.assertNotIn("early_stopping_rounds", captured["params"])
        self.assertNotIn("eval_set", captured["fit_kwargs"])
        np.testing.assert_array_equal(captured["y"], problem.y)
        np.testing.assert_array_equal(pred, np.full(len(test), 5.0))

    def test_expired_full_fit_fails_before_backend_and_preserves_caller_fallback(self):
        train, test = future_data()
        problem = make_problem(train, regression_spec(), test=test)
        with self.assertRaises(TimeoutError):
            fit_full(problem, train[["signal"]], test[["signal"]], deadline=time.time() - 1)

    def test_full_fit_other_families_use_fixed_counts_without_eval_data(self):
        train, test = future_data()
        problem = make_problem(train, regression_spec(), test=test)
        for kind, module, count_key in (("catboost", "catboost", "iterations"),
                                         ("xgb", "xgboost", "n_estimators")):
            captured = {}

            class FakeRegressor:
                def __init__(self, **kwargs):
                    captured["params"] = kwargs

                def fit(self, frame, y, **kwargs):
                    captured["rows"] = len(frame)
                    captured["fit_kwargs"] = kwargs

                def predict(self, frame):
                    return np.full(len(frame), 5.0)

            fake = (types.SimpleNamespace(CatBoostRegressor=FakeRegressor, CatBoostClassifier=FakeRegressor)
                    if kind == "catboost" else types.SimpleNamespace(
                        XGBRegressor=FakeRegressor, callback=types.SimpleNamespace(TrainingCallback=object)))
            with patch.dict(sys.modules, {module: fake}):
                pred = fit_full(problem, train[["signal"]], test[["signal"]], kind=kind,
                                iterations=9, params={"early_stopping_rounds": 100}, log=lambda message: None)
            self.assertEqual(captured["rows"], len(train))
            self.assertEqual(captured["params"][count_key], 9)
            self.assertNotIn("early_stopping_rounds", captured["params"])
            self.assertNotIn("eval_set", captured["fit_kwargs"])
            self.assertEqual(len(pred), len(test))


class ClassWeightPolicyTests(unittest.TestCase):
    def test_sqrt_and_existing_weight_modes_in_all_backends_and_full_refit(self):
        y = np.array([0] * 90 + [1] * 10)
        train = pd.DataFrame({"id": np.arange(len(y)), "signal": np.arange(len(y)), "target": y})
        test = pd.DataFrame({"signal": np.arange(5)})
        spec = replace(regression_spec(), task_type="binary", metric="f1_macro", pred_format="label",
                       classes=[0, 1], positive_class=1)
        problem = make_problem(train, spec, holdout=False)
        for kind, module in (("lgbm", "lightgbm"), ("catboost", "catboost"), ("xgb", "xgboost")):
            for mode in (None, False, True, "sqrt"):
                captured = []

                class FakeClassifier:
                    classes_ = np.array([0, 1])

                    def __init__(self, **kwargs):
                        self.params = kwargs

                    def fit(self, frame, target, **kwargs):
                        captured.append((self.params, np.asarray(target), kwargs))
                        return self

                    def predict_proba(self, frame):
                        return np.tile([0.9, 0.1], (len(frame), 1))

                if kind == "lgbm":
                    fake = types.SimpleNamespace(LGBMClassifier=FakeClassifier,
                                                 early_stopping=lambda *args, **kwargs: object())
                elif kind == "catboost":
                    fake = types.SimpleNamespace(CatBoostClassifier=FakeClassifier, CatBoostRegressor=FakeClassifier)
                else:
                    fake = types.SimpleNamespace(XGBClassifier=FakeClassifier,
                                                 callback=types.SimpleNamespace(TrainingCallback=object))
                with patch.dict(sys.modules, {module: fake}):
                    run_cv(problem, train[["signal"]], test, kind=kind, balanced=mode,
                           log=lambda message: None)
                    fit_full(problem, train[["signal"]], test, kind=kind, balanced=mode, iterations=3,
                             log=lambda message: None)
                self.assertEqual(len(captured), len(problem.folds) + 1)
                effective = True if mode is None else mode  # f1_macro keeps its existing default
                for params, target, kwargs in captured:
                    if kind == "catboost":
                        self.assertEqual(params["auto_class_weights"], "SqrtBalanced" if effective == "sqrt"
                                         else ("Balanced" if effective else None))
                        continue
                    if kind == "lgbm" and effective != "sqrt":
                        self.assertEqual(params["class_weight"], "balanced" if effective else None)
                        continue
                    if kind == "xgb" and not effective:
                        self.assertIsNone(kwargs["sample_weight"])
                        continue
                    counts = np.bincount(target, minlength=2)
                    inverse = len(target) / (2 * counts)
                    expected = np.sqrt(inverse) if effective == "sqrt" else inverse
                    if effective == "sqrt":
                        expected /= np.mean(expected[target])
                    if kind == "lgbm":
                        np.testing.assert_allclose([params["class_weight"][i] for i in range(2)], expected)
                        self.assertGreater(params["class_weight"][1], params["class_weight"][0])
                        self.assertLess(params["class_weight"][1] / params["class_weight"][0],
                                        inverse[1] / inverse[0])
                    else:
                        np.testing.assert_allclose(kwargs["sample_weight"], expected[target])
                        self.assertAlmostEqual(float(np.mean(kwargs["sample_weight"])), 1.)
                        self.assertGreater(kwargs["sample_weight"][target == 1][0],
                                           kwargs["sample_weight"][target == 0][0])


if __name__ == "__main__":
    unittest.main()
