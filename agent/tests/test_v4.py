"""Entity history on temporal tasks and label-shift correction."""
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from automl.history import find_entity_key, history_features
from automl.shift import MAX_WEIGHT, em_prior_weights, total_variation


def temporal_frames(n_objects=800, periods=(1, 2, 3, 4, 5), test_period=6, seed=0):
    """Objects with static attributes (area, kind) observed every period; weather changes per period."""
    rng = np.random.default_rng(seed)
    area = rng.integers(1000, 200000, n_objects)
    kind = rng.choice(["office", "home", "shop"], n_objects)
    effect = rng.normal(0, 30, n_objects)
    rows = []
    for p in list(periods) + [test_period]:
        present = rng.random(n_objects) < 0.8
        weather = rng.normal(10 + p, 3)  # one value per period, the same for all objects
        for i in np.flatnonzero(present):
            rows.append({"period": p, "area": area[i], "kind": kind[i], "temp": round(weather, 2),
                         "noise_feature": rng.normal(),
                         "y": 100 + effect[i] + 2 * weather + rng.normal(0, 3)})
    df = pd.DataFrame(rows)
    df.insert(0, "id", np.arange(len(df)))
    train = df[df["period"] < test_period].reset_index(drop=True)
    test = df[df["period"] == test_period].drop(columns="y").reset_index(drop=True)
    answers = df[df["period"] == test_period]["y"].to_numpy()
    return train, test, answers


class HistoryTests(unittest.TestCase):
    def test_key_ignores_values_that_change_every_period(self):
        train, test, _ = temporal_frames()
        key, info = find_entity_key(train, test, "period", {"id", "y"}, log=lambda m: None)
        self.assertIsNotNone(key)
        self.assertIn("area", key)
        self.assertNotIn("temp", key)
        self.assertNotIn("noise_feature", key)
        self.assertGreater(info["coverage"], 0.7)

    def test_features_use_only_earlier_periods(self):
        train, test, answers = temporal_frames()
        feats = history_features(train, test, ["area", "kind"], "period", train["y"].to_numpy())
        tr = feats.iloc[:len(train)]
        self.assertEqual(int(tr[train["period"].eq(1).to_numpy()]["hist_last"].notna().sum()), 0)
        # A train row's hist_last is never its own target (strictly earlier periods only).
        has = tr["hist_last"].notna().to_numpy()
        self.assertFalse(np.any(np.isclose(tr["hist_last"].to_numpy()[has], train["y"].to_numpy()[has])))
        # For the test period the last known value is a strong predictor of the answer.
        te = feats.iloc[len(train):].reset_index(drop=True)
        known = te["hist_last"].notna().to_numpy()
        corr = np.corrcoef(te["hist_last"].to_numpy()[known], answers[known])[0, 1]
        self.assertGreater(corr, 0.8)
        self.assertTrue((te["hist_gap"].dropna() >= 1).all())

    def test_column_static_in_train_but_new_in_test_is_not_in_key(self):
        # Like days_above_38C on the public task: constant for an object over the train periods,
        # different in the test period. It splits colliding areas, but test objects stop matching.
        rng = np.random.default_rng(3)
        n = 800
        area = rng.integers(0, 300, n)  # collisions: several buildings share an area
        kind = rng.choice(["office", "home", "shop"], n)
        region = rng.choice(["north", "south", "east", "west"], n)
        tag_train, tag_test = rng.integers(0, 2, n), rng.integers(0, 2, n)
        rows = []
        for p in (1, 2, 3, 4, 5, 6):
            for i in np.flatnonzero(rng.random(n) < 0.8):
                rows.append({"period": p, "area": area[i], "kind": kind[i], "region": region[i],
                             "tag": tag_test[i] if p == 6 else tag_train[i], "y": float(i)})
        df = pd.DataFrame(rows)
        df.insert(0, "id", np.arange(len(df)))
        train = df[df["period"] < 6].reset_index(drop=True)
        test = df[df["period"] == 6].drop(columns="y").reset_index(drop=True)
        key, info = find_entity_key(train, test, "period", {"id", "y"}, log=lambda m: None)
        self.assertIsNotNone(key)
        self.assertNotIn("tag", key)
        self.assertGreater(info["coverage"], 0.6)

    def test_good_llm_key_is_chosen_and_logged(self):
        train, test, _ = temporal_frames()
        logs = []
        key, _ = find_entity_key(train, test, "period", {"id", "y"}, proposed=["kind", "area"], log=logs.append)
        self.assertEqual(sorted(key), ["area", "kind"])
        self.assertTrue(any("chosen: LLM" in m for m in logs), logs)

    def test_weak_llm_key_is_rejected(self):
        train, test, _ = temporal_frames()
        logs = []
        key, _ = find_entity_key(train, test, "period", {"id", "y"}, proposed=["kind"], log=logs.append)
        self.assertIn("area", key)
        self.assertTrue(any("chosen: rule" in m for m in logs), logs)

    def test_llm_key_with_train_only_column_does_not_crash(self):
        train, test, _ = temporal_frames()
        train["only_in_train"] = 1
        key, _ = find_entity_key(train, test, "period", {"id", "y"}, proposed=["only_in_train", "area"],
                                 log=lambda m: None)
        self.assertIn("area", key)

    def test_continuous_time_is_skipped(self):
        train, test, _ = temporal_frames(periods=range(1, 200), test_period=200, n_objects=50)
        logs = []
        key, info = find_entity_key(train, test, "period", {"id", "y"}, log=logs.append)
        self.assertIsNone(key)
        self.assertIn("periods", info["reason"])

    def test_deadline_stops_search(self):
        train, test, _ = temporal_frames()
        started = time.time()
        key, info = find_entity_key(train, test, "period", {"id", "y"}, log=lambda m: None,
                                    deadline=time.time() - 1)
        self.assertIsNone(key)
        self.assertIn("time", info["reason"])
        self.assertLess(time.time() - started, 5)

    def test_history_matches_naive_with_duplicates_and_missing_keys(self):
        train = pd.DataFrame({"k": ["a", "a", "a", "b", None, None, "a"],
                              "t": [1, 1, 2, 2, 1, 2, 3],
                              "y": [1.0, 3.0, 5.0, 7.0, 10.0, 20.0, 9.0]})
        test = pd.DataFrame({"k": ["a", "b", "c", None], "t": [4, 4, 4, 4]})
        feats = history_features(train, test, ["k"], "t", train["y"].to_numpy())
        # a: period means 1->2, 2->5, 3->9. Row "a"@3 sees periods 1,2; test "a"@4 sees 1,2,3.
        self.assertTrue(np.isnan(feats["hist_last"].iloc[0]))  # "a"@1: no earlier period
        self.assertAlmostEqual(feats["hist_last"].iloc[2], 2.0)  # "a"@2 sees the mean of period 1 (1, 3)
        self.assertAlmostEqual(feats["hist_mean"].iloc[6], 3.5)
        self.assertEqual(feats["hist_count"].iloc[6], 2)
        te = feats.iloc[len(train):].reset_index(drop=True)
        self.assertAlmostEqual(te["hist_last"][0], 9.0)
        self.assertAlmostEqual(te["hist_mean"][0], (2 + 5 + 9) / 3)
        self.assertEqual(te["hist_gap"][0], 1)
        self.assertAlmostEqual(te["hist_last"][1], 7.0)
        self.assertEqual(te["hist_gap"][1], 2)
        self.assertTrue(np.isnan(te["hist_last"][2]))  # unseen object
        self.assertAlmostEqual(te["hist_last"][3], 20.0)  # missing key values form one group

    def test_no_key_without_repeated_objects(self):
        rng = np.random.default_rng(1)
        train = pd.DataFrame({"id": range(2000), "period": np.repeat([1, 2, 3, 4], 500),
                              "x": rng.normal(size=2000), "y": rng.normal(size=2000)})
        test = pd.DataFrame({"id": range(2000, 2500), "period": 5, "x": rng.normal(size=500)})
        key, _ = find_entity_key(train, test, "period", {"id", "y"}, log=lambda m: None)
        self.assertIsNone(key)


class LabelShiftTests(unittest.TestCase):
    def test_em_recovers_shifted_priors(self):
        rng = np.random.default_rng(0)
        k, n = 3, 20000
        source = np.array([0.6, 0.3, 0.1])
        target = np.array([0.3, 0.3, 0.4])
        means = np.array([[0, 0], [2, 0], [0, 2]])

        def posterior(x, prior):
            like = np.exp(-0.5 * ((x[:, None, :] - means[None]) ** 2).sum(-1))
            p = like * prior
            return p / p.sum(1, keepdims=True)

        y_test = rng.choice(k, n, p=target)
        x_test = means[y_test] + rng.normal(size=(n, 2))
        proba = posterior(x_test, source)  # calibrated for the source distribution
        weights, estimate, _ = em_prior_weights(source, proba)
        self.assertLess(total_variation(estimate, target), 0.03)
        before = (proba.argmax(1) == y_test).mean()
        after = ((proba * weights).argmax(1) == y_test).mean()
        self.assertGreater(after, before)

    def test_no_shift_means_unit_weights(self):
        rng = np.random.default_rng(1)
        proba = rng.dirichlet([1, 1, 1], size=5000)
        weights, estimate, _ = em_prior_weights(proba.mean(0), proba)
        self.assertTrue(np.allclose(weights, 1, atol=0.05))

    def test_weights_are_clipped(self):
        proba = np.tile([0.2, 0.8], (1000, 1))
        weights, _, _ = em_prior_weights(np.array([0.999, 0.001]), proba)
        self.assertLessEqual(weights.max(), MAX_WEIGHT)
        self.assertGreaterEqual(weights.min(), 1 / MAX_WEIGHT)


class FakeBench:
    def __init__(self, oof_prior):
        self.oof_prior = np.asarray(oof_prior, float)
        self.problem = type("P", (), {"classes": ["a", "b"]})()
        self.logs = []
        self.log = self.logs.append


class PipelineHookTests(unittest.TestCase):
    def test_small_shift_keeps_class_weights(self):
        from automl.pipeline import _label_shift
        bench = FakeBench([0.5, 0.5])
        proba = np.tile([0.5, 0.5], (100, 1))
        tuned = np.array([1.0, 2.0])
        self.assertIs(_label_shift(bench, proba, tuned), tuned)
        self.assertTrue(any("not applied" in m for m in bench.logs))

    def test_large_shift_multiplies_class_weights(self):
        from automl.pipeline import _label_shift
        bench = FakeBench([0.5, 0.5])
        proba = np.tile([0.2, 0.8], (100, 1))
        weights, _, _ = em_prior_weights(bench.oof_prior, proba)
        np.testing.assert_allclose(_label_shift(bench, proba, None), weights)
        np.testing.assert_allclose(_label_shift(bench, proba, np.array([1.0, 2.0])), weights * [1.0, 2.0])

    def test_label_shift_is_off_unless_asked(self):
        # On the public UNSW task EM put DoS at 0% of the test (about 5% in fact): off by default.
        from unittest.mock import patch
        from automl.research.rd import Toggles
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_LABEL_SHIFT", None)
            self.assertFalse(Toggles.from_env().label_shift)
            self.assertTrue(Toggles.from_env().history)
        with patch.dict(os.environ, {"AGENT_LABEL_SHIFT": "1"}):
            self.assertTrue(Toggles.from_env().label_shift)

    def test_skip_reasons(self):
        from automl.pipeline import history_skip_reason, label_shift_skip_reason
        from automl.research.rd import Toggles
        on, off = Toggles(label_shift=True), Toggles(history=False, label_shift=False)
        temporal = type("P", (), {"scheme": "temporal", "classification": True})()
        random_cv = type("P", (), {"scheme": "random", "classification": True})()
        spec = type("S", (), {"task_type": "binary", "pred_format": "label"})()
        multi = type("S", (), {"task_type": "multiclass", "pred_format": "label"})()
        proba_spec = type("S", (), {"task_type": "binary", "pred_format": "proba"})()
        self.assertIsNone(history_skip_reason(on, temporal, spec))
        self.assertIn("AGENT_HISTORY", history_skip_reason(off, temporal, spec))
        self.assertIn("random", history_skip_reason(on, random_cv, spec))
        self.assertIn("multiclass", history_skip_reason(on, temporal, multi))
        self.assertIsNone(label_shift_skip_reason(on, temporal, spec, np.array([0.5, 0.5])))
        self.assertIn("AGENT_LABEL_SHIFT", label_shift_skip_reason(off, temporal, spec, np.array([0.5, 0.5])))
        self.assertIn("proba", label_shift_skip_reason(on, temporal, proba_spec, np.array([0.5, 0.5])))
        self.assertIn("prior", label_shift_skip_reason(on, temporal, spec, None))


class RefitIterationsTests(unittest.TestCase):
    def test_temporal_uses_the_latest_fold(self):
        from automl.pipeline import refit_iterations
        # Early folds train on little data and stop early; the latest fold is closest to the full refit.
        self.assertEqual(refit_iterations([12, 30, 140], "temporal"), 140)

    def test_random_uses_the_median(self):
        from automl.pipeline import refit_iterations
        self.assertEqual(refit_iterations([90, 100, 130, 80, 120], "random"), 100)
        self.assertIsNone(refit_iterations(None, "random"))


class PipelineV4Tests(unittest.TestCase):
    def test_losing_blend_falls_back_to_the_v2_base_model_not_v1(self):
        # The v2 base model has history features and temporal validation; v1 has neither.
        from unittest.mock import patch
        from automl import pipeline
        train, test, _ = temporal_frames(n_objects=600)
        with tempfile.TemporaryDirectory() as tmp:
            spec = self._write_task(Path(tmp), train, test)
            logs = []
            now = time.time()
            with patch.object(pipeline, "blend_and_check", return_value=(None, None)):
                pred = pipeline.run_v2(train, test, spec, now + 90, now + 100, log=logs.append,
                                       llm=__import__("tests.test_research", fromlist=["FakeLLM"]).FakeLLM({}))
        self.assertEqual(len(pred), len(test))
        self.assertTrue(any("base model" in m and "refit" in m for m in logs), logs[-5:])

    @staticmethod
    def _write_task(root: Path, train, test):
        from automl.task import build_spec
        train.to_csv(root / "train.csv", index=False)
        test.to_csv(root / "test.csv", index=False)
        pd.DataFrame({"id": test["id"], "y": 0.0}).to_csv(root / "sample_submission.csv", index=False)
        (root / "task.md").write_text("# t\n\nМетрика оценки — **RMSE**.\n\nФайл `submission.csv`.\n", encoding="utf-8")
        return build_spec(root / "task.md")

    def test_temporal_task_uses_history(self):
        from automl.pipeline import run_v2
        from automl.task import build_spec
        from tests.test_research import FakeLLM
        train, test, _ = temporal_frames(n_objects=1500)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            pd.DataFrame({"id": test["id"], "y": 0.0}).to_csv(root / "sample_submission.csv", index=False)
            pd.DataFrame({"feature": ["id", "period", "area", "kind", "temp", "noise_feature", "y"],
                          "role": ["id", "feature", "feature", "feature", "feature", "feature", "target"],
                          "description_ru": ["id", "Период наблюдения (год)", "Площадь", "Тип", "Температура года",
                                             "шум", "цель"]}).to_csv(root / "data_dictionary.csv", index=False)
            (root / "task.md").write_text("# t\n\nМетрика оценки — **RMSE**.\n\nФайл `submission.csv`.\n",
                                          encoding="utf-8")
            spec = build_spec(root / "task.md")
            logs = []
            now = time.time()
            pred = run_v2(train, test, spec, now + 120, now + 130, log=logs.append, llm=FakeLLM({}))
            self.assertEqual(len(pred), len(test))
            self.assertTrue(any("validation=temporal" in m for m in logs))
            history_logs = [m for m in logs if m.startswith("history: key")]
            self.assertTrue(history_logs, "history features must be added on a temporal task")
            self.assertIn("'area'", history_logs[0])


if __name__ == "__main__":
    unittest.main()


class SeedBaggingTests(unittest.TestCase):
    def test_refit_averages_distinct_seeds(self):
        from automl.pipeline import bag_seeds
        seen = []

        def fit(seed):
            seen.append(seed)
            return np.full(3, float(seed))

        pred, used = bag_seeds(fit, base_seed=42, n_seeds=4, soft_deadline=time.time() + 60)
        self.assertEqual(used, 4)
        self.assertEqual(len(set(seen)), 4)
        self.assertEqual(seen[0], 42)
        self.assertTrue(np.allclose(pred, np.mean(seen)))

    def test_refit_bagging_stops_at_the_deadline(self):
        from automl.pipeline import bag_seeds

        def slow_fit(seed):
            time.sleep(0.2)
            return np.ones(2)

        pred, used = bag_seeds(slow_fit, base_seed=1, n_seeds=10, soft_deadline=time.time() + 0.5)
        self.assertGreaterEqual(used, 1)
        self.assertLess(used, 10)
        self.assertTrue(np.allclose(pred, 1.0))


class ResidualOffsetTests(unittest.TestCase):
    def test_offset_models_predict_the_difference_and_add_it_back(self):
        # v7: regression candidates learn y - hist_last; predictions get hist_last back.
        from pathlib import Path
        from automl.cv import fit_full, make_problem, run_cv
        from automl.task import build_spec
        rng = np.random.default_rng(0)
        n = 2000
        last = rng.uniform(50, 150, n)
        noise = rng.normal(size=n)
        y = last + 2.0 * noise
        train = pd.DataFrame({"id": np.arange(n), "hist_last": last, "noise": noise, "y": y})
        test = pd.DataFrame({"id": np.arange(n, n + 100), "hist_last": np.full(100, 500.0), "noise": np.zeros(100)})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train.to_csv(root / "train.csv", index=False)
            test.to_csv(root / "test.csv", index=False)
            pd.DataFrame({"id": test["id"], "y": 0.0}).to_csv(root / "sample_submission.csv", index=False)
            (root / "task.md").write_text("# t\n\nМетрика оценки — **RMSE**.\n\nФайл `submission.csv`.\n",
                                          encoding="utf-8")
            spec = build_spec(root / "task.md")
            problem = make_problem(train, spec, holdout=False)
            cols = ["hist_last", "noise"]
            res = run_cv(problem, train[cols], test[cols], log=lambda m: None, offset="hist_last")
            self.assertLess(float(np.sqrt(np.nanmean((res.oof - y) ** 2))), 1.0)
            full = fit_full(problem, train[cols], test[cols], iterations=100, log=lambda m: None, offset="hist_last")
            # beyond the train range the plain model would cap at ~150; the offset model follows hist_last
            # until fit_full clips to the observed target range
            self.assertGreater(float(np.mean(full)), 140.0)
