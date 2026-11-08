import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from automl.metrics import CLASSIFICATION, REGRESSION, score
from automl.submission import write_submission
from automl.task import METRIC_PATTERNS, build_spec
from tests.helpers import make_task


class MetricRegistryTests(unittest.TestCase):
    def test_every_detectable_metric_is_scoreable(self):
        y_reg = np.array([1.0, 2.0, 3.0])
        y_cls = np.array([0, 1, 1])
        proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
        for name, _ in METRIC_PATTERNS:
            self.assertIn(name, REGRESSION | CLASSIFICATION)
            if name in REGRESSION:
                self.assertTrue(np.isfinite(score(name, y_reg, y_reg + 0.5)))
            else:
                self.assertTrue(np.isfinite(score(name, y_cls, proba.argmax(1), proba=proba)))

    def test_plain_f1_on_multiclass_does_not_crash(self):
        y = np.array([0, 1, 2, 2])
        self.assertAlmostEqual(score("f1", y, y), 1.0)

    def test_values(self):
        self.assertAlmostEqual(score("rmse", np.array([0.0, 0.0]), np.array([3.0, 4.0])),
                               np.sqrt(12.5))
        self.assertAlmostEqual(score("f1_macro", np.array([0, 1]), np.array([0, 0])), 1 / 3)


class WriteSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = build_spec(make_task(self.root, "multiclass"))
        self.test = pd.read_csv(self.spec.test_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rejects_bad_predictions_and_keeps_previous_file(self):
        write_submission(self.test, ["mid"] * len(self.test), self.spec)
        for bad in (["mid"] * (len(self.test) - 1), ["unknown"] * len(self.test),
                    [None] * len(self.test)):
            with self.assertRaises(ValueError):
                write_submission(self.test, bad, self.spec)
        sub = pd.read_csv(self.spec.output_path)
        self.assertEqual(len(sub), len(self.test))
        self.assertEqual(set(sub["kind"]), {"mid"})

    def test_rejects_duplicate_ids(self):
        test = self.test.copy()
        test.loc[test.index[1], "id"] = test["id"].iloc[0]
        with self.assertRaises(ValueError):
            write_submission(test, ["mid"] * len(test), self.spec)


if __name__ == "__main__":
    unittest.main()
