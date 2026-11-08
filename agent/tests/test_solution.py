import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import solution
from tests.helpers import make_task

AGENT_DIR = Path(__file__).resolve().parent.parent


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Short budget so v2 finishes quickly on tiny synthetic data; deadline tests patch again.
        self.patches = [patch.object(solution, "TIME_LIMIT_SEC", 90.0),
                        patch.object(solution, "SAFETY_MARGIN_SEC", 30.0),
                        patch.object(solution, "HARD_MARGIN_SEC", 10.0),
                        patch.dict(os.environ, {"OPENAI_API_KEY": ""})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_task(self, kind, target=None, **kwargs):
        task_path = make_task(self.root, kind, **kwargs)
        self.assertEqual(solution.main(["solution.py", str(task_path)]), 0)
        return self.read_submission(kind, target)

    def read_submission(self, kind, target=None):
        test = pd.read_csv(self.root / "data" / "participants" / "test.csv")
        sub = pd.read_csv(self.root / "submission.csv")
        self.assertEqual(list(sub["id"]), list(test["id"]), "ids must keep test.csv order")
        self.assertEqual(sub.shape[1], 2)
        self.assertFalse(sub.isna().any().any())
        return sub

    def test_regression_learns_signal_within_target_range(self):
        sub = self.run_task("regression")
        train = pd.read_csv(self.root / "data" / "participants" / "train.csv")
        self.assertTrue(np.isfinite(sub["price"]).all())
        self.assertGreater(sub["price"].std(), 1.0, "model must not predict a constant")
        self.assertTrue(sub["price"].between(train["price"].min(), train["price"].max()).all())

    def test_multiclass_outputs_known_labels(self):
        sub = self.run_task("multiclass")
        self.assertTrue(set(sub["kind"]) <= {"high", "low", "mid", "rare"})
        self.assertGreaterEqual(sub["kind"].nunique(), 3)

    def test_binary_auc_outputs_probabilities(self):
        sub = self.run_task("binary")
        self.assertTrue(sub["is_bad"].between(0, 1).all())
        self.assertGreater(sub["is_bad"].nunique(), 10)

    def test_class_with_single_row_still_trains_model(self):
        sub = self.run_task("multiclass", rare_rows=1)
        self.assertGreaterEqual(sub["kind"].nunique(), 3, "must not fall back to a constant")

    def test_json_special_characters_in_column_names(self):
        sub = self.run_task("regression", odd_column=True)
        self.assertGreater(sub["price"].std(), 1.0, "must not fall back to a constant")

    def test_expired_deadline_still_trains_one_fold(self):
        # Whole budget already spent: one truncated fold, but a model answer, not the constant.
        with patch.object(solution, "TIME_LIMIT_SEC", 1.0), \
                patch.object(solution, "SAFETY_MARGIN_SEC", 1.0), \
                patch.object(solution, "HARD_MARGIN_SEC", 1.0):
            sub = self.run_task("regression")
        self.assertGreater(sub["price"].std(), 0.0)

    def test_soft_deadline_keeps_full_boosting_in_first_fold(self):
        with patch.object(solution, "SAFETY_MARGIN_SEC", solution.TIME_LIMIT_SEC):
            sub = self.run_task("regression")
        self.assertGreater(sub["price"].std(), 1.0)

    def test_answer_column_name_follows_sample(self):
        task_path = make_task(self.root, "regression", answer_col="prediction")
        solution.main(["solution.py", str(task_path)])
        self.assertEqual(list(pd.read_csv(self.root / "submission.csv").columns), ["id", "prediction"])

    def test_model_crash_still_leaves_valid_fallback(self):
        with patch("solution.train_and_predict", side_effect=RuntimeError("boom")), \
                patch.dict(os.environ, {"AGENT_MODE": "v1"}):
            sub = self.run_task("multiclass")
        self.assertEqual(sub["kind"].nunique(), 1)  # majority class

    def test_v2_crash_keeps_v1_answer(self):
        with patch("automl.pipeline.run_v2", side_effect=RuntimeError("boom")):
            sub = self.run_task("multiclass")
        self.assertGreaterEqual(sub["kind"].nunique(), 3)  # v1 model answer, not the constant

    def test_broken_v2_module_keeps_v1_answer(self):
        with patch.dict(sys.modules, {"automl.pipeline": None}):  # import raises ImportError
            sub = self.run_task("multiclass")
        self.assertGreaterEqual(sub["kind"].nunique(), 3)

    def test_v2_answer_replaces_v1(self):
        with patch("automl.pipeline.run_v2", side_effect=lambda train, test, spec, *a, **k:
                   np.full(len(test), "low", dtype=object)):
            sub = self.run_task("multiclass")
        self.assertEqual(set(sub["kind"]), {"low"})

    def test_broken_task_copies_sample_as_last_resort(self):
        task_path = make_task(self.root, "regression")
        (self.root / "data" / "participants" / "train.csv").unlink()
        self.assertEqual(solution.main(["solution.py", str(task_path)]), 0)
        self.assertTrue((self.root / "submission.csv").is_file())


class PackagingTests(unittest.TestCase):
    def test_zip_runs_like_on_platform(self):
        """Exactly the platform rules: only root *.py and requirements.txt are copied into the
        task folder, then `python solution.py task.md` runs there."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = tmp / "solution.zip"
            subprocess.run([sys.executable, str(AGENT_DIR / "scripts" / "make_zip.py"), str(archive)],
                           check=True, capture_output=True)
            task_dir = tmp / "task"
            make_task(task_dir, "binary")
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
                for name in names:
                    if "/" not in name and (name.endswith(".py") or name == "requirements.txt"):
                        (task_dir / name).write_bytes(z.read(name))
            self.assertIn("solution.py", names)
            self.assertIn("requirements.txt", names)
            self.assertFalse([n for n in names if "/" in n], "no folders allowed in the archive")
            env = {**os.environ, "AGENT_TIME_LIMIT_SEC": "90", "OPENAI_API_KEY": ""}
            run = subprocess.run([sys.executable, "solution.py", "task.md"], cwd=task_dir,
                                 check=True, capture_output=True, timeout=300, env=env, text=True,
                                 encoding="utf-8", errors="replace")
            self.assertTrue((task_dir / "submission.csv").is_file())
            self.assertIn("v2 submission written", run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
