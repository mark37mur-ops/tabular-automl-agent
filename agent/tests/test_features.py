import unittest

import numpy as np
import pandas as pd

from automl.features import apply_op, apply_recipe, base_frame, categorical_hints, normalize_op, validate_op
from automl.guard import GuardError, check_code, run_feature_code
from automl.task import TaskSpec


def spec(**kw):
    base = dict(task_dir=None, train_path=None, test_path=None, sample_path=None, dict_path=None,
                output_path=None, id_col="id", target="y", answer_col="y", task_type="regression",
                metric="rmse", pred_format="value")
    base.update(kw)
    return TaskSpec(**base)


FRAME = pd.DataFrame({"a": [1.0, 2, 3, 4], "b": [2.0, 0, 1, 4], "c": pd.Categorical(["x", "y", "x", "x"])})


class OpsTests(unittest.TestCase):
    def test_arithmetic_and_group_ops(self):
        out = apply_op(FRAME, {"op": "ratio", "a": "a", "b": "b"})
        self.assertTrue(np.isnan(out["ratio__a__b"][1]))  # division by zero -> NaN, not inf
        out = apply_op(FRAME, {"op": "group_mean", "by": ["c"], "col": "a"})
        self.assertEqual(list(out["group_mean__a__c"]), [8 / 3, 2.0, 8 / 3, 8 / 3])
        out = apply_op(FRAME, {"op": "group_count", "by": ["c"]})
        self.assertEqual(list(out["group_count__c"]), [3, 1, 3, 3])
        out = apply_op(FRAME, {"op": "freq", "col": "c"})
        self.assertEqual(list(out["freq__c"]), [3, 1, 3, 3])

    def test_bad_op_is_skipped_in_recipe(self):
        logs = []
        out = apply_recipe(FRAME, [{"op": "ratio", "a": "a", "b": "missing"},
                                   {"op": "diff", "a": "a", "b": "b"}], log=logs.append)
        self.assertIn("diff__a__b", out.columns)
        self.assertEqual(len(logs), 1)

    def test_validate_op(self):
        cols = {"a", "b", "c"}
        self.assertIsNone(validate_op(normalize_op({"op": "group_dev", "by": "c", "col": "a", "extra": 1}), cols))
        self.assertEqual(normalize_op({"op": "group_dev", "by": "c", "col": "a", "extra": 1}),
                         {"op": "group_dev", "by": ["c"], "col": "a"})
        self.assertIsNotNone(validate_op({"op": "ratio", "a": "a", "b": "zzz"}, cols))
        self.assertIsNotNone(validate_op({"op": "explode"}, cols))
        # categorical inputs to arithmetic / group means are rejected up front when types are known
        numeric = {"a", "b"}
        self.assertIsNotNone(validate_op({"op": "group_mean", "by": ["a"], "col": "c"}, cols, numeric))
        self.assertIsNone(validate_op({"op": "group_mean", "by": ["c"], "col": "a"}, cols, numeric))
        self.assertIsNotNone(validate_op({"op": "ratio", "a": "c", "b": "a"}, cols, numeric))

    def test_code_op(self):
        code = "def feature(df):\n    return df['a'] * 2 + df['b']"
        out = apply_op(FRAME, {"op": "code", "name": "double a", "code": code})
        self.assertEqual(list(out["code_double_a"]), [4.0, 4, 7, 12])


class GuardTests(unittest.TestCase):
    def test_rejects_dangerous_code(self):
        bad = [
            "import os\ndef feature(df):\n    return df['a']",
            "def feature(df):\n    return open('x').read()",
            "def feature(df):\n    return df.__class__",
            "def feature(df):\n    while True:\n        pass",
            "def feature(df):\n    df.to_csv('leak.csv')\n    return df['a']",
            "def other(df):\n    return df['a']",
            "def feature(df):\n    return __import__('os').getcwd()",
            "def feature(df):\n    return pd.io.common.os.environ['OPENAI_API_KEY']",  # reviewer's bypass
            "def feature(df):\n    pd.read_table('x')\n    return df['a']",
            "def feature(df):\n    df.to_html('x')\n    return df['a']",
            "def feature(df):\n    np.random.seed(1)\n    return df['a']",
            "def feature(df):\n    np.lib.npyio.savetxt('x', df)\n    return df['a']",
            "def feature(df):\n    for i in range(10):\n        pass\n    return df['a']",
            "def feature(df):\n    return np.ones(10**11)",
            "def feature(df):\n    return df['a'] ** 10**9",
            "x" * 4000,
            123,
        ]
        for code in bad:
            with self.assertRaises(GuardError, msg=str(code)[:60]):
                check_code(code)

    def test_allows_normal_vectorised_code(self):
        code = ("def feature(df):\n    age = df['a'] - df['b']\n"
                "    return np.where(age > 0, np.log1p(age.clip(lower=0)), np.nan)")
        check_code(code)
        self.assertEqual(len(run_feature_code(code, FRAME)), 4)

    def test_runtime_failures_become_guard_errors(self):
        cases = ["def feature(df):\n    return df['a'] * 0",                 # constant
                 "def feature(df):\n    return df['missing']",               # KeyError inside
                 "def feature(df):\n    return df[['a', 'b']]",              # two columns
                 "def feature(df):\n    return df['a'].head(2)"]             # wrong length
        for code in cases:
            with self.assertRaises(GuardError, msg=code):
                run_feature_code(code, FRAME)

    def test_categorical_result(self):
        out = run_feature_code("def feature(df):\n    return df['c'].astype(str) + '_' + df['a'].astype(str)", FRAME)
        self.assertIsInstance(out.dtype, pd.CategoricalDtype)

    def test_validate_op_never_raises_on_garbage(self):
        cols = {"a", "b", "c"}
        garbage = [{"op": "ratio", "a": ["a"], "b": "b"}, {"op": "ratio", "a": {"x": 1}, "b": "b"},
                   {"op": "group_mean", "by": [["a"]], "col": "a"}, {"op": "group_mean", "by": 5, "col": "a"},
                   {"op": "freq", "col": None}, {"op": "code", "name": "x", "code": "import os"},
                   {"op": None}, [], None, "ratio"]
        for op in garbage:
            self.assertIsNotNone(validate_op(op, cols), msg=repr(op))


class BaseFrameTests(unittest.TestCase):
    def test_hints_dates_and_constants(self):
        train = pd.DataFrame({"id": [1, 2, 3] * 10, "role_id": [10, 20, 30] * 10, "x": np.arange(30.0),
                              "when": pd.date_range("2024-01-01", periods=30).astype(str),
                              "const": 1, "y": np.arange(30.0)})
        test = train.drop(columns="y")
        dictionary = pd.DataFrame({"feature": ["role_id", "x", "when"], "role": ["feature"] * 3,
                                   "dtype": ["int", "float", "str"],
                                   "description_ru": ["Идентификатор роли", "число", "дата"]})
        s = spec()
        hints = categorical_hints(s, train, dictionary)
        self.assertEqual(hints, ["role_id", "when"])
        frame, notes = base_frame(train, test, s, categorical=["role_id"])
        self.assertIsInstance(frame["role_id"].dtype, pd.CategoricalDtype)
        self.assertIn("when__day", frame.columns)
        self.assertNotIn("const", frame.columns)
        self.assertEqual(len(frame), 60)

    def test_timezone_dates_do_not_crash(self):
        stamps = [f"2021-03-{d:02d}T10:00:00{'Z' if d % 2 else '+03:00'}" for d in range(1, 31)]
        train = pd.DataFrame({"id": range(30), "when": stamps, "y": np.arange(30.0)})
        frame, notes = base_frame(train, train.drop(columns="y"), spec())
        self.assertIn("when__day", frame.columns)
        self.assertTrue(frame["when__ts"].notna().all())


if __name__ == "__main__":
    unittest.main()
