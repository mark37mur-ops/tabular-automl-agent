import tempfile
import unittest
from pathlib import Path

import pandas as pd

from automl.task import build_spec, detect_metric
from tests.helpers import make_task


class DetectMetricTests(unittest.TestCase):
    def test_primary_metric_beats_earlier_reference_metric(self):
        cases = [
            "Additional metric: Accuracy. Primary metric: Macro F1.",
            "Дополнительная метрика: Accuracy. Основная метрика: Macro F1.",
            "Дополнительно оценивается Accuracy.\nОсновная\nметрика:\n**Macro F1**.",
            "Additional metric: MAE.\n## Main metric\n\n**RMSE**\n",
            "Additional metric: Accuracy. Macro F1 is the primary metric.",
        ]
        for text in cases:
            with self.subTest(text=text):
                expected = "rmse" if "RMSE" in text else "f1_macro"
                self.assertEqual(detect_metric(text), expected)

    def test_micro_f1_preserves_its_average(self):
        for name in ("Micro F1", "micro-averaged F1-score", "F1 (micro)",
                     "f1_score(y, p, average='micro')"):
            with self.subTest(name=name):
                self.assertEqual(detect_metric(f"Main metric:\n{name}."), "f1_micro")

    def test_main_metric_wins_over_reference_metrics(self):
        text = ("Поэтому основной\nметрикой выбран Macro F1, а не Accuracy.\n\n"
                "Дополнительно рассчитываются **Accuracy** и **Weighted F1**.")
        self.assertEqual(detect_metric(text), "f1_macro")

    def test_regression_metrics(self):
        self.assertEqual(detect_metric("Метрика оценки — **RMSE** на исходной шкале."), "rmse")
        self.assertEqual(detect_metric("Метрика: RMSLE."), "rmsle")
        self.assertEqual(detect_metric("Качество оценивается по MAE."), "mae")

    def test_auc_and_unknown(self):
        self.assertEqual(detect_metric("Основная метрика — ROC-AUC."), "roc_auc")
        self.assertIsNone(detect_metric("Просто предскажите таргет."))

    def test_sklearn_function_names(self):
        self.assertEqual(detect_metric("Метрика: `f1_score(y, p)`."), "f1")
        self.assertEqual(detect_metric("Метрика: `mean_absolute_error`."), "mae")
        self.assertEqual(detect_metric("Метрика: root mean squared error."), "rmse")
        self.assertEqual(detect_metric("Метрика: `roc_auc_score`."), "roc_auc")


class BuildSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_regression_nested_layout(self):
        spec = build_spec(make_task(self.root, "regression"))
        self.assertEqual((spec.task_type, spec.metric, spec.target, spec.id_col),
                         ("regression", "rmse", "price", "id"))
        self.assertEqual(spec.pred_format, "value")
        self.assertEqual(spec.train_path, self.root / "data" / "participants" / "train.csv")
        self.assertEqual(spec.output_path, self.root / "submission.csv")

    def test_flat_layout_files_next_to_task_md(self):
        spec = build_spec(make_task(self.root, "regression", nested=False))
        self.assertEqual(spec.train_path, self.root / "train.csv")

    def test_multiclass_macro_f1(self):
        spec = build_spec(make_task(self.root, "multiclass"))
        self.assertEqual((spec.task_type, spec.metric, spec.pred_format),
                         ("multiclass", "f1_macro", "label"))
        self.assertEqual(sorted(spec.classes), ["high", "low", "mid", "rare"])

    def test_binary_auc_predicts_probability(self):
        spec = build_spec(make_task(self.root, "binary"))
        self.assertEqual((spec.task_type, spec.metric, spec.pred_format),
                         ("binary", "roc_auc", "proba"))
        self.assertEqual(spec.positive_class, 1)

    def test_probability_contract_overrides_minority_heuristic(self):
        task_path = make_task(self.root, "binary", extra_md="Return P(target=approved).\n")
        train_path = self.root / "data" / "participants" / "train.csv"
        train = pd.read_csv(train_path)
        train["is_bad"] = train["is_bad"].map({0: "approved", 1: "rejected"})
        self.assertGreater((train["is_bad"] == "approved").sum(),
                           (train["is_bad"] == "rejected").sum())
        train.to_csv(train_path, index=False)
        spec = build_spec(task_path)
        self.assertEqual((spec.positive_class, spec.pred_format), ("approved", "proba"))
        self.assertTrue(any("probability contract" in note for note in spec.notes))

    def test_probability_contract_overrides_numeric_one_heuristic(self):
        task_path = make_task(self.root, "binary", extra_md="Return P(is_bad = 0).\n")
        self.assertEqual(build_spec(task_path).positive_class, 0)

    def test_probability_contract_matches_quoted_class(self):
        task_path = make_task(self.root, "binary", extra_md="Return P(is_bad = '0').\n")
        self.assertEqual(build_spec(task_path).positive_class, 0)

    def test_unknown_probability_contract_class_is_rejected(self):
        task_path = make_task(self.root, "binary", extra_md="Return P(target=missing).\n")
        with self.assertRaisesRegex(ValueError, "unknown positive class"):
            build_spec(task_path)

    def test_main_metric_beats_earlier_additional_metric_in_task(self):
        task_path = make_task(self.root, "multiclass", metric_text=(
            "Additional metric: Accuracy.\n\n## Primary metric\n\nMacro F1."))
        self.assertEqual(build_spec(task_path).metric, "f1_macro")

    def test_multiclass_micro_f1_is_not_changed_to_macro(self):
        task_path = make_task(self.root, "multiclass", metric_text="Main metric: Micro F1.")
        self.assertEqual(build_spec(task_path).metric, "f1_micro")

    def test_unknown_metric_falls_back_to_type_default(self):
        spec = build_spec(make_task(self.root, "multiclass", metric_text="Угадайте класс."))
        self.assertEqual((spec.task_type, spec.metric), ("multiclass", "f1_macro"))

    def test_custom_output_name(self):
        spec = build_spec(make_task(self.root, "regression", output_name="answer.csv"))
        self.assertEqual(spec.output_path, self.root / "answer.csv")

    def test_output_file_ignores_other_file_mentions(self):
        spec = build_spec(make_task(self.root, "regression",
                                    extra_md="Файл `meta.csv` содержит справку.\n\n"))
        self.assertEqual(spec.output_path, self.root / "submission.csv")

    def test_plain_f1_on_multiclass_becomes_macro(self):
        spec = build_spec(make_task(self.root, "multiclass",
                                    metric_text="Метрика: macro-averaged F1-score."))
        self.assertEqual(spec.metric, "f1_macro")

    def test_unknown_metric_binary_with_label_sample_predicts_labels(self):
        spec = build_spec(make_task(self.root, "binary", metric_text="Предскажите класс.",
                                    sample_value=0))
        self.assertEqual(spec.pred_format, "label")

    def test_answer_column_taken_from_sample(self):
        spec = build_spec(make_task(self.root, "regression", answer_col="prediction"))
        self.assertEqual((spec.target, spec.answer_col), ("price", "prediction"))


if __name__ == "__main__":
    unittest.main()
