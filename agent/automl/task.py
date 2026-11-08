"""Task understanding by rules: locate files, find target, task type, metric, answer format."""
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from automl.metrics import CLASSIFICATION, DEFAULT_METRIC, LABEL, PROBA, REGRESSION

# Order matters: more specific patterns first (rmsle before rmse, macro f1 before f1).
# Text is matched with "_" replaced by spaces, so sklearn names (mean_absolute_error) work too.
METRIC_PATTERNS = [
    ("rmsle", r"\brmsle\b|root mean squared? log(arithmic)? error"),
    ("rmse", r"\brmse\b|root mean squared? error"),
    ("mape", r"\bmape\b|mean absolute percentage error"),
    ("mae", r"\bmae\b|mean absolute error"),
    ("mse", r"\bmse\b|mean squared? error"),
    ("r2", r"\br2\b|r²|коэффициент детерминации"),
    ("f1_micro", r"micro[\s-]*(averaged?[\s-]*)?f1|f1[\s-]*(score)?[\s-]*\(?micro"
                 r"|\bf1\s*score\s*\([^)]*\baverage\s*=\s*[\"']micro[\"']"
                 r"|average\s*=\s*[\"']micro"),
    ("f1_macro", r"macro[\s-]*(averaged?[\s-]*)?f1|f1[\s-]*(score)?[\s-]*\(?macro"
                 r"|average\s*=\s*[\"']macro"),
    ("f1_weighted", r"weighted[\s-]*(averaged?[\s-]*)?f1|f1[\s-]*(score)?[\s-]*\(?weighted"),
    ("pr_auc", r"\bpr[\s-]*auc|average precision"),
    ("roc_auc", r"roc[\s-]*auc|\bauc\b"),
    ("logloss", r"log[\s-]*loss|кросс[\s-]*энтропи|cross[\s-]*entropy"),
    ("balanced_accuracy", r"balanced[\s-]*accuracy"),
    ("accuracy", r"\baccuracy\b"),
    ("f1", r"\bf1\b"),
]
METRIC_CONTEXT = re.compile(r"метрик|metric|оцени|оценк|score|скор", re.IGNORECASE)
PRIMARY_METRIC_CONTEXT = re.compile(
    r"\b(?:primary|main)\s+(?:evaluation\s+)?(?:metric|score)\b"
    r"|\b(?:основн|главн|целев)[а-яё]*\s+(?:оценочн[а-яё]*\s+)?метрик[а-яё]*\b",
    re.IGNORECASE,
)
DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv", "data_dictionary.csv")


@dataclass
class TaskSpec:
    """Everything the pipeline needs to know about a task.

    classes: raw target values sorted by str(); class index i in model.py means classes[i].
    target: column in train.csv; answer_col: column name expected in the answer file.
    notes: guesses made while building the spec, logged by solution.py.
    """
    task_dir: Path
    train_path: Path
    test_path: Path
    sample_path: Path
    dict_path: Path | None
    output_path: Path
    id_col: str
    target: str
    answer_col: str
    task_type: str  # regression | binary | multiclass
    metric: str
    pred_format: str  # value | label | proba
    classes: list | None = None
    positive_class: object = None
    notes: list[str] = field(default_factory=list)
    task_path: Path | None = None


def _find_metrics(text: str) -> list[tuple[int, str]]:
    found = []
    for priority, (name, pattern) in enumerate(METRIC_PATTERNS):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            found.append((match.start(), priority, match.end(), name))
    found.sort()
    # Drop hits nested in an earlier, more specific one ("f1" inside "weighted f1").
    result, taken_until = [], -1
    for start, _, end, name in found:
        if start >= taken_until:
            result.append((start, name))
            taken_until = end
    return result


def detect_metric(text: str) -> str | None:
    """Honor an explicit primary metric before incidental/reference metric mentions."""
    normalized = re.sub(r"[*`]+", "", text.replace("_", " "))
    flat = re.sub(r"\s+", " ", normalized)
    for context in PRIMARY_METRIC_CONTEXT.finditer(flat):
        # Colons and line breaks belong to the metric heading, not a new sentence.
        tail = re.split(r"[.!?;]", flat[context.end():], maxsplit=1)[0]
        hits = _find_metrics(tail[:300])
        if hits:
            return hits[0][1]
        # Also accept "Macro F1 is the primary metric" in the same sentence.
        head = re.split(r"[.!?;]", flat[:context.start()])[-1]
        hits = _find_metrics(head)
        if hits:
            return hits[-1][1]
    sentences = re.split(r"(?<=[.!?])\s+", flat)
    for sentence in sentences:
        if METRIC_CONTEXT.search(sentence):
            hits = _find_metrics(sentence)
            if hits:
                return hits[0][1]
    hits = _find_metrics(flat)
    return hits[0][1] if hits else None


def _locate(task_dir: Path, md: str, name: str) -> Path | None:
    for mentioned in re.findall(r"`([^`\n]*" + re.escape(name) + r")`", md):
        path = (task_dir / mentioned).resolve()
        if path.is_file():
            return path
    candidates = sorted(task_dir.rglob(name), key=lambda p: (len(p.parts), str(p)))
    return candidates[0].resolve() if candidates else None


def resolve_output_path(task_path: Path) -> Path:
    """Answer file named in task.md ("Файл `submission.csv`"), else submission.csv next to it."""
    task_path = Path(task_path)
    try:
        md = task_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        md = ""
    names = [n for n in re.findall(r"(?:Файл|File)\s+`([^`\n]+\.csv)`", md, flags=re.IGNORECASE)
             if Path(n).name not in DATA_FILES]
    preferred = [n for n in names if re.search(r"sub|answer|ответ|pred", n, re.IGNORECASE)]
    name = (preferred or names or ["submission.csv"])[-1]
    return (task_path.parent / name).resolve()


def _task_type(y: pd.Series, metric: str | None) -> str:
    n_unique = y.nunique()
    if metric in REGRESSION:
        return "regression"
    if metric in CLASSIFICATION:
        return "binary" if n_unique <= 2 else "multiclass"
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "binary" if n_unique <= 2 else "multiclass"
    is_integer = bool((y.dropna() % 1 == 0).all())
    if n_unique <= 2:
        return "binary"
    if is_integer and n_unique <= 20:
        return "multiclass"
    return "regression"


def _positive_class(y: pd.Series):
    classes = sorted(y.dropna().unique().tolist(), key=str)
    for candidate in (1, True, "1", "yes", "Yes", "true", "True"):
        if candidate in classes:
            return candidate
    return y.value_counts().idxmin()  # minority class is usually the "event"


def _explicit_positive_class(text: str, target: str, classes: list):
    """Read the event in an explicit P(target=value) probability contract."""
    name = rf"(?:target|{re.escape(target)})"
    pattern = rf"\bP\s*\(\s*{name}\s*={{1,2}}\s*([^\n)]+)\)"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        value = match.group(1).strip().strip("`\"'").strip()
        exact = [label for label in classes if str(label) == value]
        if exact:
            return exact[0]
        folded = [label for label in classes if str(label).casefold() == value.casefold()]
        if len(folded) == 1:
            return folded[0]
        raise ValueError(f"Probability contract names unknown positive class {value!r}")
    return None


def build_spec(task_path: Path) -> TaskSpec:
    task_path = Path(task_path).resolve()
    task_dir = task_path.parent
    md = task_path.read_text(encoding="utf-8", errors="replace")
    notes = []

    paths = {name: _locate(task_dir, md, name) for name in DATA_FILES}
    for name in DATA_FILES[:3]:
        if paths[name] is None:
            raise FileNotFoundError(f"{name} not found near {task_path}")

    train_cols = pd.read_csv(paths["train.csv"], nrows=0).columns.tolist()
    test_cols = pd.read_csv(paths["test.csv"], nrows=0).columns.tolist()
    sample = pd.read_csv(paths["sample_submission.csv"])

    id_col = sample.columns[0]
    only_in_train = [c for c in train_cols if c not in test_cols]
    sample_targets = [c for c in sample.columns[1:] if c in train_cols]
    target = (sample_targets or only_in_train)[0]
    answer_col = sample.columns[1] if len(sample.columns) > 1 else target
    if not sample_targets:
        notes.append(f"target {target!r} = column missing in test.csv")

    y = pd.read_csv(paths["train.csv"], usecols=[target])[target].dropna()
    detected = detect_metric(md)
    metric = detected
    task_type = _task_type(y, metric)
    if task_type == "multiclass" and metric == "f1":
        metric = "f1_macro"
        notes.append("plain F1 on a multiclass task -> f1_macro")
    if metric is None or (task_type == "regression") != (metric in REGRESSION):
        metric = DEFAULT_METRIC[task_type]
        notes.append(f"metric {detected!r} not usable for {task_type}, default {metric!r}")

    classes, positive, pred_format = None, None, "value"
    if task_type != "regression":
        classes = sorted(y.unique().tolist(), key=str)
        pred_format = "label"
        if task_type == "binary":
            positive = _explicit_positive_class(md, target, classes)
            if positive is None:
                positive = _positive_class(y)
                notes.append(f"positive class guessed as {positive!r}")
            else:
                notes.append(f"positive class from probability contract: {positive!r}")
            sample_values = pd.to_numeric(sample[answer_col], errors="coerce")
            looks_like_proba = sample_values.notna().all() and bool((sample_values % 1 != 0).any())
            if detected in PROBA or (detected not in LABEL and looks_like_proba):
                pred_format = "proba"
            elif detected not in LABEL:
                metric = "accuracy"
                notes.append("sample holds labels and metric is unknown -> accuracy")

    return TaskSpec(task_dir=task_dir, train_path=paths["train.csv"], test_path=paths["test.csv"],
                    sample_path=paths["sample_submission.csv"], dict_path=paths["data_dictionary.csv"],
                    output_path=resolve_output_path(task_path), id_col=id_col, target=target,
                    answer_col=answer_col, task_type=task_type, metric=metric,
                    pred_format=pred_format, classes=classes, positive_class=positive, notes=notes,
                    task_path=task_path)
