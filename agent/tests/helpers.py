"""Synthetic tasks in the same layout as the hackathon platform."""
from pathlib import Path

import numpy as np
import pandas as pd


def make_task(root: Path, kind: str, *, nested: bool = True, metric_text: str | None = None,
              n_train: int = 600, n_test: int = 200, output_name: str = "submission.csv",
              rare_rows: int = 5, odd_column: bool = False, sample_value=None,
              answer_col: str | None = None, extra_md: str = "") -> Path:
    """Create task.md plus train/test/sample/dictionary CSVs and return the task.md path."""
    rng = np.random.default_rng(0)
    n = n_train + n_test
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    if kind == "regression":
        target, y = "price", 3 * x1 - 2 * x2 + (cat == "a") * 5 + rng.normal(scale=0.1, size=n) + 20
        metric_text = metric_text or "Метрика оценки — **RMSE**."
        default_sample = 0.0
    elif kind == "binary":
        target, y = "is_bad", (x1 + x2 > 0.5).astype(int)
        metric_text = metric_text or "Основная метрика — ROC-AUC, Accuracy справочная."
        default_sample = 0.5
    elif kind == "multiclass":
        target = "kind"
        y = np.where(x1 > 0.5, "high", np.where(x1 < -0.5, "low", "mid"))
        y[:rare_rows] = "rare"  # tiny class, as Worms in UNSW-NB15
        metric_text = metric_text or ("Поэтому основной метрикой выбран Macro F1, а не Accuracy.\n"
                                      "Дополнительно рассчитываются Accuracy и Weighted F1.")
        default_sample = "mid"
    else:
        raise ValueError(kind)

    df = pd.DataFrame({"id": np.arange(1000, 1000 + n), "x1": x1, "x2": x2, "cat": cat,
                       "const": 1, target: y})
    if odd_column:
        df = df.rename(columns={"x2": 'temp [C], "raw"'})
    df.loc[df.sample(frac=0.05, random_state=1).index, df.columns[2]] = np.nan
    train = df.iloc[:n_train]
    test = df.iloc[n_train:].drop(columns=[target]).sample(frac=1, random_state=2)  # shuffled ids

    data_dir = root / "data" / "participants" if nested else root
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], answer_col or target:
                  default_sample if sample_value is None else sample_value}).to_csv(
        data_dir / "sample_submission.csv", index=False)
    pd.DataFrame({"feature": list(df.columns),
                  "role": ["id"] + ["feature"] * (len(df.columns) - 2) + ["target"]}).to_csv(
        data_dir / "data_dictionary.csv", index=False)

    prefix = "data/participants/" if nested else ""
    md = (f"# Synthetic {kind}\n\n## Данные\n\n{extra_md}"
          f"- `{prefix}train.csv` — обучение;\n- `{prefix}test.csv` — тест;\n"
          f"- `{prefix}data_dictionary.csv` — словарь;\n"
          f"- `{prefix}sample_submission.csv` — шаблон ответа.\n\n"
          f"## Что предсказать\n\n{metric_text}\n\n"
          f"## Формат результата\n\nФайл `{output_name}` в каталоге этого Markdown-файла.\n")
    task_path = root / "task.md"
    task_path.write_text(md, encoding="utf-8")
    return task_path
