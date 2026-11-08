"""Polygon tasks derived from the organizers' public tasks, reproducing their traps with known answers.

  wids_future  — buildings (site_eui, RMSE): train on periods Year_Factor 1-4, test = period 5.
                 The public task tests on a FUTURE period (6) and most test buildings repeat
                 from earlier years; random CV hides both effects.
  wids_seen    — the same split without test rows whose category (state, facility type) never
                 occurs in train. In the public task every test category occurs in train; in
                 wids_future 17% of test rows come from a state first seen in period 5, and their
                 error hides everything else. wids_future stays as a robustness check.
  unsw_shift   — network attacks (attack_cat, Macro F1): test has different class shares than
                 train (more Normal, fewer attacks), like UNSW-NB15's official hold-out split.

Usage: python -I polygon/build_derived.py <organizers_regression_dir> <organizers_multiclass_dir>
Each dir holds train.csv and data_dictionary.csv (any sub-folder). Output: polygon/tasks/<name>/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED = 42
# Multipliers of class shares in the test part (Normal up, attacks down), then renormalised.
SHIFT = {"Normal": 1.45, "Generic": 1.0, "Exploits": 0.7, "Fuzzers": 0.7, "DoS": 0.7, "Reconnaissance": 0.7,
         "Analysis": 0.75, "Backdoor": 0.75, "Shellcode": 0.75, "Worms": 0.75}


def find(folder: Path, name: str) -> Path:
    return sorted(folder.rglob(name), key=lambda p: len(p.parts))[0]


def baseline(task_type: str, X_tr: pd.DataFrame, y_tr, X_te: pd.DataFrame):
    """The 'simplest model' reference used for scoring, as in build_tasks.py."""
    cat = [c for c in X_tr.columns if X_tr[c].dtype == object]
    num = [c for c in X_tr.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), cat)])
    model = Ridge() if task_type == "regression" else LogisticRegression(max_iter=1000)
    return make_pipeline(pre, model).fit(X_tr, y_tr).predict(X_te)


def task_md(title: str, intro: str, target: str, n_train: int, n_test: int, metric_text: str,
            answer_text: str) -> str:
    return f"""# {title}

## Описание

{intro}

## Данные

Все CSV-файлы используют кодировку UTF-8, запятую как разделитель и точку как десятичный разделитель.

- `train.csv` — обучающая выборка с таргетом `{target}`, {n_train} строк.
- `test.csv` — тестовая выборка без таргета, {n_test} строк.
- `data_dictionary.csv` — описание столбцов.
- `sample_submission.csv` — пример ответа: колонки `id` и `{target}`.

## Метрика

{metric_text}

## Формат результата

Файл `submission.csv` рядом с этим файлом, колонки `id` и `{target}`: {answer_text}.
Порядок строк совпадает с `test.csv`.
"""


def write(name: str, train: pd.DataFrame, test: pd.DataFrame, target: str, dictionary: pd.DataFrame,
          md: str, metric: str, task_type: str, sample_value) -> dict:
    public, private = ROOT / "tasks" / name / "public", ROOT / "tasks" / name / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    train.to_csv(public / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(public / "test.csv", index=False)
    test[["id", target]].to_csv(private / "answers.csv", index=False)
    pd.DataFrame({"id": test["id"], target: sample_value}).to_csv(public / "sample_submission.csv", index=False)
    dictionary.to_csv(public / "data_dictionary.csv", index=False)
    (public / "task.md").write_text(md, encoding="utf-8")
    features = [c for c in train.columns if c not in ("id", target)]
    pred = baseline(task_type, train[features], train[target], test[features])
    value = metrics.compute(metric, test[target], pred)
    meta = {"name": name, "metric": metric, "target": target, "task_type": task_type, "baseline_value": value}
    (private / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def build_wids(src: Path, seen_only: bool = False) -> dict:
    df = pd.read_csv(find(src, "train.csv"))
    dictionary = pd.read_csv(find(src, "data_dictionary.csv"))
    train = df[df["Year_Factor"] <= 4].reset_index(drop=True)
    test = df[df["Year_Factor"] == 5].reset_index(drop=True)
    if seen_only:
        for col in train.columns[train.dtypes == object]:
            test = test[test[col].isin(set(train[col].dropna())) | test[col].isna()]
        test = test.reset_index(drop=True)
    md = task_md("Энергопотребление зданий (периоды)",
                 "По характеристикам здания и погоде за год оцените годовое удельное энергопотребление "
                 "`site_eui` (kBtu/ft² за год). Одна строка — наблюдение за зданием в определённом году.",
                 "site_eui", len(train), len(test), "Метрика оценки — **RMSE**. Меньшее значение лучше.",
                 "одно числовое предсказание `site_eui` на строку")
    name = "wids_seen" if seen_only else "wids_future"
    return write(name, train, test, "site_eui", dictionary, md, "rmse", "regression", 0.0)


def build_unsw(src: Path) -> dict:
    df = pd.read_csv(find(src, "train.csv"))
    dictionary = pd.read_csv(find(src, "data_dictionary.csv"))
    rng = np.random.default_rng(SEED)
    is_test = np.zeros(len(df), dtype=bool)
    for label, rows in df.groupby("attack_cat").indices.items():
        is_test[rng.choice(rows, size=int(0.3 * len(rows)), replace=False)] = True
    train, pool = df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)
    shares = pool["attack_cat"].value_counts(normalize=True)
    wanted = {k: shares[k] * SHIFT.get(k, 1.0) for k in shares.index}
    total = sum(wanted.values())
    wanted = {k: v / total for k, v in wanted.items()}
    # Largest test size that keeps every class within what the pool has.
    size = int(min(pool["attack_cat"].eq(k).sum() / wanted[k] for k in wanted))
    parts = [pool[pool["attack_cat"] == k].sample(n=max(1, int(round(wanted[k] * size))), random_state=SEED)
             for k in wanted]
    test = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    md = task_md("Типы сетевых атак (сдвиг классов)",
                 "По характеристикам сетевого потока определите тип атаки `attack_cat` (одно из десяти "
                 "значений, включая `Normal`). Классы несбалансированы.",
                 "attack_cat", len(train), len(test),
                 "Основная метрика — **Macro F1**: `f1_score(y_true, y_pred, average=\"macro\")`.",
                 "одна строковая метка класса на строку")
    return write("unsw_shift", train, test, "attack_cat", dictionary, md, "macro_f1", "multiclass", "Normal")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for meta in (build_wids(Path(sys.argv[1])), build_wids(Path(sys.argv[1]), seen_only=True),
                 build_unsw(Path(sys.argv[2]))):
        print(meta)
