"""Собирает тестовые задачи в формате кейса из скачанных датасетов OpenML.

Запуск: .venv/Scripts/python.exe -I polygon/build_tasks.py

Вход:  polygon/raw/<openml_id>/data.pq
Выход: polygon/tasks/<name>/public/  — то, что видит агент (task.md, train.csv, test.csv,
                                       data_dictionary.csv, sample_submission.csv)
       polygon/tasks/<name>/private/ — answers.csv и meta.json со скором бейзлайна
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import descriptions  # noqa: E402
import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parent
SEED = 42

METRIC_TEXT = {
    "roc_auc": (
        "Метрика — площадь под ROC-кривой (ROC AUC) по вероятностям положительного класса. "
        "Чем выше, тем лучше."
    ),
    "rmsle": (
        "Метрика — RMSLE (корень из среднеквадратичной логарифмической ошибки):\n\n"
        "RMSLE = sqrt( (1/n) · Σ (log(1 + ŷ_i) − log(1 + y_i))² ),\n\n"
        "где y_i — истинное значение, ŷ_i — предсказание. Чем меньше, тем лучше."
    ),
}

TASKS = [
    {
        "name": "amazon_access",
        "openml_id": 4135,
        "target_src": "target",
        "target": "ACTION",
        "task_type": "binary",
        "metric": "roc_auc",
        "describe": lambda c: descriptions.AMAZON[c],
        "title": "Одобрение доступа сотрудников к ресурсам",
        "intro": (
            "Когда сотрудник начинает работу в компании, ему нужен доступ к ресурсам: приложениям, "
            "порталам, базам данных. Сейчас запросы на доступ вручную проверяют руководители. "
            "По историческим решениям нужно построить модель, которая предсказывает, одобрят ли "
            "сотруднику доступ к ресурсу. Все признаки — закодированные идентификаторы ролей, "
            "подразделений и ресурсов."
        ),
        "target_text": "`ACTION` — 1, если доступ одобрен, 0 — если отклонён.",
        "submission_text": "вероятность того, что `ACTION` = 1",
    },
    {
        "name": "bike_sharing",
        "openml_id": 42712,
        "target_src": "count",
        "target": "count",
        "drop": ["casual", "registered"],
        "task_type": "regression",
        "metric": "rmsle",
        "describe": lambda c: descriptions.BIKE[c],
        "title": "Прогноз спроса на прокат велосипедов",
        "intro": (
            "Городская система велопроката собирает почасовую статистику аренды вместе с "
            "календарём и погодой. Нужно предсказать, сколько велосипедов возьмут в прокат "
            "за каждый час из тестовой выборки."
        ),
        "target_text": "`count` — количество велосипедов, арендованных за час.",
        "submission_text": "прогноз `count` (неотрицательное число)",
    },
    {
        "name": "home_credit",
        "openml_id": 45567,
        "target_src": "TARGET",
        "target": "TARGET",
        "id_src": "SK_ID_CURR",
        "sample_n": 110_000,
        "task_type": "binary",
        "metric": "roc_auc",
        "describe": descriptions.home_credit,
        "title": "Оценка риска дефолта по кредитной заявке",
        "intro": (
            "Банк выдаёт потребительские кредиты клиентам, у многих из которых нет кредитной "
            "истории. По анкетным данным заявки, сведениям о жилье, документах и запросах в бюро "
            "кредитных историй нужно оценить вероятность того, что у клиента возникнут трудности "
            "с выплатой кредита."
        ),
        "target_text": "`TARGET` — 1, если у клиента были трудности с выплатой, 0 — если нет.",
        "submission_text": "вероятность того, что `TARGET` = 1",
    },
]


def load(spec: dict) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "raw" / str(spec["openml_id"]) / "data.pq")
    df = df.drop(columns=spec.get("drop", []))
    df = df.rename(columns={spec["target_src"]: spec["target"]})

    # Категории из OpenML приводим к обычным типам, как в CSV организаторов.
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            values = s.astype(object)
            as_num = pd.to_numeric(values, errors="coerce")
            df[col] = as_num.astype("Int64") if as_num.notna().sum() == values.notna().sum() else values
        elif s.dtype == bool:
            df[col] = s.astype(int)

    if spec.get("id_src"):
        df = df.rename(columns={spec["id_src"]: "id"})
        df["id"] = df["id"].astype("int64")
    if spec.get("sample_n") and len(df) > spec["sample_n"]:
        df, _ = train_test_split(
            df, train_size=spec["sample_n"], random_state=SEED,
            stratify=df[spec["target"]] if spec["task_type"] != "regression" else None,
        )
    if "id" not in df.columns:
        df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        df.insert(0, "id", np.arange(len(df)))
    cols = ["id"] + [c for c in df.columns if c not in ("id", spec["target"])] + [spec["target"]]
    return df[cols].reset_index(drop=True)


def dtype_name(s: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        return "float"
    return "str"


def fit_predict(kind: str, task_type: str, X_tr, y_tr, X_te):
    """kind='baseline' — простая линейная модель; kind='lgbm' — LightGBM по умолчанию (ориентир)."""
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == object]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    if kind == "lgbm":
        X_tr, X_te = X_tr.copy(), X_te.copy()
        for c in cat_cols:
            cats = pd.Categorical(X_tr[c]).categories
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats)
            X_te[c] = pd.Categorical(X_te[c], categories=cats)
        X_tr[num_cols] = X_tr[num_cols].astype(float)
        X_te[num_cols] = X_te[num_cols].astype(float)
        if task_type == "regression":
            model = lgb.LGBMRegressor(random_state=SEED, verbose=-1)
            return model.fit(X_tr, y_tr).predict(X_te)
        model = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
        return model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=20)), cat_cols),
    ])
    X_tr = X_tr.astype({c: float for c in num_cols})
    X_te = X_te.astype({c: float for c in num_cols})
    if task_type == "regression":
        return make_pipeline(pre, Ridge()).fit(X_tr, y_tr).predict(X_te)
    model = make_pipeline(pre, LogisticRegression(max_iter=2000))
    return model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def task_md(spec: dict, n_train: int, n_test: int, n_features: int) -> str:
    target = spec["target"]
    return f"""# {spec['title']}

## Описание

{spec['intro']}

## Данные

Все CSV-файлы используют кодировку UTF-8, запятую как разделитель и точку как десятичный
разделитель. Пустая ячейка означает пропуск.

- `train.csv` — обучающая выборка: {n_features} входных признаков, идентификатор `id` и таргет `{target}`. Объём — {n_train} строк.
- `test.csv` — тестовая выборка: те же признаки и `id`, без таргета. Объём — {n_test} строк.
- `data_dictionary.csv` — описание столбцов: `feature`, `role`, `dtype`, `unit`, `description_ru`.
- `sample_submission.csv` — пример сабмита: колонки `id` и `{target}`.

## Целевая переменная

{spec['target_text']}

## Метрика

{METRIC_TEXT[spec['metric']]}

## Формат ответа

Файл с колонками `id` и `{target}`: для каждой строки `test.csv` — {spec['submission_text']}.
Порядок и количество строк должны совпадать с `test.csv`.
"""


def build(spec: dict) -> dict:
    df = load(spec)
    target = spec["target"]
    stratify = df[target] if spec["task_type"] != "regression" else None
    train, test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=stratify)
    train = train.sort_values("id").reset_index(drop=True)
    test = test.sort_values("id").reset_index(drop=True)
    features = [c for c in df.columns if c not in ("id", target)]

    out = ROOT / "tasks" / spec["name"]
    public, private = out / "public", out / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train.to_csv(public / "train.csv", index=False)
    test.drop(columns=[target]).to_csv(public / "test.csv", index=False)
    test[["id", target]].to_csv(private / "answers.csv", index=False)

    placeholder = 0.5 if spec["task_type"] == "binary" else float(train[target].median())
    pd.DataFrame({"id": test["id"], target: placeholder}).to_csv(public / "sample_submission.csv", index=False)

    rows = []
    for col in df.columns:
        role = "id" if col == "id" else "target" if col == target else "feature"
        desc, unit = spec["describe"](col)
        rows.append({"feature": col, "role": role, "dtype": dtype_name(df[col]), "unit": unit, "description_ru": desc})
    pd.DataFrame(rows).to_csv(public / "data_dictionary.csv", index=False)

    (public / "task.md").write_text(task_md(spec, len(train), len(test), len(features)), encoding="utf-8")

    y_tr, y_te = train[target].astype(float), test[target].astype(float)
    results = {}
    for kind in ("baseline", "lgbm"):
        pred = fit_predict(kind, spec["task_type"], train[features], y_tr, test[features])
        if spec["task_type"] == "regression":
            pred = np.clip(pred, 0, None)
        results[kind] = metrics.compute(spec["metric"], y_te, pred)

    meta = {
        "name": spec["name"],
        "source": f"OpenML {spec['openml_id']}",
        "task_type": spec["task_type"],
        "target": target,
        "metric": spec["metric"],
        "lower_is_better": spec["metric"] in metrics.LOWER_IS_BETTER,
        "n_train": len(train),
        "n_test": len(test),
        "n_features": len(features),
        "seed": SEED,
        "baseline_model": "LogisticRegression/Ridge + median impute + scaling + one-hot (min_frequency=20)",
        "baseline_value": results["baseline"],
        "reference_model": "LightGBM с параметрами по умолчанию",
        "reference_value": results["lgbm"],
        "reference_score": metrics.to_score(spec["metric"], results["lgbm"], results["baseline"]),
    }
    (private / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    only = set(sys.argv[1:])
    for spec in TASKS:
        if only and spec["name"] not in only:
            continue
        meta = build(spec)
        print(f"{meta['name']:15} train={meta['n_train']:>6} test={meta['n_test']:>6} feats={meta['n_features']:>3} "
              f"{meta['metric']}: baseline={meta['baseline_value']:.4f} lgbm={meta['reference_value']:.4f} "
              f"→ балл LightGBM {meta['reference_score']:.1f}")
