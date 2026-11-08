"""Feature table for v2: a base frame over train+test rows and declarative feature ops.

Ops never see the target, so they cannot leak it; train and test get identical treatment.
A recipe is a list of op dicts, e.g.
  {"op": "ratio", "a": "floor_area", "b": "year_built"}
  {"op": "group_mean", "by": ["State_Factor"], "col": "energy_star_rating"}
  {"op": "code", "name": "building_age", "code": "def feature(df): ..."}
"""
import re

import numpy as np
import pandas as pd

from automl.guard import GuardError, check_code, run_block_code, run_feature_code
from automl.task import TaskSpec

CATEGORY_DTYPES = {"category", "categorical", "cat", "str", "string", "object", "text"}
ID_WORDS = re.compile(r"идентификатор|\bкод\b|\bid\b|\bидентиф", re.IGNORECASE)
OPS = {
    "ratio": "a / b",
    "diff": "a - b",
    "product": "a * b",
    "sum": "a + b",
    "log1p": "log(1 + col) for non-negative col",
    "freq": "how often the value of col occurs (count encoding)",
    "group_mean": "mean of col within groups defined by `by` columns",
    "group_std": "std of col within groups",
    "group_dev": "col minus its group mean",
    "group_count": "size of the group defined by `by` columns",
    "group_nunique": "number of distinct values of col within groups",
    "as_category": "treat col as categorical (codes / IDs stored as numbers)",
    "drop": "remove col",
    "code": "custom pandas code: def feature(df): return Series",
    "code_block": "developer-written block: def features(df): return DataFrame of up to 20 new columns",
}


def read_dictionary(spec: TaskSpec, log=print) -> pd.DataFrame | None:
    if spec.dict_path is None:
        log("dictionary: no data_dictionary.csv")
        return None
    try:
        return pd.read_csv(spec.dict_path)
    except Exception as error:
        log(f"dictionary: unreadable ({error!r}), continuing without hints")
        return None


def categorical_hints(spec: TaskSpec, train: pd.DataFrame, dictionary: pd.DataFrame | None) -> list[str]:
    """Numeric columns that are really categories: dictionary dtype says so, or the
    description calls an integer column an identifier/code."""
    if dictionary is None or "feature" not in dictionary.columns:
        return []
    hints = []
    for _, row in dictionary.iterrows():
        col = str(row.get("feature"))
        if col not in train.columns or col in (spec.id_col, spec.target) or str(row.get("role")) != "feature":
            continue
        dtype = str(row.get("dtype", "")).strip().lower()
        desc = str(row.get("description_ru", ""))
        integer_like = pd.api.types.is_integer_dtype(train[col]) or (
            pd.api.types.is_float_dtype(train[col]) and bool((train[col].dropna() % 1 == 0).all()))
        if dtype in CATEGORY_DTYPES or (integer_like and ID_WORDS.search(desc)):
            hints.append(col)
    return hints


def _looks_like_datetime(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if len(sample) < 20 or not sample.str.contains(r"\d{1,4}[-/.:]\d{1,2}", regex=True).mean() > 0.9:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.9


def base_frame(train: pd.DataFrame, test: pd.DataFrame, spec: TaskSpec,
               categorical: list[str] = (), drop: list[str] = ()) -> tuple[pd.DataFrame, list[str]]:
    """Rows: train then test. Strings and hinted columns become categories, date strings
    become parts, constants and free text are dropped. Returns (frame, notes)."""
    columns = [c for c in test.columns if c in train.columns and c not in (spec.id_col, spec.target)
               and c not in set(drop)]
    df = pd.concat([train[columns], test[columns]], ignore_index=True)
    notes = []
    for col in columns:
        try:
            df, note = _prepare_column(df, col, col in categorical)
        except Exception as error:  # one odd column must not cost the whole v2 stage
            df[col] = df[col].astype("string").astype("category")
            note = f"{col}: fallback to category after {error!r}"[:200]
        if note:
            notes.append(note)
    return df, notes


def _prepare_column(df: pd.DataFrame, col: str, hinted_category: bool) -> tuple[pd.DataFrame, str]:
    s = df[col]
    if s.nunique(dropna=False) <= 1:
        return df.drop(columns=col), f"drop constant {col}"
    if pd.api.types.is_bool_dtype(s):
        df[col] = s.astype(float)
    elif hinted_category:
        df[col] = s.astype("string").astype("category")
    elif not pd.api.types.is_numeric_dtype(s):
        if _looks_like_datetime(s):
            # utc=True handles "Z"/"+03:00" and mixed offsets; then drop the zone.
            parsed = pd.to_datetime(s, errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
            for part in ("year", "month", "day", "dayofweek", "hour"):
                values = getattr(parsed.dt, part)
                if values.nunique() > 1:
                    df[f"{col}__{part}"] = values.astype(float)
            df[f"{col}__ts"] = (parsed - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
            return df.drop(columns=col), f"datetime {col} -> parts"
        if s.nunique() > max(1000, 0.5 * len(s)):
            return df.drop(columns=col), f"drop free-text {col}"
        df[col] = s.astype("string").astype("category")
    else:
        df[col] = s.replace([np.inf, -np.inf], np.nan)
    return df, ""


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if isinstance(s.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(s):
        raise ValueError(f"{col} is not numeric")
    return s.astype(float)


def _key(df: pd.DataFrame, by: list[str]) -> pd.Series:
    """Integer group id for the combination of `by` values (missing values form a group)."""
    for col in by:
        if col not in df.columns:
            raise ValueError(f"unknown column {col}")
    return df.groupby(list(by), dropna=False, observed=True, sort=False).ngroup()


def op_name(op: dict) -> str:
    kind = op.get("op")
    if kind in ("code", "code_block"):
        prefix = "code" if kind == "code" else "block"
        return f"{prefix}_{re.sub(r'[^0-9A-Za-z_]+', '_', str(op.get('name', 'feature')))[:40]}"
    parts = [kind] + [str(op[k]) if not isinstance(op[k], list) else "+".join(map(str, op[k]))
                      for k in ("a", "b", "col", "by") if k in op]
    return "__".join(parts)


def apply_op(df: pd.DataFrame, op: dict) -> pd.DataFrame:
    """Return df with the op applied (new column added or column changed/dropped)."""
    kind = op.get("op")
    if kind not in OPS:
        raise ValueError(f"unknown op {kind!r}")
    name = op_name(op)
    out = df.copy()
    if kind in ("ratio", "diff", "product", "sum"):
        a, b = _num(df, op["a"]), _num(df, op["b"])
        value = {"ratio": a / b.replace(0, np.nan), "diff": a - b, "product": a * b, "sum": a + b}[kind]
        out[name] = value.replace([np.inf, -np.inf], np.nan)
    elif kind == "log1p":
        s = _num(df, op["col"])
        out[name] = np.log1p(s.clip(lower=0))
    elif kind == "freq":
        key = _key(df, [op["col"]])
        out[name] = key.map(key.value_counts()).astype(float)
    elif kind in ("group_mean", "group_std", "group_dev", "group_nunique"):
        key = _key(df, list(op["by"]))
        if kind == "group_nunique":
            out[name] = df[op["col"]].groupby(key).transform("nunique").astype(float)
        else:
            s = _num(df, op["col"])
            stat = s.groupby(key).transform("std" if kind == "group_std" else "mean")
            out[name] = s - stat if kind == "group_dev" else stat
    elif kind == "group_count":
        key = _key(df, list(op["by"]))
        out[name] = key.map(key.value_counts()).astype(float)
    elif kind == "as_category":
        col = op["col"]
        if col not in df.columns:
            raise ValueError(f"unknown column {col}")
        out[col] = df[col].astype("string").astype("category")
        return out
    elif kind == "drop":
        if op["col"] not in df.columns:
            raise ValueError(f"unknown column {op['col']}")
        return out.drop(columns=op["col"])
    elif kind == "code":
        out[name] = run_feature_code(str(op["code"]), df)
    elif kind == "code_block":
        block = run_block_code(str(op["code"]), df)
        for col in block.columns:
            out[f"{name}__{re.sub(r'[^0-9A-Za-z_]+', '_', col)[:40]}"] = block[col]
        return out
    if name in out.columns and out[name].nunique(dropna=False) <= 1:
        raise ValueError(f"{name} is constant")
    return out


def apply_recipe(base: pd.DataFrame, recipe: list[dict], log=None, strict: bool = False) -> pd.DataFrame:
    """Apply ops in order. strict=True raises on the first broken op (used to evaluate a new
    idea honestly); otherwise a broken op is logged and skipped."""
    df = base
    for op in recipe:
        try:
            df = apply_op(df, op)
        except Exception as error:
            if strict:
                raise ValueError(f"op {op_name(op) if isinstance(op, dict) else op} failed: {error!r}") from error
            if log:
                log(f"op {op_name(op) if isinstance(op, dict) and 'op' in op else op} skipped: {error!r}")
    return df


OP_FIELDS = {"ratio": ("a", "b"), "diff": ("a", "b"), "product": ("a", "b"), "sum": ("a", "b"),
             "log1p": ("col",), "freq": ("col",), "group_mean": ("by", "col"), "group_std": ("by", "col"),
             "group_dev": ("by", "col"), "group_count": ("by",), "group_nunique": ("by", "col"),
             "as_category": ("col",), "drop": ("col",), "code": ("name", "code"),
             "code_block": ("name", "code")}
assert set(OP_FIELDS) == set(OPS), "OPS and OP_FIELDS must list the same ops"


def normalize_op(op) -> dict | None:
    """Copy of an LLM op with only the known fields; `by` as a list. None if malformed."""
    if not isinstance(op, dict) or op.get("op") not in OP_FIELDS:
        return None
    out = {"op": op["op"]}
    for key in OP_FIELDS[op["op"]]:
        value = op.get(key)
        if key == "by" and isinstance(value, str):
            value = [value]
        out[key] = value
    return out


NUMERIC_FIELDS = {"ratio": ("a", "b"), "diff": ("a", "b"), "product": ("a", "b"), "sum": ("a", "b"),
                  "log1p": ("col",), "group_mean": ("col",), "group_std": ("col",), "group_dev": ("col",)}


def numeric_columns(frame: pd.DataFrame) -> set[str]:
    return {c for c in frame.columns
            if not isinstance(frame[c].dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(frame[c])}


def validate_op(op, columns: set[str], numeric: set[str] | None = None) -> str | None:
    """None if the (normalised) op is well-formed against known columns, else the reason.
    `numeric` (optional) lets arithmetic/group statistics reject categorical inputs up front.
    Never raises, whatever the LLM sent."""
    if not isinstance(op, dict) or op.get("op") not in OP_FIELDS:
        return f"unknown op {op.get('op') if isinstance(op, dict) else type(op).__name__!r}"
    for key in OP_FIELDS[op["op"]]:
        value = op.get(key)
        if key == "by":
            if not isinstance(value, list) or not value or not all(isinstance(c, str) for c in value):
                return f"`by` must be a non-empty list of column names, got {value!r}"[:200]
            if any(c not in columns for c in value):
                return f"unknown group column in {value!r}"[:200]
        elif key in ("a", "b", "col"):
            if not isinstance(value, str) or value not in columns:
                return f"unknown column {value!r}"[:200]
        elif not isinstance(value, str) or not value:
            return f"{op['op']} needs a string {key!r}"
    if numeric is not None:
        for key in NUMERIC_FIELDS.get(op["op"], ()):
            if op[key] not in numeric:
                return f"{op['op']} needs a numeric {key!r}, {op[key]!r} is categorical"[:200]
    if op["op"] in ("code", "code_block"):
        try:
            check_code(op["code"], func="feature" if op["op"] == "code" else "features")
        except GuardError as error:
            return f"code rejected by guard: {error}"
    return None
