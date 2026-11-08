"""Guard for feature code written by the LLM: static AST checks plus a restricted namespace.

The code must define `def feature(df):` returning a pandas Series (or array) of len(df).
Allowed: methods on the DataFrame/Series it gets, a WHITELIST of pd.* / np.* functions and a few
builtins. Not allowed: imports, loops (for/while), dunder or private attributes, file/network
access, attribute chains into pandas/numpy internals (pd.io..., np.lib...), huge literals.
The function first runs on a small sample, then on the full frame under a time limit.
"""
import ast
import builtins
import threading

import numpy as np
import pandas as pd

MAX_CODE_CHARS = 3000
MAX_LITERAL = 10 ** 6
TIME_LIMIT_SEC = 60
SAMPLE_ROWS = 1000
PD_ALLOWED = {"Series", "DataFrame", "to_numeric", "to_datetime", "to_timedelta", "cut", "qcut", "isna",
              "notna", "isnull", "notnull", "Timestamp", "Timedelta", "concat", "NA", "NaT", "Categorical",
              "factorize", "crosstab"}
NP_ALLOWED = {"log", "log1p", "log2", "log10", "exp", "expm1", "sqrt", "abs", "absolute", "where", "clip",
              "nan", "inf", "pi", "e", "sign", "floor", "ceil", "round", "rint", "maximum", "minimum",
              "fmax", "fmin", "mean", "median", "std", "var", "sum", "prod", "min", "max", "nanmean",
              "nanmedian", "nanstd", "nansum", "nanmin", "nanmax", "isnan", "isfinite", "isinf", "select",
              "sin", "cos", "tan", "arctan", "arctan2", "hypot", "power", "square", "reciprocal",
              "digitize", "percentile", "quantile", "float64", "float32", "int64", "int32", "array",
              "asarray", "nan_to_num", "logical_and", "logical_or", "logical_not", "divide", "multiply",
              "add", "subtract", "mod", "remainder", "floor_divide", "cumsum", "diff", "argsort", "unique",
              "bincount", "searchsorted", "interp", "tanh", "radians", "degrees"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help",
    "os", "sys", "subprocess", "shutil", "socket", "pathlib", "importlib", "builtins",
    "memoryview", "type", "object", "classmethod", "staticmethod", "super", "property",
}
FORBIDDEN_ATTRS = {"to_csv", "to_parquet", "to_pickle", "to_json", "to_excel", "to_html", "to_latex",
                   "to_xml", "to_feather", "to_hdf", "to_sql", "to_stata", "to_clipboard", "to_markdown",
                   "to_string", "to_records", "read_csv", "read_parquet", "read_pickle", "read_json",
                   "load", "save", "savetxt", "tofile", "fromfile", "system", "popen", "eval", "query",
                   "pipe", "style", "plot", "hist", "attrs", "flags"}
SAFE_BUILTINS = {name: getattr(builtins, name)
                 for name in ("abs", "min", "max", "len", "range", "sum", "round", "float", "int",
                              "str", "list", "dict", "zip", "enumerate", "isinstance", "any", "all",
                              "sorted", "map", "filter", "set", "tuple", "bool", "pow", "ValueError",
                              "TypeError", "KeyError", "ZeroDivisionError", "Exception")}


class GuardError(ValueError):
    pass


def check_code(code: str, func: str = "feature") -> None:
    """Raise GuardError if the code is not allowed to run or does not define `func(df)`."""
    if not isinstance(code, str) or len(code) > MAX_CODE_CHARS:
        raise GuardError(f"code must be a string up to {MAX_CODE_CHARS} chars")
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise GuardError(f"syntax error: {error}") from None
    has_feature = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise GuardError("imports are not allowed")
        if isinstance(node, (ast.While, ast.For, ast.AsyncFor, ast.AsyncFunctionDef, ast.Await, ast.Global,
                             ast.Nonlocal, ast.ClassDef, ast.Delete, ast.With, ast.AsyncWith, ast.Try)):
            raise GuardError(f"{type(node).__name__} is not allowed (write vectorised pandas code)")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise GuardError(f"name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS:
                raise GuardError(f"attribute {node.attr!r} is not allowed")
            if isinstance(node.value, ast.Name) and node.value.id == "pd" and node.attr not in PD_ALLOWED:
                raise GuardError(f"pd.{node.attr} is not in the allowed list")
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr not in NP_ALLOWED:
                raise GuardError(f"np.{node.attr} is not in the allowed list")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool) and abs(node.value) > MAX_LITERAL:
            raise GuardError(f"literal {node.value} is too large")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) \
                and isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)) \
                and abs(node.right.value) > 8:
            raise GuardError("large powers are not allowed")
        if isinstance(node, ast.FunctionDef) and node.name == func:
            has_feature = True
    if not has_feature:
        raise GuardError(f"code must define `def {func}(df):`")


def _call_with_timeout(func, df: pd.DataFrame, seconds: float):
    box = {}

    def target():
        try:
            box["value"] = func(df)
        except BaseException as error:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = error

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():  # the thread cannot be killed; it is a daemon and its result is ignored
        raise GuardError(f"feature code ran longer than {seconds:.0f}s")
    if "error" in box:
        raise GuardError(f"feature code raised {box['error']!r}"[:300])
    return box.get("value")


def _to_series(result, df: pd.DataFrame) -> pd.Series:
    if isinstance(result, pd.DataFrame):
        if result.shape[1] != 1:
            raise GuardError("feature must return one column")
        result = result.iloc[:, 0]
    values = result.to_numpy() if isinstance(result, pd.Series) else np.asarray(result)
    if values.ndim != 1 or len(values) != len(df):
        raise GuardError(f"feature returned shape {values.shape} for {len(df)} rows")
    series = pd.Series(values, index=df.index)
    if pd.api.types.is_bool_dtype(series):
        series = series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    return series.astype("string").astype("category")


MAX_BLOCK_COLUMNS = 20


def run_block_code(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """Developer component: `def features(df): return DataFrame` with up to MAX_BLOCK_COLUMNS new
    columns (row-aligned with df). Same guard, sample run and time limit as single features."""
    check_code(code, func="features")
    namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    exec(compile(code, "<llm_features>", "exec"), namespace)  # noqa: S102 - guarded above
    func = namespace["features"]

    def to_frame(result, frame):
        if isinstance(result, pd.Series):
            result = result.to_frame()
        if not isinstance(result, pd.DataFrame):
            raise GuardError(f"features must return a DataFrame, got {type(result).__name__}")
        if result.shape[1] == 0 or result.shape[1] > MAX_BLOCK_COLUMNS:
            raise GuardError(f"features returned {result.shape[1]} columns (1..{MAX_BLOCK_COLUMNS} allowed)")
        if len(result) != len(frame):
            raise GuardError(f"features returned {len(result)} rows for {len(frame)}")
        out = pd.DataFrame(index=frame.index)
        for name in result.columns:
            out[str(name)] = _to_series(result[name].to_numpy(), frame)
        return out.loc[:, out.nunique(dropna=False) > 1]

    sample = df.head(SAMPLE_ROWS).copy()
    to_frame(_call_with_timeout(func, sample, TIME_LIMIT_SEC / 6), sample)
    block = to_frame(_call_with_timeout(func, df.copy(), TIME_LIMIT_SEC), df)
    if block.shape[1] == 0:
        raise GuardError("all returned columns are constant")
    return block


def run_feature_code(code: str, df: pd.DataFrame) -> pd.Series:
    """Check and run the code on a copy of df; return a numeric or categorical Series."""
    check_code(code)
    namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    exec(compile(code, "<llm_feature>", "exec"), namespace)  # noqa: S102 - guarded above
    feature = namespace["feature"]
    sample = df.head(SAMPLE_ROWS).copy()
    _to_series(_call_with_timeout(feature, sample, TIME_LIMIT_SEC / 6), sample)  # fail fast on a sample
    series = _to_series(_call_with_timeout(feature, df.copy(), TIME_LIMIT_SEC), df)
    if series.nunique(dropna=False) <= 1:
        raise GuardError("feature is constant")
    return series
