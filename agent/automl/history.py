"""Entity history for temporal tasks: "what did the same object show in earlier periods?"

When the test is a future period and the same objects (buildings, clients, devices) repeat
across periods, the object's own past target is often the strongest signal. Generic steps:
  1. find_entity_key: columns that identify an object. Static columns are those whose exact
     values reappear from one period to the next (a building's area does; a year's weather
     does not). The key must be fine (about one row per object and period) and must cover a
     good share of test rows. An LLM-proposed key goes through the same check.
  2. history_features: for every row, the object's target in strictly EARLIER periods only
     (last value, mean, count, gap). Train rows never see their own or later targets, so the
     features are leak-free under the temporal validation in cv.make_problem.
Switch: AGENT_HISTORY (default on).
"""
import time

import numpy as np
import pandas as pd

MAX_ROWS_PER_OBJECT = 1.3  # mean rows per (key, period): above this the key merges different objects
MIN_TEST_COVERAGE = 0.3  # share of test rows whose object appeared in train
MIN_PERSISTENCE = 0.2  # share of objects seen in the previous period
MIN_COLUMN_OVERLAP = 0.3  # a key column's values must recur between consecutive periods
KEEP_SHARE = 0.85  # a refining column may cost at most 15% of the recurrence
ANCHOR_SHARE = 0.1  # anchor: distinct values >= 10% of the rows of an average period
MAX_ANCHORS = 4
MAX_POOL = 30  # columns with the highest overlap considered for the key
MAX_PERIODS = 50  # more periods means continuous time: "seen last period" has no meaning
LLM_KEY_SLACK = 0.9  # the LLM key wins at >= 90% of the rule key's persistence x coverage


def column_codes(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    """Integer codes per column over train+test rows; missing values get code 0, values 1.."""
    both = pd.concat([train[columns], test[columns]], ignore_index=True)
    codes = {}
    for col in columns:
        values = both[col]
        if values.dtype == object or isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype("string")  # mixed types and categories group by their text
        codes[col] = pd.factorize(values, use_na_sentinel=True)[0].astype(np.int64) + 1
    return codes


def combine(code_list: list[np.ndarray]) -> np.ndarray:
    """One dense id per distinct combination of codes."""
    ids = np.zeros(len(code_list[0]), dtype=np.int64)
    for codes in code_list:
        ids = pd.factorize(ids * (int(codes.max()) + 1) + codes)[0].astype(np.int64)
    return ids


class _KeySearch:
    """Key statistics on integer codes, with per-period row indices computed once."""

    def __init__(self, codes: dict[str, np.ndarray], times: np.ndarray):
        self.codes = codes
        self.n_train = len(times)
        periods, self.period_of = np.unique(times, return_inverse=True)
        order = np.argsort(self.period_of, kind="stable")
        bounds = np.searchsorted(self.period_of[order], np.arange(len(periods) + 1))
        self.rows = [order[a:b] for a, b in zip(bounds[:-1], bounds[1:])]

    def overlap(self, col: str) -> float:
        """Share of a period's non-missing values already seen in the previous period."""
        tr = self.codes[col][:self.n_train]
        shares = []
        for prev, nxt in zip(self.rows[:-1], self.rows[1:]):
            cur = tr[nxt][tr[nxt] > 0]
            if len(cur):
                shares.append(np.isin(cur, tr[prev]).mean())
        return float(np.mean(shares)) if shares else 0.0

    def stats(self, key: list[str]) -> dict:
        ids = combine([self.codes[c] for c in key])
        tr, te = ids[:self.n_train], ids[self.n_train:]
        shares = [np.isin(tr[nxt], tr[prev]).mean() for prev, nxt in zip(self.rows[:-1], self.rows[1:])]
        groups = pd.factorize(tr * len(self.rows) + self.period_of)[0]
        return {"persistence": float(np.mean(shares)) if shares else 0.0,
                "rows_per_object": len(tr) / max(int(groups.max()) + 1, 1),
                "coverage": float(np.isin(te, tr).mean()) if len(te) else 0.0}


def _passes(q: dict) -> bool:
    return (q["rows_per_object"] <= MAX_ROWS_PER_OBJECT and q["coverage"] >= MIN_TEST_COVERAGE
            and q["persistence"] >= MIN_PERSISTENCE)


def _describe(q: dict) -> str:
    return (f"persistence {q['persistence']:.2f}, coverage {q['coverage']:.2f}, "
            f"rows/object {q['rows_per_object']:.2f}")


def find_entity_key(train: pd.DataFrame, test: pd.DataFrame, time_col: str, exclude: set[str],
                    proposed: list[str] | None = None, log=print,
                    deadline: float = float("inf")) -> tuple[list[str] | None, dict]:
    """Return (key columns, quality) or (None, info with "reason"). `proposed` (e.g. from the LLM
    analyst) wins when it passes the same check and scores at least LLM_KEY_SLACK of the rule
    key's persistence x coverage.

    Rule: start from an "anchor" — a column with many distinct values that recur over time (a
    building's floor area, a client's birth date). Then add refining columns one by one while
    the key gets finer and objects still recur (persistence may fall at most to KEEP_SHARE of
    its value: separating two buildings with the same area costs a little; adding a column that
    changes every period — the year's weather, a yearly rating — halves it and is refused).
    Test coverage may fall by the same share at most: a column that stayed put over the train
    periods but changes in the test period (a heat-days count of a new year) would stop test
    objects from matching their past. The best anchor's key by persistence x coverage wins. Past `deadline` the search stops and
    keeps the best key found so far."""
    times = train[time_col].to_numpy()
    n_periods = len(pd.unique(times))
    if n_periods > MAX_PERIODS:
        return None, {"reason": f"{n_periods} periods > {MAX_PERIODS}: time looks continuous", "pool": []}
    candidates = [c for c in train.columns if c in test.columns and c not in exclude and c != time_col]
    if proposed:
        dropped = [c for c in proposed if c not in candidates]
        if dropped:
            log(f"history: LLM key columns not usable (absent from test, time or target): {dropped}")
        proposed = [c for c in proposed if c in candidates]
    search = _KeySearch(column_codes(train, test, candidates), times)
    overlap = {}
    for col in candidates:
        if time.time() > deadline:
            return None, {"reason": "out of time while measuring column overlap", "pool": []}
        overlap[col] = search.overlap(col)
    pool = sorted((c for c in candidates if overlap[c] >= MIN_COLUMN_OVERLAP), key=overlap.get, reverse=True)
    pool = pool[:MAX_POOL]
    info = {"pool": pool, "overlap": {c: round(overlap[c], 2) for c in pool}}

    best, best_q = None, None
    per_period = len(train) / max(n_periods, 1)
    cardinality = {c: int(search.codes[c][:len(train)].max()) for c in pool}
    anchors = [c for c in pool if cardinality[c] >= ANCHOR_SHARE * per_period]
    if not anchors and pool:
        anchors = [max(pool, key=cardinality.get)]
    out_of_time = False
    for anchor in sorted(anchors, key=cardinality.get, reverse=True)[:MAX_ANCHORS]:
        key, q = [anchor], search.stats([anchor])
        while not out_of_time:
            trials = []
            for col in pool:
                if time.time() > deadline:
                    out_of_time = True
                    break
                if col in key:
                    continue
                tq = search.stats(key + [col])
                if tq["rows_per_object"] < q["rows_per_object"] - 1e-9 and \
                        tq["persistence"] >= KEEP_SHARE * q["persistence"] and \
                        tq["coverage"] >= KEEP_SHARE * q["coverage"]:
                    trials.append((tq["persistence"] * tq["coverage"], -tq["rows_per_object"], col, tq))
            if not trials:
                break
            _, _, col, tq = max(trials, key=lambda item: item[:2])
            key, q = key + [col], tq
        ok = _passes(q)
        log(f"history: anchor {anchor!r} -> key {key}: {_describe(q)}, ok={ok}")
        if ok and (best_q is None or q["persistence"] * q["coverage"] > best_q["persistence"] * best_q["coverage"]):
            best, best_q = key, q
        if out_of_time:
            log("history: key search stopped by its time budget")
            break

    chosen = "rule" if best else None
    if proposed:
        pq = search.stats(proposed)
        rule_score = best_q["persistence"] * best_q["coverage"] if best_q else 0.0
        accepted = _passes(pq) and pq["persistence"] * pq["coverage"] >= LLM_KEY_SLACK * rule_score
        log(f"history: LLM key {proposed}: {_describe(pq)}, accepted={accepted}")
        if accepted:
            best, best_q, chosen = proposed, pq, "LLM"
    if best is None:
        reason = "out of time" if out_of_time else "no key passes the checks"
        return None, {**info, "reason": reason}
    log(f"history: chosen: {chosen} key {best}")
    return best, {**info, **best_q, "chosen": chosen}


def history_features(train: pd.DataFrame, test: pd.DataFrame, key: list[str], time_col: str,
                     y: np.ndarray) -> pd.DataFrame:
    """Rows: train then test (same order as features.base_frame). Columns: hist_last, hist_mean,
    hist_count, hist_gap — the object's target in strictly earlier periods (NaN if none)."""
    codes = column_codes(train, test, key)
    key_id = combine([codes[c] for c in key])
    times = np.concatenate([train[time_col].to_numpy(dtype=float), test[time_col].to_numpy(dtype=float)])
    n_train = len(train)

    past = pd.DataFrame({"key_id": key_id[:n_train], "period": times[:n_train], "y": np.asarray(y, float)})
    past = past.groupby(["key_id", "period"], as_index=False)["y"].mean().sort_values(["key_id", "period"])
    past["cum_mean"] = past.groupby("key_id")["y"].cumsum() / (past.groupby("key_id").cumcount() + 1)
    past["cum_count"] = past.groupby("key_id").cumcount() + 1
    past["past_period"] = past["period"]
    past = past.rename(columns={"y": "last"}).sort_values("period")

    rows = pd.DataFrame({"key_id": key_id, "period": times, "row": np.arange(len(times))}).sort_values("period")
    merged = pd.merge_asof(rows, past, on="period", by="key_id", allow_exact_matches=False, direction="backward")
    merged["hist_gap"] = merged["period"] - merged["past_period"]
    merged = merged.sort_values("row")
    return pd.DataFrame({"hist_last": merged["last"].to_numpy(), "hist_mean": merged["cum_mean"].to_numpy(),
                         "hist_count": merged["cum_count"].to_numpy(dtype=float),
                         "hist_gap": merged["hist_gap"].to_numpy(dtype=float)})
