"""Experiment: average the refit over 1, 3, 5, 10 seeds.

Finding: mean of 10 seeds 32.28 vs 31.12 for one seed on wids_seen, 12.17 vs 10.88 on
wids_future (single seeds ranged 1.9..19.1). Became REFIT_SEEDS / bag_seeds in the agent.
Usage: python scripts/experiments/refit_seed_bagging_probe.py agent polygon/tasks
"""
import sys, json
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, sys.argv[1])
from automl.history import find_entity_key, history_features

def load(task):
    root = f"{sys.argv[2]}/{task}"
    tr = pd.read_csv(f"{root}/public/train.csv"); te = pd.read_csv(f"{root}/public/test.csv")
    ans = pd.read_csv(f"{root}/private/answers.csv"); meta = json.load(open(f"{root}/private/meta.json"))
    return tr, te, ans["site_eui"].to_numpy(), meta["baseline_value"]

def frame(tr, te):
    key, _ = find_entity_key(tr, te, "Year_Factor", {"id", "site_eui"}, log=lambda m: None)
    h = history_features(tr, te, key, "Year_Factor", tr["site_eui"].to_numpy())
    X = pd.concat([tr.drop(columns=["id", "site_eui"]), te.drop(columns=["id"])], ignore_index=True)
    for c in X.columns[X.dtypes == object]:
        X[c] = X[c].astype("category")
    X = pd.concat([X, h.reset_index(drop=True)], axis=1)
    return X.iloc[:len(tr)].reset_index(drop=True), X.iloc[len(tr):].reset_index(drop=True)

BASE = dict(n_estimators=5000, learning_rate=0.1, num_leaves=63, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=8, random_state=42)

for task in ("wids_seen", "wids_future"):
    tr, te, y_te, B = load(task)
    X, Xt = frame(tr, te)
    y, yr = tr["site_eui"].to_numpy(), tr["Year_Factor"].to_numpy()
    last = yr.max()
    preds = []
    for seed in range(1, 11):
        p = dict(BASE, random_state=seed)
        m = lgb.LGBMRegressor(**p).fit(X[yr < last], y[yr < last], eval_set=[(X[yr == last], y[yr == last])],
                                       callbacks=[lgb.early_stopping(100, verbose=False)])
        preds.append(lgb.LGBMRegressor(**dict(p, n_estimators=m.best_iteration_)).fit(X, y).predict(Xt))
    sc = lambda pr: 100 * (1 - float(np.sqrt(np.mean((pr - y_te) ** 2))) / B)
    singles = [sc(p) for p in preds]
    for k in (1, 3, 5, 10):
        groups = [np.mean(preds[i:i + k], axis=0) for i in range(0, 10 - k + 1, k)]
        print(f"{task:12s} mean of {k:2d} seeds: score {np.mean([sc(g) for g in groups]):6.2f} (groups: {[round(sc(g), 2) for g in groups]})")
