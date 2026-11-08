"""Experiment: learning rate of the refit, three seeds each.

Finding: the single-seed gain of lr 0.03 was seed noise; on wids_future one model scores
anywhere from 8.8 to 14.1 at the same lr. Led to seed bagging (refit_seed_bagging_probe.py).
Usage: python scripts/experiments/refit_lr_seed_probe.py agent polygon/tasks
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
VARIANTS = {}
for lr in (0.1, 0.05, 0.03, 0.02):
    for seed in (1, 2, 3):
        VARIANTS[f"lr {lr} seed {seed}"] = (dict(BASE, learning_rate=lr, random_state=seed), 1.0)
for task in ("wids_seen", "wids_future"):
    tr, te, y_te, B = load(task)
    X, Xt = frame(tr, te)
    y, yr = tr["site_eui"].to_numpy(), tr["Year_Factor"].to_numpy()
    last = yr.max()
    for name, (p, scale) in VARIANTS.items():
        m = lgb.LGBMRegressor(**p).fit(X[yr < last], y[yr < last], eval_set=[(X[yr == last], y[yr == last])],
                                       callbacks=[lgb.early_stopping(100, verbose=False)])
        n = max(1, int(round(m.best_iteration_ * scale)))
        full = lgb.LGBMRegressor(**dict(p, n_estimators=n)).fit(X, y)
        rmse = float(np.sqrt(np.mean((full.predict(Xt) - y_te) ** 2)))
        print(f"{task:12s} {name:32s} trees={n:5d} rmse={rmse:7.3f} score={100 * (1 - rmse / B):6.2f}")
