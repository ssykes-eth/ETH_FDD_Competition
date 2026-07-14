"""
One-off comparison: single tuned XGBRegressor vs. the StackingEnsemble,
both run through the exact leak-free 5-fold CV harness in main.py.

Run: python eval_ensemble.py
"""
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from pipeline import RANDOM_STATE, load_data, run_pipeline

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.4, reg_lambda=5.0, min_child_weight=1,
    random_state=RANDOM_STATE, n_jobs=-1,
)

CONFIGS = {
    "single_xgb": dict(
        fs_method="xgb_importance", n_features=300, contamination=0.0,
        xgb_params=XGB_PARAMS, near_constant_std=None, model_type="xgb",
    ),
    "stacking_ensemble": dict(
        fs_method="xgb_importance", n_features=300, contamination=0.0,
        near_constant_std=None, model_type="stack",
    ),
}


def cross_validate(X, y, config, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes, rmses = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model, X_val_sel, _ = run_pipeline(X_tr, y_tr, X_val, config)
        preds = model.predict(X_val_sel)
        r2s.append(r2_score(y_val, preds))
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.array(r2s), np.array(maes), np.array(rmses)


def main():
    X, y, _ = load_data()
    for name, config in CONFIGS.items():
        t0 = time.time()
        r2s, maes, rmses = cross_validate(X, y, config)
        print(f"{name}: R2={r2s.mean():.4f}+/-{r2s.std():.4f}  MAE={maes.mean():.4f}  "
              f"RMSE={rmses.mean():.4f}  ({time.time()-t0:.0f}s)  folds={np.round(r2s, 4)}")


if __name__ == "__main__":
    main()
