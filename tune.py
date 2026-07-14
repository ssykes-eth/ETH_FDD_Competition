"""
Staged hyperparameter / feature-selection / outlier-contamination search.

Reuses the exact building blocks in pipeline.py (so results transfer directly
to main.py) but caches per-fold intermediates (cleaned training data, feature
rankings) across sweep values that don't invalidate them, since MI/importance
ranking and IsolationForest fitting are the expensive steps (~5-10s each) and
would otherwise be recomputed redundantly for every hyperparameter combo.

Stages (each stage fixes what the previous stage found best):
  1. n_features x contamination, fs_method="mi"        (3-fold, fixed baseline XGB params)
  2. fs_method comparison at the stage-1 winner          (5-fold)
  3. randomized XGB hyperparameter search                (5-fold, reusing cached selected data)
  4. local reg_lambda refinement around the stage-3 winner (5-fold)

Prints a table after each stage and the final chosen config. Run once,
offline, to pick CONFIG for main.py -- not part of the production pipeline.

Run: python tune.py
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pipeline import (
    RANDOM_STATE,
    fit_preprocessing,
    apply_preprocessing,
    remove_outlier_rows,
    select_features,
    fit_xgb,
    load_data,
    _mi_ranking,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASELINE_XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_lambda=2.0,
    min_child_weight=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

N_FEATURES_GRID = [50, 100, 200, 300, 500, "all"]
CONTAMINATION_GRID = [0, 0.01, 0.02, 0.05, 0.10]
FS_METHOD_GRID = ["mi", "xgb_importance"]

RNG = np.random.RandomState(RANDOM_STATE)


def score_fold(model, X_val, y_val):
    preds = model.predict(X_val)
    return {
        "r2": r2_score(y_val, preds),
        "mae": mean_absolute_error(y_val, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
    }


def summarize(rows):
    r2 = np.array([r["r2"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows])
    return {
        "r2_mean": r2.mean(), "r2_std": r2.std(),
        "mae_mean": mae.mean(), "mae_std": mae.std(),
        "rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
    }


def build_folds(X, y, n_splits):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [
        (X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
        for tr, va in kf.split(X)
    ]


def n_features_value(n_features, n_available):
    return n_available if n_features == "all" else n_features


def main():
    X_train, y_train, _ = load_data()
    print(f"X_train {X_train.shape}")

    results = {"stage1": [], "stage2": [], "stage3": []}

    # ---------------- Stage 1: n_features x contamination (fs=mi, 3-fold) ----------------
    print("\n=== Stage 1: n_features x contamination (fs_method=mi, 3-fold CV) ===")
    folds3 = build_folds(X_train, y_train, n_splits=3)

    # cache per (fold_idx, contamination): (X_tr_clean, y_tr_clean, X_val_imp, mi_ranking)
    cache = {}
    for fi, (X_tr, y_tr, X_val, y_val) in enumerate(folds3):
        state, X_tr_imp = fit_preprocessing(X_tr)
        X_val_imp = apply_preprocessing(X_val, state)
        for contamination in CONTAMINATION_GRID:
            X_tr_clean, y_tr_clean = remove_outlier_rows(X_tr_imp, y_tr, contamination)
            mi_scores = _mi_ranking(X_tr_clean, y_tr_clean).sort_values(ascending=False)
            cache[(fi, contamination)] = (X_tr_clean, y_tr_clean, X_val_imp, y_val, mi_scores)

    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for contamination in CONTAMINATION_GRID:
            fold_rows = []
            for fi in range(len(folds3)):
                X_tr_clean, y_tr_clean, X_val_imp, y_val, mi_scores = cache[(fi, contamination)]
                k = n_features_value(n_features, len(mi_scores))
                cols = mi_scores.head(k).index
                model = fit_xgb(X_tr_clean[cols], y_tr_clean, BASELINE_XGB_PARAMS)
                fold_rows.append(score_fold(model, X_val_imp[cols], y_val))
            summary = summarize(fold_rows)
            summary.update(n_features=n_features, contamination=contamination)
            results["stage1"].append(summary)
            print(f"  n_features={n_features!s:>4} contamination={contamination:<5} "
                  f"R2={summary['r2_mean']:.4f}+/-{summary['r2_std']:.4f} "
                  f"MAE={summary['mae_mean']:.3f} RMSE={summary['rmse_mean']:.3f}")
    print(f"Stage 1 took {time.time()-t0:.0f}s")

    best1 = max(results["stage1"], key=lambda r: r["r2_mean"])
    best_n_features, best_contamination = best1["n_features"], best1["contamination"]
    print(f"Stage 1 winner: n_features={best_n_features} contamination={best_contamination} "
          f"(R2={best1['r2_mean']:.4f})")

    # ---------------- Stage 2: fs_method comparison (5-fold) ----------------
    print("\n=== Stage 2: feature-selection method comparison (5-fold CV) ===")
    folds5 = build_folds(X_train, y_train, n_splits=5)
    t0 = time.time()
    for fs_method in FS_METHOD_GRID:
        fold_rows = []
        for X_tr, y_tr, X_val, y_val in folds5:
            state, X_tr_imp = fit_preprocessing(X_tr)
            X_val_imp = apply_preprocessing(X_val, state)
            X_tr_clean, y_tr_clean = remove_outlier_rows(X_tr_imp, y_tr, best_contamination)
            k = n_features_value(best_n_features, X_tr_clean.shape[1])
            cols = select_features(X_tr_clean, y_tr_clean, method=fs_method, n_features=k)
            model = fit_xgb(X_tr_clean[cols], y_tr_clean, BASELINE_XGB_PARAMS)
            fold_rows.append(score_fold(model, X_val_imp[cols], y_val))
        summary = summarize(fold_rows)
        summary.update(fs_method=fs_method)
        results["stage2"].append(summary)
        print(f"  fs_method={fs_method:<15} R2={summary['r2_mean']:.4f}+/-{summary['r2_std']:.4f} "
              f"MAE={summary['mae_mean']:.3f} RMSE={summary['rmse_mean']:.3f}")
    print(f"Stage 2 took {time.time()-t0:.0f}s")

    best2 = max(results["stage2"], key=lambda r: r["r2_mean"])
    best_fs_method = best2["fs_method"]
    print(f"Stage 2 winner: fs_method={best_fs_method} (R2={best2['r2_mean']:.4f})")

    # ---------------- Stage 3: XGB hyperparameter random search (5-fold) ----------------
    print("\n=== Stage 3: XGB hyperparameter random search (5-fold CV) ===")
    # Precompute the selected, cleaned data per fold once -- fixed across all hyperparam draws.
    fold_data = []
    for X_tr, y_tr, X_val, y_val in folds5:
        state, X_tr_imp = fit_preprocessing(X_tr)
        X_val_imp = apply_preprocessing(X_val, state)
        X_tr_clean, y_tr_clean = remove_outlier_rows(X_tr_imp, y_tr, best_contamination)
        k = n_features_value(best_n_features, X_tr_clean.shape[1])
        cols = select_features(X_tr_clean, y_tr_clean, method=best_fs_method, n_features=k)
        fold_data.append((X_tr_clean[cols], y_tr_clean, X_val_imp[cols], y_val))

    search_space = dict(
        max_depth=[3, 4, 5, 6],
        learning_rate=[0.01, 0.02, 0.03, 0.05],
        n_estimators=[400, 600, 800],
        subsample=[0.6, 0.7, 0.8, 0.9],
        colsample_bytree=[0.3, 0.4, 0.5, 0.7],
        reg_lambda=[1.0, 2.0, 3.0, 5.0],
        min_child_weight=[1, 3, 5, 10],
    )
    n_draws = 15
    t0 = time.time()
    for draw in range(n_draws):
        params = {k: RNG.choice(v) for k, v in search_space.items()}
        params = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        params.update(random_state=RANDOM_STATE, n_jobs=-1)
        fold_rows = []
        for X_tr_sel, y_tr_clean, X_val_sel, y_val in fold_data:
            model = fit_xgb(X_tr_sel, y_tr_clean, params)
            fold_rows.append(score_fold(model, X_val_sel, y_val))
        summary = summarize(fold_rows)
        summary.update(params=params)
        results["stage3"].append(summary)
        print(f"  draw {draw+1:>2}: R2={summary['r2_mean']:.4f}+/-{summary['r2_std']:.4f} "
              f"MAE={summary['mae_mean']:.3f} RMSE={summary['rmse_mean']:.3f}  {params}")
    print(f"Stage 3 took {time.time()-t0:.0f}s")

    # BASELINE_XGB_PARAMS is itself a candidate -- the random draws above sample
    # the space around it but don't include it verbatim.
    baseline_summary = dict(summarize([
        score_fold(fit_xgb(X_tr_sel, y_tr_clean, dict(BASELINE_XGB_PARAMS)), X_val_sel, y_val)
        for X_tr_sel, y_tr_clean, X_val_sel, y_val in fold_data
    ]), params=dict(BASELINE_XGB_PARAMS))
    stage3_candidates = results["stage3"] + [baseline_summary]
    best3 = max(stage3_candidates, key=lambda r: r["r2_mean"])
    print(f"\nStage 3 winner (incl. baseline): R2={best3['r2_mean']:.4f}+/-{best3['r2_std']:.4f} "
          f"MAE={best3['mae_mean']:.3f} RMSE={best3['rmse_mean']:.3f}")
    print(f"  params={best3['params']}")

    # ---------------- Stage 4: local refinement of reg_lambda around the winner (5-fold) ----------------
    print("\n=== Stage 4: local reg_lambda refinement around stage-3 winner (5-fold CV) ===")
    t0 = time.time()
    for reg_lambda in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
        params = dict(best3["params"], reg_lambda=reg_lambda)
        fold_rows = [
            score_fold(fit_xgb(X_tr_sel, y_tr_clean, params), X_val_sel, y_val)
            for X_tr_sel, y_tr_clean, X_val_sel, y_val in fold_data
        ]
        summary = summarize(fold_rows)
        summary.update(params=params)
        results["stage4"] = results.get("stage4", []) + [summary]
        print(f"  reg_lambda={reg_lambda:<4} R2={summary['r2_mean']:.4f}+/-{summary['r2_std']:.4f} "
              f"MAE={summary['mae_mean']:.3f} RMSE={summary['rmse_mean']:.3f}")
    print(f"Stage 4 took {time.time()-t0:.0f}s")

    best4 = max(results["stage4"], key=lambda r: r["r2_mean"])
    best_overall = max([best3, best4], key=lambda r: r["r2_mean"])
    print(f"\nStage 4 winner: R2={best4['r2_mean']:.4f}+/-{best4['r2_std']:.4f}  params={best4['params']}")

    print("\n=== FINAL CHOSEN CONFIG ===")
    print(f"fs_method={best_fs_method}")
    print(f"n_features={best_n_features}")
    print(f"contamination={best_contamination}")
    print(f"xgb_params={best_overall['params']}")
    print(f"CV R2={best_overall['r2_mean']:.4f}+/-{best_overall['r2_std']:.4f} "
          f"MAE={best_overall['mae_mean']:.3f} RMSE={best_overall['rmse_mean']:.3f}")

    with open("tune_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nWrote tune_results.json")


if __name__ == "__main__":
    main()
