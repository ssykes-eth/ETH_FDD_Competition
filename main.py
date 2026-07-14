"""
Brain Age Prediction pipeline.

Cross-validates the pipeline defined in pipeline.py (leak-free: every fitted
step — cell-outlier bounds, imputer, IsolationForest, feature ranking, and
the model(s) themselves — is refit inside each training fold and only ever
applied to, never fit on, the held-out fold), then fits the same pipeline
once on the full training set and writes submission.csv.

The model is a 5-way stacking ensemble (see pipeline.build_base_models):
two XGBoost configs, ElasticNet, ExtraTrees, and HistGradientBoosting,
blended by a non-negative-least-squares meta-model trained on out-of-fold
predictions. This beat a single tuned XGBoost on every CV fold (see
eval_ensemble.py): R^2 0.5168 vs. 0.5015, MAE 4.99 vs. 5.12, RMSE 6.72 vs.
6.83. Set MODEL_TYPE="xgb" to fall back to the single-model pipeline (~15x
cheaper to fit, since the ensemble's inner CV refits 5 models per fold).

Run: python main.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from pipeline import RANDOM_STATE, load_data, run_pipeline

N_SELECTED_FEATURES = 300
FEATURE_SELECTION_METHOD = "xgb_importance"
OUTLIER_CONTAMINATION = 0.0
NEAR_CONSTANT_STD = None  # e.g. 1e-8 to also drop near-zero-variance columns; see tune.py
MODEL_TYPE = "stack"  # "stack" (5-model ensemble, best CV score) or "xgb" (single model, much faster)

# Only used when MODEL_TYPE="xgb"; the "stack" ensemble's own XGBoost members
# are configured in pipeline.build_base_models (its "xgb_main" matches this).
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.4,
    reg_lambda=5.0,
    min_child_weight=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

CONFIG = dict(
    fs_method=FEATURE_SELECTION_METHOD,
    n_features=N_SELECTED_FEATURES,
    contamination=OUTLIER_CONTAMINATION,
    xgb_params=XGB_PARAMS,
    near_constant_std=NEAR_CONSTANT_STD,
    model_type=MODEL_TYPE,
)


def cross_validate(X, y, config, n_splits=5):
    """Run the exact same pipeline used for the final fit inside each CV
    fold, reporting R^2, MAE and RMSE on the untouched validation fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2s, maes, rmses = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, X_val_sel, _ = run_pipeline(X_tr, y_tr, X_val, config)
        preds = model.predict(X_val_sel)

        r2s.append(r2_score(y_val, preds))
        maes.append(mean_absolute_error(y_val, preds))
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
        print(f"  fold {fold}: R^2={r2s[-1]:.4f}  MAE={maes[-1]:.4f}  RMSE={rmses[-1]:.4f}")

    r2s, maes, rmses = np.array(r2s), np.array(maes), np.array(rmses)
    print(f"CV R^2:  {r2s.mean():.4f} +/- {r2s.std():.4f}")
    print(f"CV MAE:  {maes.mean():.4f} +/- {maes.std():.4f}")
    print(f"CV RMSE: {rmses.mean():.4f} +/- {rmses.std():.4f}")
    return {"r2": r2s, "mae": maes, "rmse": rmses}


def main():
    X_train, y_train, X_test = load_data()
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    print("Cross-validating (leak-free: preprocessing/outliers/features refit per fold)...")
    cross_validate(X_train, y_train, CONFIG)

    print("Fitting final model on the full training set...")
    model, X_test_sel, selected = run_pipeline(X_train, y_train, X_test, CONFIG)
    print(f"Selected {len(selected)} / {X_train.shape[1]} features for the final model")

    predictions = model.predict(X_test_sel)
    submission = pd.DataFrame({"id": X_test.index, "y": predictions})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")


if __name__ == "__main__":
    main()
