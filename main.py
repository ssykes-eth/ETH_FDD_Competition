"""
Brain Age Prediction pipeline.

Steps (matching the project's subtasks):
  0. Impute missing values
  1. Detect & remove outlier samples (rows)
  2. Select relevant features, drop irrelevant/redundant ones
  3. Fit a regression model to predict age and produce submission.csv

Run: python brain_age.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

RANDOM_STATE = 0
N_SELECTED_FEATURES = 250
OUTLIER_CONTAMINATION = 0.05
CELL_OUTLIER_Z_THRESH = 15  # robust (MAD-based) z-score threshold for single-cell corruption

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.4,
    reg_lambda=2.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def load_data():
    X_train = pd.read_csv("X_train.csv").set_index("id")
    y_train = pd.read_csv("y_train.csv").set_index("id")["y"]
    X_test = pd.read_csv("X_test.csv").set_index("id")
    return X_train, y_train, X_test


def neutralize_cell_outliers(X_train, X_test, z_thresh=CELL_OUTLIER_Z_THRESH):
    """A handful of cells hold absurd corrupted values (e.g. 1e22) far outside
    their column's own distribution. Flag them via a robust MAD z-score
    (computed on train only) and treat them as missing so imputation fills
    them in instead."""
    med = X_train.median()
    mad = (X_train - med).abs().median().replace(0, np.nan)

    def flag(df):
        modz = 0.6745 * (df - med) / mad
        return modz.abs() > z_thresh

    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[flag(X_train)] = np.nan
    X_test[flag(X_test)] = np.nan
    return X_train, X_test


def impute_and_clean(X_train, X_test):
    """Subtask 0: fill missing values (median, fit on train only).
    Also drops constant columns, which carry no signal."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    non_constant = X_train_imp.nunique() > 1
    keep_cols = non_constant[non_constant].index
    print(f"Dropped {(~non_constant).sum()} constant column(s)")

    return X_train_imp[keep_cols], X_test_imp[keep_cols]


def detect_outlier_rows(X_train_scaled, contamination=OUTLIER_CONTAMINATION):
    """Subtask 1: classify each training sample as outlier / inlier."""
    detector = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=RANDOM_STATE
    )
    labels = detector.fit_predict(X_train_scaled)  # 1 = inlier, -1 = outlier
    is_outlier = labels == -1
    print(f"Flagged {is_outlier.sum()} / {len(labels)} training samples as outliers")
    return is_outlier


def select_features(X_train, y_train, n_features=N_SELECTED_FEATURES):
    """Subtask 2: rank features by mutual information with age and keep the
    top N, discarding irrelevant/redundant ones."""
    mi = mutual_info_regression(X_train, y_train, random_state=RANDOM_STATE)
    ranking = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    selected = ranking.head(n_features).index
    print(f"Selected {len(selected)} / {X_train.shape[1]} features")
    return selected


def evaluate_pipeline(X, y, n_splits=5):
    """Nested cross-validation: feature selection is refit inside each fold
    so the reported score isn't inflated by leakage."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        cols = select_features(X_tr, y_tr)
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr[cols], y_tr)
        scores.append(model.score(X_val[cols], y_val))
    scores = np.array(scores)
    print(f"Cross-validated R^2: {scores.mean():.4f} +/- {scores.std():.4f} {scores}")
    return scores


def main():
    X_train, y_train, X_test = load_data()
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    X_train, X_test = neutralize_cell_outliers(X_train, X_test)
    X_train, X_test = impute_and_clean(X_train, X_test)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    is_outlier = detect_outlier_rows(X_train_scaled)
    X_clean = X_train_scaled[~is_outlier]
    y_clean = y_train[~is_outlier]

    evaluate_pipeline(X_clean, y_clean)

    selected_features = select_features(X_clean, y_clean)
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_clean[selected_features], y_clean)

    predictions = model.predict(X_test_scaled[selected_features])
    submission = pd.DataFrame({"id": X_test.index, "y": predictions})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")


if __name__ == "__main__":
    main()
