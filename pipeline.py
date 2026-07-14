"""
Brain Age Prediction pipeline — shared building blocks.

Every step that touches training-set statistics (MAD outlier bounds, the
median imputer, the constant-column mask, the outlier-row detector, feature
selection, and the model itself) is fit *only* on whatever data is passed in
as "training" data. `run_pipeline` applies exactly these steps in exactly
this order whether it's called from inside a CV fold or on the full training
set, so cross-validation and the final fit are the same pipeline — no leakage
between fitting and evaluation.

Steps (matching the project's subtasks):
  0. Impute missing values (median, fit on train only)
  1. Detect & remove outlier samples (rows) via IsolationForest (train only)
  2. Select relevant features, drop irrelevant/redundant ones (train only)
  3. Fit a regression model (XGBoost) to predict age
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

RANDOM_STATE = 0
CELL_OUTLIER_Z_THRESH = 15  # robust (MAD-based) z-score threshold for single-cell corruption
ES_FRACTION = 0.1  # fraction of the (cleaned) training fold held out for early stopping
EARLY_STOPPING_ROUNDS = 30


def load_data(data_dir="data"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").set_index("id")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").set_index("id")["y"]
    X_test = pd.read_csv(f"{data_dir}/X_test.csv").set_index("id")
    return X_train, y_train, X_test


# ---------------------------------------------------------------------------
# Preprocessing: cell-outlier neutralization + median imputation + constant
# column removal, all fit on a training set and reusable on any other set.
# ---------------------------------------------------------------------------


def fit_preprocessing(X_train_raw, z_thresh=CELL_OUTLIER_Z_THRESH, near_constant_std=None):
    """Fit MAD-based cell-outlier bounds, a median imputer, and the
    constant/near-constant column mask on a training set only. Returns a
    state dict usable by `apply_preprocessing` on any other set (validation
    fold, test set), plus the transformed training data itself."""
    med = X_train_raw.median()
    mad = (X_train_raw - med).abs().median().replace(0, np.nan)

    def neutralize(df):
        modz = 0.6745 * (df - med) / mad
        clean = df.copy()
        clean[modz.abs() > z_thresh] = np.nan
        return clean

    X_neut = neutralize(X_train_raw)
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_neut), index=X_train_raw.index, columns=X_train_raw.columns
    )

    keep = X_imp.nunique() > 1
    if near_constant_std is not None:
        keep &= X_imp.std() > near_constant_std
    keep_cols = keep[keep].index

    state = {"med": med, "mad": mad, "z_thresh": z_thresh, "imputer": imputer, "keep_cols": keep_cols}
    return state, X_imp[keep_cols]


def apply_preprocessing(X_raw, state):
    """Apply a previously-fit preprocessing state (see `fit_preprocessing`)
    to a new set of rows (validation fold or test set)."""
    modz = 0.6745 * (X_raw - state["med"]) / state["mad"]
    clean = X_raw.copy()
    clean[modz.abs() > state["z_thresh"]] = np.nan
    X_imp = pd.DataFrame(
        state["imputer"].transform(clean), index=X_raw.index, columns=X_raw.columns
    )
    return X_imp[state["keep_cols"]]


# ---------------------------------------------------------------------------
# Outlier row removal (Subtask 1). RobustScaler is used only to give
# IsolationForest scale-free distances; XGBoost never sees the scaled copy.
# ---------------------------------------------------------------------------


def remove_outlier_rows(X_train_imp, y_train, contamination):
    """Fit IsolationForest on (scaled) training-fold data only and drop the
    rows it flags. contamination=0 disables outlier removal entirely."""
    if not contamination:
        return X_train_imp, y_train

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train_imp)
    detector = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1
    )
    is_outlier = detector.fit_predict(X_scaled) == -1
    return X_train_imp[~is_outlier], y_train[~is_outlier]


# ---------------------------------------------------------------------------
# Feature selection (Subtask 2). Modular so alternative rankers can be
# swapped in and compared under the same CV harness.
# ---------------------------------------------------------------------------


def _mi_ranking(X, y):
    scores = mutual_info_regression(X, y, random_state=RANDOM_STATE)
    return pd.Series(scores, index=X.columns)


def _xgb_importance_ranking(X, y):
    """Gain-based importance from a quick XGBoost fit on all features."""
    booster = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    booster.fit(X, y)
    return pd.Series(booster.feature_importances_, index=X.columns)


def _permutation_ranking(X, y):
    """Permutation importance from a quick XGBoost fit. Roughly 30-50x
    slower than MI or gain importance on this feature count (~800), since it
    requires one re-prediction pass per feature per repeat — kept available
    for completeness but not used in the full grid search (see tune.py)."""
    from sklearn.inspection import permutation_importance

    booster = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    booster.fit(X, y)
    result = permutation_importance(
        booster, X, y, n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X.columns)


FEATURE_SELECTION_METHODS = {
    "mi": _mi_ranking,
    "xgb_importance": _xgb_importance_ranking,
    "permutation": _permutation_ranking,
}


def select_features(X, y, method, n_features):
    """Rank features on training data only and keep the top n_features."""
    if n_features >= X.shape[1]:
        return X.columns
    ranking = FEATURE_SELECTION_METHODS[method](X, y).sort_values(ascending=False)
    return ranking.head(n_features).index


# ---------------------------------------------------------------------------
# Model fitting with early stopping. The early-stopping holdout is carved out
# of the (already-cleaned) training fold only, so it never touches the outer
# validation fold / test set.
# ---------------------------------------------------------------------------


def fit_xgb(X_train, y_train, xgb_params, es_fraction=ES_FRACTION, early_stopping_rounds=EARLY_STOPPING_ROUNDS):
    if not early_stopping_rounds:
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, y_train)
        return model

    X_fit, X_es, y_fit, y_es = train_test_split(
        X_train, y_train, test_size=es_fraction, random_state=RANDOM_STATE
    )
    model = xgb.XGBRegressor(
        **xgb_params, early_stopping_rounds=early_stopping_rounds, eval_metric="mae"
    )
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Stacking ensemble: several diverse model types blended by a meta-model.
# Out-of-fold (OOF) predictions used to fit the meta-model come from an inner
# K-fold CV run entirely inside the training data handed to `fit` -- the
# outer validation fold / test set is only ever touched by `predict`, on base
# models that were refit on the *full* training data. Same no-leakage
# contract as the rest of the pipeline, just nested one level deeper.
# ---------------------------------------------------------------------------


def build_base_models():
    """Diverse model families so the meta-model has genuinely different
    errors to blend: two XGBoost configs (shallow/heavily-regularized vs.
    deeper/less-regularized), a linear model (ElasticNet, scaled internally
    via a Pipeline so no leakage across the models that reuse this training
    data), extremely randomized trees (bagging, as opposed to boosting), and
    sklearn's own gradient boosting (a different boosting implementation).

    ElasticNet scores notably lower solo (~0.39 R^2 vs. ~0.51 for the
    XGBoost configs) but still pulls real weight in the Ridge meta-model
    (see eval_ensemble.py diagnostics) -- its errors are decorrelated enough
    from the trees' to be worth blending in anyway. An RBF-kernel SVR was
    also tried here and left the ensemble's CV score statistically
    unchanged (0.5149 vs. 0.5148) for ~90s of extra fit time per fold, so it
    was dropped again rather than kept as dead weight."""
    return {
        "xgb_main": lambda: xgb.XGBRegressor(
            n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.4, reg_lambda=5.0, min_child_weight=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgb_diverse": lambda: xgb.XGBRegressor(
            n_estimators=800, max_depth=6, learning_rate=0.02, subsample=0.9,
            colsample_bytree=0.3, reg_lambda=2.0, min_child_weight=3,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "elasticnet": lambda: make_pipeline(
            RobustScaler(), ElasticNet(alpha=0.5, l1_ratio=0.3, random_state=RANDOM_STATE, max_iter=20000)
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gb": lambda: HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }


def build_meta_model():
    """Non-negative least squares blends the base models' predictions --
    tried against Ridge (alpha 0.1-10, all statistically identical) and a
    plain average; NNLS won on R^2/MAE/RMSE (0.5168 vs. 0.5148 vs. 0.5107
    R^2). The problem is tiny (a handful of input columns) so overfitting
    risk is low; the positive-only constraint just rules out a base model's
    weight flipping sign on noise, which a plain average can't do."""
    return LinearRegression(positive=True)


class StackingEnsemble:
    """Fits `base_model_builders` via inner K-fold CV to get leak-free OOF
    predictions, trains `meta_model_builder` on those, then refits every
    base model on the full training data for use at `predict` time."""

    def __init__(self, base_model_builders, meta_model_builder, n_inner_folds=5, random_state=RANDOM_STATE):
        self.base_model_builders = base_model_builders
        self.meta_model_builder = meta_model_builder
        self.n_inner_folds = n_inner_folds
        self.random_state = random_state

    def fit(self, X, y):
        names = list(self.base_model_builders.keys())
        oof = np.zeros((len(X), len(names)))
        inner_kf = KFold(n_splits=self.n_inner_folds, shuffle=True, random_state=self.random_state)
        for train_idx, val_idx in inner_kf.split(X):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            for j, name in enumerate(names):
                model = self.base_model_builders[name]()
                model.fit(X_tr, y_tr)
                oof[val_idx, j] = model.predict(X_va)

        self.meta_model_ = self.meta_model_builder()
        self.meta_model_.fit(oof, y)

        self.base_models_ = {}
        for name in names:
            model = self.base_model_builders[name]()
            model.fit(X, y)
            self.base_models_[name] = model
        self.names_ = names
        return self

    def predict(self, X):
        base_preds = np.column_stack([self.base_models_[name].predict(X) for name in self.names_])
        return self.meta_model_.predict(base_preds)


# ---------------------------------------------------------------------------
# Model-fitting dispatcher: "xgb" (default) fits a single tuned XGBRegressor
# with early stopping; "stack" fits the StackingEnsemble above. Both expose
# the same fit/predict interface so `run_pipeline` doesn't care which is used.
# ---------------------------------------------------------------------------


def fit_model(X_train, y_train, config):
    model_type = config.get("model_type", "xgb")
    if model_type == "xgb":
        return fit_xgb(
            X_train,
            y_train,
            config["xgb_params"],
            es_fraction=config.get("es_fraction", ES_FRACTION),
            early_stopping_rounds=config.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS),
        )
    if model_type == "stack":
        base_models = config.get("base_models")
        if base_models is None:
            base_models = build_base_models()
        ensemble = StackingEnsemble(
            base_models,
            config.get("meta_model", build_meta_model),
            n_inner_folds=config.get("n_inner_folds", 5),
        )
        ensemble.fit(X_train, y_train)
        return ensemble
    raise ValueError(f"Unknown model_type: {model_type!r}")


# ---------------------------------------------------------------------------
# The single pipeline used by both cross-validation and the final fit.
# ---------------------------------------------------------------------------


def run_pipeline(X_train_raw, y_train, X_holdout_raw, config):
    """Fit the full pipeline (preprocessing -> outlier removal -> feature
    selection -> model) on X_train_raw/y_train only, then prepare
    X_holdout_raw (a CV validation fold or the real test set) with the same
    fitted transforms. Returns (model, X_holdout_selected, selected_columns).
    """
    state, X_tr_imp = fit_preprocessing(
        X_train_raw, near_constant_std=config.get("near_constant_std")
    )
    X_hold_imp = apply_preprocessing(X_holdout_raw, state)

    X_tr_clean, y_tr_clean = remove_outlier_rows(X_tr_imp, y_train, config["contamination"])

    selected = select_features(
        X_tr_clean, y_tr_clean, method=config["fs_method"], n_features=config["n_features"]
    )

    model = fit_model(X_tr_clean[selected], y_tr_clean, config)

    return model, X_hold_imp[selected], selected
