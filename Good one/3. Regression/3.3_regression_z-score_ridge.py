import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Columns that will be winsorized and standardized
list_for_winsorization = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]  # Credit to Deposit, NPL, Z-Score
list_for_standardization = ["GFDD.SI.01"]  # Z-Score

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / "3.1_split_data"
TRAIN_PATH = SPLIT_DIR / "train_1985_2007.xlsx"

CRISIS_COL = "GFDD.OI.19"  # 1 = crisis, 0 = no crisis
ZSCORE_COL = "GFDD.SI.01"  # Z-Score (standardized)
COUNTRY_COL = "Country Name"
TIME_COL = "Time"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standardization = _load_module("standardization", SCRIPT_DIR / "3.3_standardization.py")
class_weights = _load_module("class_weights", SCRIPT_DIR / "3.2_class_weights.py")
fixed_effects = _load_module("fixed_effects", SCRIPT_DIR / "3.4_fixed_effects.py")


def _alpha_vector(columns: pd.Index, alpha_fe: float) -> np.ndarray:
    """
    Ridge penalty weights for GLM.fit_regularized.

    We do NOT penalize the key regressor (Z-Score) so the coefficient remains
    as comparable as possible across specifications. We DO penalize the fixed
    effects and intercept, which is the part that tends to blow up under
    (quasi-)separation.
    """
    alpha = np.full(len(columns), float(alpha_fe), dtype=float)
    if ZSCORE_COL in columns:
        alpha[columns.get_loc(ZSCORE_COL)] = 0.0
    return alpha


def _fit_ridge_glm(
    y: np.ndarray,
    X: pd.DataFrame,
    *,
    obs_weights: np.ndarray,
    alpha_fe: float,
):
    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=obs_weights,
    )
    alpha = _alpha_vector(X.columns, alpha_fe=alpha_fe)
    return model.fit_regularized(
        method="elastic_net",
        L1_wt=0.0,  # pure ridge
        alpha=alpha,
        maxiter=2000,
    )


def select_alpha_via_cv(
    y: np.ndarray,
    X: pd.DataFrame,
    *,
    obs_weights: np.ndarray,
    alpha_grid: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for alpha_fe in alpha_grid:
        fold_aucs = []
        fold_logloss = []
        for train_idx, test_idx in cv.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            w_train = obs_weights[train_idx]

            X_test = X.iloc[test_idx]
            y_test = y[test_idx]
            w_test = obs_weights[test_idx]

            try:
                res = _fit_ridge_glm(y_train, X_train, obs_weights=w_train, alpha_fe=float(alpha_fe))
            except Exception:
                continue

            proba = res.predict(X_test)
            try:
                auc = roc_auc_score(y_test, proba, sample_weight=w_test)
            except Exception:
                continue
            fold_aucs.append(float(auc))

            try:
                ll = log_loss(y_test, proba, sample_weight=w_test, labels=[0, 1])
            except Exception:
                continue
            fold_logloss.append(float(ll))

        if len(fold_aucs) == 0:
            rows.append(
                {
                    "alpha_fe": float(alpha_fe),
                    "folds": 0,
                    "auc_mean": np.nan,
                    "auc_std": np.nan,
                    "logloss_mean": np.nan,
                    "logloss_std": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "alpha_fe": float(alpha_fe),
                    "folds": int(len(fold_aucs)),
                    "auc_mean": float(np.mean(fold_aucs)),
                    "auc_std": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0,
                    "logloss_mean": float(np.mean(fold_logloss)) if len(fold_logloss) else np.nan,
                    "logloss_std": float(np.std(fold_logloss, ddof=1)) if len(fold_logloss) > 1 else 0.0,
                }
            )

    out = pd.DataFrame(rows)
    # Prefer calibration/stability under separation: lowest weighted log-loss.
    # Break ties by higher AUC, then weaker penalty (smaller alpha).
    out = out.sort_values(
        ["logloss_mean", "auc_mean", "folds", "alpha_fe"],
        ascending=[True, False, False, True],
    )
    return out


def main():
    # Avoid UnicodeEncodeError on Windows consoles with non-UTF8 code pages (e.g. "Curaçao").
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df_train = pd.read_excel(TRAIN_PATH)
    df_train = standardization.winsorize(df_train, list_for_winsorization)
    df_train = standardization.standardize(df_train, list_for_standardization)

    required = [ZSCORE_COL, CRISIS_COL]
    if COUNTRY_COL in df_train.columns:
        required.append(COUNTRY_COL)
    if TIME_COL in df_train.columns:
        required.append(TIME_COL)
    df_train = df_train[[c for c in required if c in df_train.columns]].dropna()

    y = np.asarray(df_train[CRISIS_COL], dtype=np.int64)

    X = fixed_effects.build_design_matrix(
        df_train,
        [ZSCORE_COL],
        country_col=COUNTRY_COL if COUNTRY_COL in df_train.columns else None,
        time_col=TIME_COL if TIME_COL in df_train.columns else None,
    )
    X_const = sm.add_constant(X, has_constant="add").astype(float)

    weights = class_weights.compute_inverse_frequency_weights(df_train, target_col=CRISIS_COL)
    obs_weights = np.array([weights[int(yi)] for yi in y], dtype=float)

    alpha_grid = np.logspace(-4, 1, 12)
    cv_table = select_alpha_via_cv(
        y,
        X_const,
        obs_weights=obs_weights,
        alpha_grid=alpha_grid,
        n_splits=5,
        random_state=42,
    )

    best_row = cv_table.dropna(subset=["logloss_mean"]).iloc[0]
    alpha_fe_best = float(best_row["alpha_fe"])

    res = _fit_ridge_glm(y, X_const, obs_weights=obs_weights, alpha_fe=alpha_fe_best)

    print("Ridge-penalized logit (Z-Score + country & year fixed effects)")
    print("Weights: inverse frequency (GFDD.OI.19)")
    print(f"Alpha grid (fixed effects + intercept only): {alpha_grid}")
    print("\nCross-validated alpha selection (weighted AUC):")
    print(cv_table.to_string(index=False))
    print(f"\nSelected alpha_fe: {alpha_fe_best:g}")

    params = pd.Series(res.params, index=X_const.columns)
    print("\nCoefficients:")
    print(params.to_string())

    print("\nOdds ratios (exp(coef)):")
    print(np.exp(params).to_string())

    y_pred_proba = res.predict(X_const)
    auc_train = roc_auc_score(y, y_pred_proba, sample_weight=obs_weights)
    print(f"\nROC AUC (train, weighted): {auc_train:.4f}")

    if ZSCORE_COL in params.index:
        beta = float(params[ZSCORE_COL])
        print(f"\nEconomic interpretation unit: 1 SD increase in Z-Score (standardized)")
        print(f"Log-odds change: {beta:.4f}")
        print(f"Odds ratio: {np.exp(beta):.4f}")


if __name__ == "__main__":
    main()
