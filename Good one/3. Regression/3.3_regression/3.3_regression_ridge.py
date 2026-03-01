import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score

# Columns that will be winsorized and standardized
list_for_winsorization = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]  # Credit to Deposit, NPL, Z-Score
list_for_standardization = ["GFDD.SI.01"]  # Z-Score

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # Good one/3. Regression
SPLIT_DIR = PARENT_DIR / "3.1_split_data"
TRAIN_PATH = SPLIT_DIR / "train_1985_2007.xlsx"
VAL_PATH = SPLIT_DIR / "validation_2008_2010.xlsx"
TEST_PATH = SPLIT_DIR / "test_2011_2021.xlsx"


CRISIS_COL = "GFDD.OI.19"  # 1 = crisis, 0 = no crisis
COUNTRY_COL = "Country Name"
TIME_COL = "Time"

# Control variables to include (from Final data / 3.5_control_variables.py).
# Adjust this list to your final choice of controls.
CONTROL_COLS = []

# Columns to lag (all main banking variables + controls).
COLS_TO_LAG = list_for_winsorization + CONTROL_COLS
LAG_PERIODS = 1


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standardization = _load_module("standardization", PARENT_DIR / "3.3_standardization.py")
class_weights = _load_module("class_weights", PARENT_DIR / "3.2_class_weights" / "3.2_class_weights.py")
fixed_effects = _load_module("fixed_effects", PARENT_DIR / "3.4_fixed_effects.py")


def _fit_winsor_params(df: pd.DataFrame, columns: list[str], lower: float = 0.01, upper: float = 0.99) -> dict:
    params = {}
    for col in columns:
        if col in df.columns:
            params[col] = (float(df[col].quantile(lower)), float(df[col].quantile(upper)))
    return params


def _apply_winsor_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    for col, (q_low, q_high) in params.items():
        if col in out.columns:
            out[col] = out[col].clip(lower=q_low, upper=q_high)
    return out


def _fit_standard_params(df: pd.DataFrame, columns: list[str]) -> dict:
    params = {}
    for col in columns:
        if col in df.columns:
            params[col] = (float(df[col].mean()), float(df[col].std()))
    return params


def _apply_standard_params(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    for col, (mean, std) in params.items():
        if col not in out.columns:
            continue
        if std == 0 or np.isnan(std):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std
    return out


def _align_X(X: pd.DataFrame, columns_ref: pd.Index) -> pd.DataFrame:
    return X.reindex(columns=columns_ref, fill_value=0.0).astype(float)


def _safe_proba(proba: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)


def _metrics(y: np.ndarray, proba: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    p = _safe_proba(proba)
    ll = log_loss(y, p, sample_weight=weights, labels=[0, 1])
    try:
        auc = roc_auc_score(y, p, sample_weight=weights)
    except Exception:
        auc = float("nan")
    return float(ll), float(auc)


def _alpha_vector(
    columns: pd.Index,
    alpha_fe: float,
    key_regressor_cols: Optional[List[str]] = None,
) -> np.ndarray:
    key_regressor_cols = key_regressor_cols or []
    alpha = np.full(len(columns), float(alpha_fe), dtype=float)
    for col in key_regressor_cols:
        if col in columns:
            alpha[columns.get_loc(col)] = 0.0
    return alpha


def _fit_ridge_glm(
    y: np.ndarray,
    X: pd.DataFrame,
    *,
    obs_weights: np.ndarray,
    alpha_fe: float,
    key_regressor_cols: Optional[List[str]] = None,
):
    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=obs_weights,
    )
    alpha = _alpha_vector(
        X.columns,
        alpha_fe=alpha_fe,
        key_regressor_cols=key_regressor_cols,
    )
    return model.fit_regularized(
        method="elastic_net",
        L1_wt=0.0,  # pure ridge
        alpha=alpha,
        maxiter=2000,
    )


def _select_alpha_on_validation(
    y_train: np.ndarray,
    X_train: pd.DataFrame,
    w_train: np.ndarray,
    y_val: np.ndarray,
    X_val: pd.DataFrame,
    w_val: np.ndarray,
    alpha_grid: np.ndarray,
    key_regressor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    for alpha_fe in alpha_grid:
        try:
            res = _fit_ridge_glm(
                y_train,
                X_train,
                obs_weights=w_train,
                alpha_fe=float(alpha_fe),
                key_regressor_cols=key_regressor_cols,
            )
            train_ll, train_auc = _metrics(y_train, res.predict(X_train), w_train)
            val_ll, val_auc = _metrics(y_val, res.predict(X_val), w_val)
            rows.append(
                {
                    "alpha_fe": float(alpha_fe),
                    "train_logloss": train_ll,
                    "train_auc": train_auc,
                    "val_logloss": val_ll,
                    "val_auc": val_auc,
                }
            )
        except Exception:
            rows.append(
                {
                    "alpha_fe": float(alpha_fe),
                    "train_logloss": np.nan,
                    "train_auc": np.nan,
                    "val_logloss": np.nan,
                    "val_auc": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values(["val_logloss", "val_auc", "alpha_fe"], ascending=[True, False, True])
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df_train = pd.read_excel(TRAIN_PATH)
    df_val = pd.read_excel(VAL_PATH)
    df_test = pd.read_excel(TEST_PATH)

    # Ensure we load all columns needed for lags, controls, and identifiers.
    required = list(set(COLS_TO_LAG + CONTROL_COLS + [CRISIS_COL, COUNTRY_COL, TIME_COL]))
    df_train = df_train[[c for c in required if c in df_train.columns]].dropna().copy()
    df_val = df_val[[c for c in required if c in df_val.columns]].dropna().copy()
    df_test = df_test[[c for c in required if c in df_test.columns]].dropna().copy()

    # Fit transforms on train only, apply to validation/test.
    winsor_params = _fit_winsor_params(df_train, [c for c in list_for_winsorization if c in df_train.columns])
    df_train = _apply_winsor_params(df_train, winsor_params)
    df_val = _apply_winsor_params(df_val, winsor_params)
    df_test = _apply_winsor_params(df_test, winsor_params)

    std_params = _fit_standard_params(df_train, [c for c in list_for_standardization if c in df_train.columns])
    df_train = _apply_standard_params(df_train, std_params)
    df_val = _apply_standard_params(df_val, std_params)
    df_test = _apply_standard_params(df_test, std_params)

    df_train_lag = fixed_effects.add_lagged_columns(
        df_train, COLS_TO_LAG, LAG_PERIODS, COUNTRY_COL, TIME_COL
    )
    lagged_cols = fixed_effects.lagged_column_names(COLS_TO_LAG, LAG_PERIODS)
    df_train_lag = df_train_lag.dropna(subset=lagged_cols)

    df_val_lag = fixed_effects.add_lagged_columns_with_context(
        df_history=df_train,
        df_target=df_val,
        columns_to_lag=COLS_TO_LAG,
        lag_periods=LAG_PERIODS,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        strict_time_order=True,
    ).dropna(subset=lagged_cols)

    df_hist_for_test = pd.concat([df_train, df_val], ignore_index=True)
    df_test_lag = fixed_effects.add_lagged_columns_with_context(
        df_history=df_hist_for_test,
        df_target=df_test,
        columns_to_lag=COLS_TO_LAG,
        lag_periods=LAG_PERIODS,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        strict_time_order=True,
    ).dropna(subset=lagged_cols)

    y_train = np.asarray(df_train_lag[CRISIS_COL], dtype=np.int64)
    y_val = np.asarray(df_val_lag[CRISIS_COL], dtype=np.int64)
    y_test = np.asarray(df_test_lag[CRISIS_COL], dtype=np.int64)

    country_categories = sorted(df_train[COUNTRY_COL].dropna().astype(str).unique().tolist())
    year_categories = sorted(pd.to_numeric(df_train[TIME_COL], errors="coerce").dropna().astype(int).unique().tolist())

    X_train = fixed_effects.build_design_matrix(
        df_train_lag,
        lagged_cols,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        country_categories=country_categories,
        year_categories=year_categories,
    )
    X_val = fixed_effects.build_design_matrix(
        df_val_lag,
        lagged_cols,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        country_categories=country_categories,
        year_categories=year_categories,
    )
    X_test = fixed_effects.build_design_matrix(
        df_test_lag,
        lagged_cols,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        country_categories=country_categories,
        year_categories=year_categories,
    )

    X_train_const = sm.add_constant(X_train, has_constant="add").astype(float)
    X_val_const = _align_X(sm.add_constant(X_val, has_constant="add"), X_train_const.columns)
    X_test_const = _align_X(sm.add_constant(X_test, has_constant="add"), X_train_const.columns)

    class_weight_map = class_weights.compute_inverse_frequency_weights(df_train_lag, target_col=CRISIS_COL)
    w_train = np.array([class_weight_map[int(v)] for v in y_train], dtype=float)
    w_val = np.array([class_weight_map[int(v)] for v in y_val], dtype=float)
    w_test = np.array([class_weight_map[int(v)] for v in y_test], dtype=float)

    alpha_grid = np.logspace(-4, 1, 12)
    val_table = _select_alpha_on_validation(
        y_train=y_train,
        X_train=X_train_const,
        w_train=w_train,
        y_val=y_val,
        X_val=X_val_const,
        w_val=w_val,
        alpha_grid=alpha_grid,
        key_regressor_cols=lagged_cols,
    )
    best_row = val_table.dropna(subset=["val_logloss"]).iloc[0]
    alpha_fe_best = float(best_row["alpha_fe"])

    res = _fit_ridge_glm(
        y_train,
        X_train_const,
        obs_weights=w_train,
        alpha_fe=alpha_fe_best,
        key_regressor_cols=lagged_cols,
    )

    print("Ridge-penalized logit (lagged X + country & year fixed effects)")
    print("Weights: inverse frequency (GFDD.OI.19)")
    print(f"Alpha grid (fixed effects + intercept only): {alpha_grid}")
    print("\nValidation-based alpha selection:")
    print(val_table.to_string(index=False))
    print(f"\nSelected alpha_fe: {alpha_fe_best:g}")

    params = pd.Series(res.params, index=X_train_const.columns)
    print("\nCoefficients:")
    print(params.to_string())
    print("\nOdds ratios (exp(coef)):")
    print(np.exp(params).to_string())

    train_ll, train_auc = _metrics(y_train, res.predict(X_train_const), w_train)
    val_ll, val_auc = _metrics(y_val, res.predict(X_val_const), w_val)
    # test_ll, test_auc = _metrics(y_test, res.predict(X_test_const), w_test)
    print("\nPerformance:")
    print(f"Train (1985-2007): weighted log-loss={train_ll:.4f} | weighted AUC={train_auc:.4f}")
    print(f"Validation (2008-2010): weighted log-loss={val_ll:.4f} | weighted AUC={val_auc:.4f}")
    # print(f"Test (2011-2021): weighted log-loss={test_ll:.4f} | weighted AUC={test_auc:.4f}")

    perf_df = pd.DataFrame([
        {"Split": "Train (1985-2007)", "weighted_log_loss": train_ll, "weighted_AUC": train_auc},
        {"Split": "Validation (2008-2010)", "weighted_log_loss": val_ll, "weighted_AUC": val_auc},
    ])
    reg_df = pd.DataFrame({
        "coefficient": params,
        "odds_ratio": np.exp(params),
    })
    std_part = "_".join(list_for_winsorization)
    script_stem = Path(__file__).stem
    perf_path = SCRIPT_DIR / f"{std_part}_{script_stem}.xlsx"
    with pd.ExcelWriter(perf_path, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="Performance", index=False)
        reg_df.to_excel(writer, sheet_name="Regression")
        val_table.to_excel(writer, sheet_name="Alpha_selection", index=False)
    print(f"\nPerformance saved: {perf_path}")

    for col in lagged_cols:
        if col in params.index:
            beta = float(params[col])
            print(f"\n{col}: log-odds change = {beta:.4f}, odds ratio = {np.exp(beta):.4f}")


if __name__ == "__main__":
    main()
