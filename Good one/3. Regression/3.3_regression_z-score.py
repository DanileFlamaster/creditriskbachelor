import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score

# Columns that will be winsorized and standardized
list_for_winsorization = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]  # Credit to Deposit, NPL, Z-Score
list_for_standardization = ["GFDD.SI.01"]  # Z-Score

SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / "3.1_split_data"
TRAIN_PATH = SPLIT_DIR / "train_1985_2007.xlsx"
VAL_PATH = SPLIT_DIR / "validation_2008_2010.xlsx"
TEST_PATH = SPLIT_DIR / "test_2011_2021.xlsx"

CRISIS_COL = "GFDD.OI.19"
ZSCORE_COL = "GFDD.SI.01"
COUNTRY_COL = "Country Name"
TIME_COL = "Time"
COLS_TO_LAG = [ZSCORE_COL]
LAG_PERIODS = 1


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standardization = _load_module("standardization", SCRIPT_DIR / "3.3_standardization.py")
class_weights = _load_module("class_weights", SCRIPT_DIR / "3.2_class_weights.py")
fixed_effects = _load_module("fixed_effects", SCRIPT_DIR / "3.4_fixed_effects.py")


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


def _evaluate_split(name: str, y: np.ndarray, proba: np.ndarray, weights: np.ndarray):
    p = _safe_proba(proba)
    ll = log_loss(y, p, sample_weight=weights, labels=[0, 1])
    try:
        auc = roc_auc_score(y, p, sample_weight=weights)
    except Exception:
        auc = float("nan")
    print(f"{name}: weighted log-loss={ll:.4f} | weighted AUC={auc:.4f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    df_train = pd.read_excel(TRAIN_PATH)
    df_val = pd.read_excel(VAL_PATH)
    df_test = pd.read_excel(TEST_PATH)

    required = list(set(COLS_TO_LAG + [CRISIS_COL, COUNTRY_COL, TIME_COL]))
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

    model = sm.GLM(
        y_train,
        X_train_const,
        family=sm.families.Binomial(),
        freq_weights=w_train,
    )
    result = model.fit(maxiter=2000, disp=0)

    print("Logit (lagged X + country & year fixed effects), class weights from inverse frequency")
    print("=" * 60)
    print(result.summary())
    print("\nOdds ratios (exp(coef)):")
    params = pd.Series(result.params, index=X_train_const.columns)
    print(np.exp(params))

    p_train = result.predict(X_train_const)
    p_val = result.predict(X_val_const)
    #p_test = result.predict(X_test_const)

    print("\nPerformance:")
    _evaluate_split("Train (1985-2007)", y_train, p_train, w_train)
    _evaluate_split("Validation (2008-2010)", y_val, p_val, w_val)
    #_evaluate_split("Test (2011-2021)", y_test, p_test, w_test)


if __name__ == "__main__":
    main()
