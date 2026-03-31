import importlib.util
import math
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
list_for_control_log_transformation = []

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent  # Good one/3. Regression
SPLIT_DIR = PARENT_DIR / "3.1_split_data"

TRAIN_YEARS = (1985, 2013)
VAL_YEARS = (2014, 2015)
TEST_YEARS = (2016, 2017)


CRISIS_COL = "GFDD.OI.19"  # 1 = crisis, 0 = no crisis
COUNTRY_COL = "Country Name"
TIME_COL = "Time"
COUNTRY_FIXED_EFFECTS: bool = False  # True: include country dummies
YEAR_FIXED_EFFECTS: bool = False  # True: include year dummies
GDP: bool = True  # True: include Read_GDP_growth as a FE proxy
WORLD_GDP: bool = False  # True: include World_GDP_growth from the split workbook's World GDP Growth sheet
TIME_trend: bool = False  # True: include a linear time trend based on year
Testing: bool = True  # True: also evaluate the fitted model on the test split
TIME_TREND_COL = "time_trend"


# Control variables to include (from Final data / 3.5_control_variables.py).
# Those in list_for_control_log_transformation are log-transformed; rest in levels.
CONTROL_COLS_FULL = [
    "GFDD.OI.06",
    "GFDD.OI.02",
    "GFDD.EI.01",
    "GFDD.EI.03",
    "GFDD.EI.04",
    "GFDD.SI.05",
    "GFDD.DI.02",
    "GFDD.DI.12",
]
CONTROL_COLS = [c for c in CONTROL_COLS_FULL if c not in list_for_control_log_transformation]

# Columns to lag (all main banking variables + controls).
COLS_TO_LAG = list_for_winsorization + CONTROL_COLS + list_for_control_log_transformation
LAG_PERIODS = 1


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standardization = _load_module("standardization", PARENT_DIR / "3.3_standardization.py")
class_weights = _load_module("class_weights", PARENT_DIR / "3.2_class_weights" / "3.2_class_weights.py")
fixed_effects = _load_module("fixed_effects", PARENT_DIR / "3.4_fixed_effects.py")
split_data_config = _load_module("split_data_config", SPLIT_DIR / "split_data.py")
# 3.5 at project root (parent of "Good one")
control_variables = _load_module("control_variables", PARENT_DIR.parent.parent / "3.5_control_variables.py")

TRAIN_PATH = SPLIT_DIR / split_data_config.period_filename("train", TRAIN_YEARS)
VAL_PATH = SPLIT_DIR / split_data_config.period_filename("validation", VAL_YEARS)
TEST_PATH = SPLIT_DIR / split_data_config.period_filename("test", TEST_YEARS)
TRAIN_LABEL = split_data_config.period_label("Train", TRAIN_YEARS)
VAL_LABEL = split_data_config.period_label("Validation", VAL_YEARS)
TEST_LABEL = split_data_config.period_label("Test", TEST_YEARS)


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


def _load_split_with_world_gdp(split_path: Path) -> pd.DataFrame:
    cleaned_df = pd.read_excel(split_path, sheet_name="Cleaned data")
    if not WORLD_GDP:
        return cleaned_df

    world_gdp_df = pd.read_excel(split_path, sheet_name="World GDP Growth")
    world_gdp_df = world_gdp_df[["Year", "World_GDP_growth"]].drop_duplicates(subset=["Year"])

    out = cleaned_df.copy()
    out[TIME_COL] = pd.to_numeric(out[TIME_COL], errors="coerce").astype("Int64")
    world_gdp_df["Year"] = pd.to_numeric(world_gdp_df["Year"], errors="coerce").astype("Int64")
    return out.merge(
        world_gdp_df,
        left_on=TIME_COL,
        right_on="Year",
        how="left",
        validate="m:1",
    ).drop(columns=["Year"])


def _add_time_trend(df: pd.DataFrame, *, base_year: float, time_col: str, trend_col: str) -> pd.DataFrame:
    out = df.copy()
    years = pd.to_numeric(out[time_col], errors="coerce").astype(float)
    out[trend_col] = years - float(base_year)
    return out


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


def _ridge_inference_table(
    model,
    params: np.ndarray,
    columns: pd.Index,
    alpha: np.ndarray,
) -> pd.DataFrame:
    params = np.asarray(params, dtype=float)
    observed_info = -np.asarray(model.hessian(params), dtype=float)
    penalized_info = observed_info + np.diag(np.asarray(alpha, dtype=float))
    cov = np.linalg.pinv(penalized_info)
    std_err = np.sqrt(np.clip(np.diag(cov), a_min=0.0, a_max=None))
    z_stats = np.divide(
        params,
        std_err,
        out=np.full(params.shape, np.nan, dtype=float),
        where=std_err > 0,
    )
    p_values = np.array(
        [
            math.erfc(abs(float(z)) / math.sqrt(2.0)) if np.isfinite(z) else np.nan
            for z in z_stats
        ],
        dtype=float,
    )
    return pd.DataFrame(
        {
            "coefficient": params,
            "std_err_approx": std_err,
            "z_approx": z_stats,
            "p_value_approx": p_values,
            "odds_ratio": np.exp(params),
        },
        index=columns,
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

    df_train = _load_split_with_world_gdp(TRAIN_PATH)
    df_val = _load_split_with_world_gdp(VAL_PATH)
    df_test = _load_split_with_world_gdp(TEST_PATH)

    # Ensure we load all columns needed for lags, controls, and identifiers.
    required = list(
        set(
            COLS_TO_LAG
            + [CRISIS_COL, COUNTRY_COL, TIME_COL]
            + (["Read_GDP_growth"] if GDP else [])
            + (["World_GDP_growth"] if WORLD_GDP else [])
        )
    )
    df_train = df_train[[c for c in required if c in df_train.columns]].copy()
    df_val = df_val[[c for c in required if c in df_val.columns]].copy()
    df_test = df_test[[c for c in required if c in df_test.columns]].copy()

    pre_lag_required_cols = [CRISIS_COL, COUNTRY_COL, TIME_COL]
    if GDP:
        pre_lag_required_cols.append("Read_GDP_growth")
    if WORLD_GDP:
        pre_lag_required_cols.append("World_GDP_growth")
    df_train = df_train.dropna(subset=[c for c in pre_lag_required_cols if c in df_train.columns]).copy()
    df_val = df_val.dropna(subset=[c for c in pre_lag_required_cols if c in df_val.columns]).copy()
    df_test = df_test.dropna(subset=[c for c in pre_lag_required_cols if c in df_test.columns]).copy()

    splits_for_lag_availability = [df_train, df_val] + ([df_test] if Testing else [])
    available_lag_input_cols = [
        c for c in COLS_TO_LAG if all(c in split_df.columns for split_df in splits_for_lag_availability)
    ]

    # Fit transforms on train only, apply to validation/test.
    winsor_params = _fit_winsor_params(df_train, [c for c in list_for_winsorization if c in df_train.columns])
    df_train = _apply_winsor_params(df_train, winsor_params)
    df_val = _apply_winsor_params(df_val, winsor_params)
    df_test = _apply_winsor_params(df_test, winsor_params)

    std_params = _fit_standard_params(df_train, [c for c in list_for_standardization if c in df_train.columns])
    df_train = _apply_standard_params(df_train, std_params)
    df_val = _apply_standard_params(df_val, std_params)
    df_test = _apply_standard_params(df_test, std_params)

    # Log-transform control variables (fit on train, apply same transform to val/test)
    log_vars = [c for c in list_for_control_log_transformation if c in df_train.columns]
    if log_vars:
        train_mins = {col: float(np.nanmin(df_train[col].values)) for col in log_vars}
        control_variables.log_transform_variables(df_train, log_vars, inplace=True)
        control_variables.log_transform_variables(df_val, log_vars, inplace=True, min_per_var=train_mins)
        control_variables.log_transform_variables(df_test, log_vars, inplace=True, min_per_var=train_mins)

    df_train_lag = fixed_effects.add_lagged_columns(
        df_train, available_lag_input_cols, LAG_PERIODS, COUNTRY_COL, TIME_COL
    )
    lagged_cols = fixed_effects.lagged_column_names(available_lag_input_cols, LAG_PERIODS)

    df_val_lag = fixed_effects.add_lagged_columns_with_context(
        df_history=df_train,
        df_target=df_val,
        columns_to_lag=available_lag_input_cols,
        lag_periods=LAG_PERIODS,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        strict_time_order=True,
    )

    df_hist_for_test = pd.concat([df_train, df_val], ignore_index=True)
    df_test_lag = fixed_effects.add_lagged_columns_with_context(
        df_history=df_hist_for_test,
        df_target=df_test,
        columns_to_lag=available_lag_input_cols,
        lag_periods=LAG_PERIODS,
        country_col=COUNTRY_COL,
        time_col=TIME_COL,
        strict_time_order=True,
    )

    if TIME_trend:
        base_year = float(pd.to_numeric(df_train[TIME_COL], errors="coerce").dropna().min())
        df_train_lag = _add_time_trend(df_train_lag, base_year=base_year, time_col=TIME_COL, trend_col=TIME_TREND_COL)
        df_val_lag = _add_time_trend(df_val_lag, base_year=base_year, time_col=TIME_COL, trend_col=TIME_TREND_COL)
        df_test_lag = _add_time_trend(df_test_lag, base_year=base_year, time_col=TIME_COL, trend_col=TIME_TREND_COL)

    post_lag_required_cols = [CRISIS_COL, COUNTRY_COL, TIME_COL] + lagged_cols
    if GDP:
        post_lag_required_cols.append("Read_GDP_growth")
    if WORLD_GDP:
        post_lag_required_cols.append("World_GDP_growth")
    if TIME_trend:
        post_lag_required_cols.append(TIME_TREND_COL)
    df_train_lag = df_train_lag.dropna(subset=[c for c in post_lag_required_cols if c in df_train_lag.columns])
    df_val_lag = df_val_lag.dropna(subset=[c for c in post_lag_required_cols if c in df_val_lag.columns])
    df_test_lag = df_test_lag.dropna(subset=[c for c in post_lag_required_cols if c in df_test_lag.columns])

    if df_val_lag.empty:
        raise ValueError(
            "Validation split is empty after filtering and lag construction. "
            "Check the split years and the required regression columns."
        )
    if Testing and df_test_lag.empty:
        raise ValueError(
            "Test split is empty after filtering and lag construction. "
            "Check the split years and the required regression columns."
        )

    y_train = np.asarray(df_train_lag[CRISIS_COL], dtype=np.int64)
    y_val = np.asarray(df_val_lag[CRISIS_COL], dtype=np.int64)
    y_test = np.asarray(df_test_lag[CRISIS_COL], dtype=np.int64)

    train_classes = np.unique(y_train).tolist()
    if len(train_classes) < 2:
        raise ValueError(
            f"Training set has only one class ({train_classes}) after filtering. "
            "Ridge logit cannot learn; you get train_logloss≈0 and AUC=0.5. "
            "Ensure the train split has both crisis and non-crisis rows: check that all COLS_TO_LAG "
            "(and control columns) exist in the split files and that dropna() is not removing all of one class. "
            "You may need to add the control columns to the split data or temporarily reduce CONTROL_COLS_FULL."
        )

    country_categories = (
        sorted(df_train[COUNTRY_COL].dropna().astype(str).unique().tolist())
        if COUNTRY_FIXED_EFFECTS
        else None
    )
    year_categories = (
        sorted(pd.to_numeric(df_train[TIME_COL], errors="coerce").dropna().astype(int).unique().tolist())
        if YEAR_FIXED_EFFECTS
        else None
    )
    design_feature_cols = (
        lagged_cols
        + ([TIME_TREND_COL] if TIME_trend else [])
        + (["Read_GDP_growth"] if GDP else [])
        + (["World_GDP_growth"] if WORLD_GDP else [])
    )

    X_train = fixed_effects.build_design_matrix(
        df_train_lag,
        design_feature_cols,
        country_col=COUNTRY_COL if COUNTRY_FIXED_EFFECTS else None,
        time_col=TIME_COL if YEAR_FIXED_EFFECTS else None,
        country_categories=country_categories,
        year_categories=year_categories,
    )
    X_val = fixed_effects.build_design_matrix(
        df_val_lag,
        design_feature_cols,
        country_col=COUNTRY_COL if COUNTRY_FIXED_EFFECTS else None,
        time_col=TIME_COL if YEAR_FIXED_EFFECTS else None,
        country_categories=country_categories,
        year_categories=year_categories,
    )
    X_test = fixed_effects.build_design_matrix(
        df_test_lag,
        design_feature_cols,
        country_col=COUNTRY_COL if COUNTRY_FIXED_EFFECTS else None,
        time_col=TIME_COL if YEAR_FIXED_EFFECTS else None,
        country_categories=country_categories,
        year_categories=year_categories,
    )

    X_train_const = sm.add_constant(X_train, has_constant="add").astype(float)
    X_val_const = _align_X(sm.add_constant(X_val, has_constant="add"), X_train_const.columns)
    X_test_const = _align_X(sm.add_constant(X_test, has_constant="add"), X_train_const.columns)

    class_weight_map = class_weights.compute_inverse_frequency_weights(df_train_lag, target_col=CRISIS_COL)
    for k in (0, 1):
        class_weight_map.setdefault(k, 1.0)
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
    alpha_best = _alpha_vector(
        X_train_const.columns,
        alpha_fe=alpha_fe_best,
        key_regressor_cols=lagged_cols,
    )

    print("Ridge-penalized logit")
    print(
        f"Country fixed effects: {COUNTRY_FIXED_EFFECTS} | "
        f"Year fixed effects: {YEAR_FIXED_EFFECTS} | "
        f"GDP: {GDP} | World GDP: {WORLD_GDP} | Time trend: {TIME_trend} | Testing: {Testing}"
    )
    print("Weights: inverse frequency (GFDD.OI.19)")
    print(f"Alpha grid (fixed effects + intercept only): {alpha_grid}")
    print("\nValidation-based alpha selection:")
    print(val_table.to_string(index=False))
    print(f"\nSelected alpha_fe: {alpha_fe_best:g}")

    params = pd.Series(res.params, index=X_train_const.columns)
    reg_df = _ridge_inference_table(
        res.model,
        res.params,
        X_train_const.columns,
        alpha_best,
    )
    print("\nCoefficients with approximate p-values:")
    print(reg_df.to_string())

    train_ll, train_auc = _metrics(y_train, res.predict(X_train_const), w_train)
    val_proba = res.predict(X_val_const)
    val_ll, val_auc = _metrics(y_val, val_proba, w_val)
    if Testing:
        test_proba = res.predict(X_test_const)
        test_ll, test_auc = _metrics(y_test, test_proba, w_test)
    print("\nPerformance:")
    print(f"{TRAIN_LABEL}: weighted log-loss={train_ll:.4f} | weighted AUC={train_auc:.4f}")
    print(f"{VAL_LABEL}: weighted log-loss={val_ll:.4f} | weighted AUC={val_auc:.4f}")
    if Testing:
        print(f"{TEST_LABEL}: weighted log-loss={test_ll:.4f} | weighted AUC={test_auc:.4f}")

    validation_pred_df = df_val_lag[[COUNTRY_COL, TIME_COL, CRISIS_COL]].copy()
    validation_pred_df = validation_pred_df.rename(columns={CRISIS_COL: "actual_crisis"})
    validation_pred_df["predicted_crisis_prob"] = np.asarray(val_proba, dtype=float)
    validation_pred_df["predicted_crisis"] = (np.asarray(val_proba, dtype=float) >= 0.5).astype(np.int64)
    if Testing:
        testing_pred_df = df_test_lag[[COUNTRY_COL, TIME_COL, CRISIS_COL]].copy()
        testing_pred_df = testing_pred_df.rename(columns={CRISIS_COL: "actual_crisis"})
        testing_pred_df["predicted_crisis_prob"] = np.asarray(test_proba, dtype=float)
        testing_pred_df["predicted_crisis"] = (np.asarray(test_proba, dtype=float) >= 0.5).astype(np.int64)

    perf_df = pd.DataFrame([
        {"Split": TRAIN_LABEL, "weighted_log_loss": train_ll, "weighted_AUC": train_auc},
        {"Split": VAL_LABEL, "weighted_log_loss": val_ll, "weighted_AUC": val_auc},
    ])
    if Testing:
        perf_df = pd.concat(
            [
                perf_df,
                pd.DataFrame(
                    [{"Split": TEST_LABEL, "weighted_log_loss": test_ll, "weighted_AUC": test_auc}]
                ),
            ],
            ignore_index=True,
        )
    n_controls = len(CONTROL_COLS_FULL)
    n_logged = len(list_for_control_log_transformation)
    std_part = "_".join(list_for_winsorization)
    script_stem = Path(__file__).stem
    control_prefix = f"Control_{n_logged}_{n_controls}_"
    perf_path = SCRIPT_DIR / f"{control_prefix}{std_part}_{script_stem}.xlsx"
    with pd.ExcelWriter(perf_path, engine="openpyxl") as writer:
        perf_df.to_excel(writer, sheet_name="Performance", index=False)
        reg_df.to_excel(writer, sheet_name="Regression")
        val_table.to_excel(writer, sheet_name="Alpha_selection", index=False)
        validation_pred_df.to_excel(writer, sheet_name="Validation_predictions", index=False)
        if Testing:
            testing_pred_df.to_excel(writer, sheet_name="Testing_predictions", index=False)
    print(f"\nPerformance saved: {perf_path}")

    for col in lagged_cols:
        if col in params.index:
            beta = float(params[col])
            print(f"\n{col}: log-odds change = {beta:.4f}, odds ratio = {np.exp(beta):.4f}")


if __name__ == "__main__":
    main()
