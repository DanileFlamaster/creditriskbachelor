import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
# list_for_winsorization and list_for_standardization are the columns that will be winsorized and standardized   
list_for_winsorization = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"] # Credit to Deposit, NPL, Z-Score
list_for_standardization = ["GFDD.SI.01"] # Z-Score

SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "standardization", SCRIPT_DIR / "3.3_standardization.py"
)
standardization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(standardization)

_spec_cw = importlib.util.spec_from_file_location(
    "class_weights", SCRIPT_DIR / "3.2_class_weights.py"
)
class_weights = importlib.util.module_from_spec(_spec_cw)
_spec_cw.loader.exec_module(class_weights)

_spec_fe = importlib.util.spec_from_file_location(
    "fixed_effects", SCRIPT_DIR / "3.4_fixed_effects.py"
)
fixed_effects = importlib.util.module_from_spec(_spec_fe)
_spec_fe.loader.exec_module(fixed_effects)

SPLIT_DIR = SCRIPT_DIR / "3.1_split_data"
TRAIN_PATH = SPLIT_DIR / "train_1985_2007.xlsx"

CRISIS_COL = "GFDD.OI.19"
ZSCORE_COL = "GFDD.SI.01"
COUNTRY_COL = "Country Name"
TIME_COL = "Time"
# Columns to lag (y_t on X_{t-k}); add any regressor column names
COLS_TO_LAG = [ZSCORE_COL]
LAG_PERIODS = 1


def main():
    # Use train split from 3.1_split_data (single source of truth for period)
    df_train = pd.read_excel(TRAIN_PATH)
    df_train = standardization.winsorize(df_train, list_for_winsorization)
    df_train = standardization.standardize(df_train, list_for_standardization)
    required = list(set(COLS_TO_LAG + [CRISIS_COL]))
    if COUNTRY_COL in df_train.columns:
        required.append(COUNTRY_COL)
    if TIME_COL in df_train.columns:
        required.append(TIME_COL)
    df_train = df_train[[c for c in required if c in df_train.columns]].dropna()
    # Lagged X: within each country (y_t on X_{t-k})
    df_train = fixed_effects.add_lagged_columns(
        df_train, COLS_TO_LAG, LAG_PERIODS, COUNTRY_COL, TIME_COL
    )
    lagged_cols = fixed_effects.lagged_column_names(COLS_TO_LAG, LAG_PERIODS)
    df_train = df_train.dropna(subset=lagged_cols)
    y = np.asarray(df_train[CRISIS_COL], dtype=np.int64)

    X = fixed_effects.build_design_matrix(
        df_train,
        lagged_cols,
        country_col=COUNTRY_COL if COUNTRY_COL in df_train.columns else None,
        time_col=TIME_COL if TIME_COL in df_train.columns else None,
    )

    weights = class_weights.compute_inverse_frequency_weights(
        df_train, target_col=CRISIS_COL
    )
    obs_weights = np.array([weights[int(yi)] for yi in y])

    X_const = sm.add_constant(X).astype(float)
    model = sm.Logit(y, X_const)
    result = model.fit(weights=obs_weights, disp=0)

    print("Logit (lagged X + country & year fixed effects), class weights from inverse frequency")
    print("=" * 60)
    print(result.summary())
    print("\nOdds ratios (exp(coef)):")
    print(np.exp(result.params))

    y_pred_proba = result.predict(X_const)
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f"\nROC AUC (train): {roc_auc:.4f}")


if __name__ == "__main__":
    main()
