"""
Build regression design matrices with optional country and/or year fixed effects.
Dummies use drop_first=True (one reference category per dimension).
Lagged variables are created within each entity (e.g. country).
"""
from typing import List, Optional

import pandas as pd


def add_lagged_columns(
    df: pd.DataFrame,
    columns_to_lag: List[str],
    lag_periods: int,
    country_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    Add lagged columns within each country. New columns are named '{col}_LAG{lag_periods}'.
    Modifies df in place and returns it. Sort by country and time so shift is correct.
    """
    df = df.sort_values([country_col, time_col])
    for col in columns_to_lag:
        if col not in df.columns:
            continue
        lag_col = f"{col}_LAG{lag_periods}"
        df[lag_col] = df.groupby(country_col)[col].shift(lag_periods)
    return df


def lagged_column_names(columns_to_lag: List[str], lag_periods: int) -> List[str]:
    """Return list of lagged column names for the given columns and lag."""
    return [f"{c}_LAG{lag_periods}" for c in columns_to_lag]


def add_country_dummies(df: pd.DataFrame, country_col: str, prefix: str = "country") -> pd.DataFrame:
    """Return country dummies (drop_first=True)."""
    return pd.get_dummies(df[country_col], prefix=prefix, drop_first=True)


def add_year_dummies(df: pd.DataFrame, time_col: str, prefix: str = "year") -> pd.DataFrame:
    """Return year dummies from time column (drop_first=True)."""
    years = pd.to_numeric(df[time_col], errors="coerce").astype(int)
    return pd.get_dummies(years, prefix=prefix, drop_first=True)


def build_design_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    country_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build design matrix: feature columns + optional country dummies + optional year dummies.
    Does not add intercept (use sm.add_constant(X) in the model).
    """
    parts = [df[feature_cols].astype(float)]
    if country_col is not None and country_col in df.columns:
        parts.append(add_country_dummies(df, country_col))
    if time_col is not None and time_col in df.columns:
        parts.append(add_year_dummies(df, time_col))
    return pd.concat(parts, axis=1).astype(float)
