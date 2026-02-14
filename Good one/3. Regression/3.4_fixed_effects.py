"""
Build regression design matrices with optional country and/or year fixed effects.
Dummies use drop_first=True (one reference category per dimension).
Lagged variables are created within each entity (e.g. country).
"""
from typing import List, Optional, Sequence

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


def add_lagged_columns_with_context(
    df_history: pd.DataFrame,
    df_target: pd.DataFrame,
    columns_to_lag: List[str],
    lag_periods: int,
    country_col: str,
    time_col: str,
    *,
    strict_time_order: bool = True,
) -> pd.DataFrame:
    """
    Add within-country lags to df_target using df_history as past context.

    Use this for validation/test prediction when you need X_{t-k} but your target
    period starts at a split boundary (e.g., validation starts in 2008 but lag=1
    needs 2007 values). This avoids "leakage" by only using df_history for lag
    construction, and returning rows from df_target only.

    Parameters
    ----------
    df_history:
        Earlier period data (e.g., train for validation; train+validation for test).
    df_target:
        Target period you want to score/evaluate (e.g., validation or test).
    strict_time_order:
        If True, enforce that history ends strictly before target begins (by year),
        to guard against accidental overlap.
    """
    if df_target.empty:
        return df_target.copy()

    # Work on copies; tag origin so we can return only target rows after shifting.
    hist = df_history.copy()
    targ = df_target.copy()
    hist["_is_target"] = 0
    targ["_is_target"] = 1

    combined = pd.concat([hist, targ], ignore_index=True, sort=False)
    combined[time_col] = pd.to_numeric(combined[time_col], errors="coerce")

    if strict_time_order:
        hist_years = pd.to_numeric(hist[time_col], errors="coerce").dropna()
        targ_years = pd.to_numeric(targ[time_col], errors="coerce").dropna()
        if not hist_years.empty and not targ_years.empty:
            if hist_years.max() >= targ_years.min():
                raise ValueError(
                    f"History/target overlap in '{time_col}': "
                    f"max(history)={hist_years.max()} >= min(target)={targ_years.min()}. "
                    "This would leak future information into lag construction."
                )

    combined = combined.dropna(subset=[country_col, time_col])
    combined = add_lagged_columns(combined, columns_to_lag, lag_periods, country_col, time_col)

    out = combined[combined["_is_target"] == 1].drop(columns=["_is_target"])
    return out


def add_country_dummies(df: pd.DataFrame, country_col: str, prefix: str = "country") -> pd.DataFrame:
    """Return country dummies (drop_first=True)."""
    return pd.get_dummies(df[country_col], prefix=prefix, drop_first=True)


def add_year_dummies(df: pd.DataFrame, time_col: str, prefix: str = "year") -> pd.DataFrame:
    """Return year dummies from time column (drop_first=True)."""
    years = pd.to_numeric(df[time_col], errors="coerce").astype(int)
    return pd.get_dummies(years, prefix=prefix, drop_first=True)


def add_country_dummies_with_categories(
    df: pd.DataFrame,
    country_col: str,
    categories: Sequence[str],
    prefix: str = "country",
) -> pd.DataFrame:
    """
    Return country dummies with a fixed category set, so drop_first uses a consistent
    reference category across train/validation/test.
    """
    series = pd.Series(
        pd.Categorical(df[country_col], categories=list(categories), ordered=True),
        index=df.index,
    )
    return pd.get_dummies(series, prefix=prefix, drop_first=True)


def add_year_dummies_with_categories(
    df: pd.DataFrame,
    time_col: str,
    categories: Sequence[int],
    prefix: str = "year",
) -> pd.DataFrame:
    """
    Return year dummies with a fixed category set (years), so drop_first uses a
    consistent reference year across splits.
    """
    years = pd.to_numeric(df[time_col], errors="coerce")
    years = pd.Series(
        pd.Categorical(years.astype("Int64"), categories=list(categories), ordered=True),
        index=df.index,
    )
    return pd.get_dummies(years, prefix=prefix, drop_first=True)


def build_design_matrix(
    df: pd.DataFrame,
    feature_cols: List[str],
    *,
    country_col: Optional[str] = None,
    time_col: Optional[str] = None,
    country_categories: Optional[Sequence[str]] = None,
    year_categories: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """
    Build design matrix: feature columns + optional country dummies + optional year dummies.
    Does not add intercept (use sm.add_constant(X) in the model).
    """
    parts = [df[feature_cols].astype(float)]
    if country_col is not None and country_col in df.columns:
        if country_categories is None:
            parts.append(add_country_dummies(df, country_col))
        else:
            parts.append(add_country_dummies_with_categories(df, country_col, country_categories))
    if time_col is not None and time_col in df.columns:
        if year_categories is None:
            parts.append(add_year_dummies(df, time_col))
        else:
            parts.append(add_year_dummies_with_categories(df, time_col, year_categories))
    return pd.concat(parts, axis=1).astype(float)
