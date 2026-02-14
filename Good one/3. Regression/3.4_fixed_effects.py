"""
Build regression design matrices with optional country and/or year fixed effects.
Dummies use drop_first=True (one reference category per dimension).
"""
from typing import List, Optional

import pandas as pd


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
