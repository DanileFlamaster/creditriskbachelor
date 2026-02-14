import pandas as pd
import numpy as np
from typing import Union


def _ensure_columns(columns: Union[str, list[str]]) -> list[str]:
    """Normalize columns to a list of strings."""
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def winsorize(
    df: pd.DataFrame,
    columns: Union[str, list[str]],
    lower: float = 0.01,
    upper: float = 0.99,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Winsorize one or more columns at given percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe (e.g. final DF from 1.1_Data_clean).
    columns : str or list[str]
        Column name(s) to winsorize.
    lower : float, default 0.01
        Lower percentile (0–1). Values below are set to this quantile.
    upper : float, default 0.99
        Upper percentile (0–1). Values above are set to this quantile.
    inplace : bool, default False
        If True, modify df in place; otherwise work on a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with winsorized columns.
    """
    out = df if inplace else df.copy()
    cols = _ensure_columns(columns)
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not in dataframe")
        q_low = out[col].quantile(lower)
        q_high = out[col].quantile(upper)
        out[col] = out[col].clip(lower=q_low, upper=q_high)
    return out


def standardize(
    df: pd.DataFrame,
    columns: Union[str, list[str]],
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Z-score standardize one or more columns: (x - mean) / std.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe (e.g. after winsorization).
    columns : str or list[str]
        Column name(s) to standardize.
    inplace : bool, default False
        If True, modify df in place; otherwise work on a copy.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns.
    """
    out = df if inplace else df.copy()
    cols = _ensure_columns(columns)
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not in dataframe")
        mean = out[col].mean()
        std = out[col].std()
        if std == 0 or np.isnan(std):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std
    return out
