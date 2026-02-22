"""
Control variables and correlation matrix utilities.
Data source: Good one/1. Clean data/Final data.xlsx
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Path to Final data (from 1.1_Data_clean output)
SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DATA_PATH = SCRIPT_DIR / "Good one" / "1. Clean data" / "Final data.xlsx"

# List of variable names to use as controls (columns in Final data.xlsx)
CONTROL_VARIABLES = [
    "GFDD.OI.06",
    "GFDD.AI.25",
    "GFDD.AI.01",
    "GFDD.AI.02",
    "GFDD.OI.02",
    "GFDD.EI.01",
    "GFDD.EI.03",
    "GFDD.EI.04",
    "GFDD.SI.05",
    "GFDD.DI.02",
    "GFDD.DI.12"]


def load_final_data() -> pd.DataFrame:
    """Load Final data.xlsx."""
    return pd.read_excel(FINAL_DATA_PATH)


def compute_and_print_correlation_matrix(
    df: pd.DataFrame,
    control_variables: list,
) -> pd.DataFrame:
    """
    Compute the correlation matrix for the given variables and print it.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables.
    control_variables : list
        List of column names (e.g. CONTROL_VARIABLES) to include in the matrix.

    Returns
    -------
    pd.DataFrame
        The correlation matrix.
    """
    cols = [c for c in control_variables if c in df.columns]
    if not cols:
        print("No requested variables found in DataFrame.")
        return pd.DataFrame()

    corr = df[cols].corr()
    print("Correlation matrix:")
    print(corr.to_string())
    return corr


def plot_correlation_matrix_heatmap(
    corr: pd.DataFrame,
    *,
    cmap: str = "coolwarm",
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot correlation matrix as a heatmap with a temperature-style colormap.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix (e.g. output of compute_and_print_correlation_matrix).
    cmap : str
        Matplotlib colormap name. 'coolwarm' = blue (low) to red (high).
    vmin, vmax : float
        Color scale limits (correlation range).
    figsize : tuple
        Figure (width, height) in inches.
    """
    if corr.empty:
        return
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.show()


def compute_and_print_vif(
    df: pd.DataFrame,
    control_variables: list,
) -> pd.Series:
    """
    Compute Variance Inflation Factor (VIF) for the given variables and print it.

    VIF > 5 (or 10) is often taken as indicating problematic multicollinearity.
    Uses OLS of each variable on the others (with constant); drops rows with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variables.
    control_variables : list
        List of column names (e.g. CONTROL_VARIABLES).

    Returns
    -------
    pd.Series
        VIF per variable (index = variable name).
    """
    cols = [c for c in control_variables if c in df.columns]
    if not cols:
        print("No requested variables found in DataFrame.")
        return pd.Series(dtype=float)

    X = df[cols].dropna()
    if len(X) < len(cols) + 2:
        print("Not enough non-NaN rows to compute VIF.")
        return pd.Series(dtype=float)

    X = sm.add_constant(X)
    vif = {}
    for i in range(1, X.shape[1]):
        try:
            vif[X.columns[i]] = variance_inflation_factor(X.values, i)
        except Exception:
            vif[X.columns[i]] = float("nan")

    vif_series = pd.Series(vif)
    print("VIF (Variance Inflation Factor):")
    print(vif_series.to_string())
    return vif_series


def main():
    df = load_final_data()
    corr = compute_and_print_correlation_matrix(df, CONTROL_VARIABLES)
    plot_correlation_matrix_heatmap(corr)
    compute_and_print_vif(df, CONTROL_VARIABLES)


if __name__ == "__main__":
    main()
