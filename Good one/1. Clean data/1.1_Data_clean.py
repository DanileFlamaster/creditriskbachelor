import pandas as pd
from pathlib import Path


def excel_to_dataframe(file_path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
    """
    Open an Excel file and return it as a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the Excel file (.xlsx or .xls).
    sheet_name : str or int, default 0
        Name or index of the sheet to read. Use 0 for the first sheet.
    **kwargs
        Additional arguments passed to pandas.read_excel() (e.g. header, usecols).

    Returns
    -------
    pd.DataFrame
        The Excel data as a DataFrame.
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
    return df


def trim_by_year(df: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Trim dataframe to rows where the time column falls between start year and end year.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to trim.
    time_column : str
        Name of the column containing year or datetime values.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe.
    """
    START_YEAR = 1985  # --- Change these to your desired range ---
    END_YEAR = 2021

    series = df[time_column]
    # Extract year if column is datetime, otherwise assume it's already a year
    if pd.api.types.is_datetime64_any_dtype(series):
        years = series.dt.year
    else:
        years = pd.to_numeric(series, errors="coerce").astype("Int64")

    mask = (years >= START_YEAR) & (years <= END_YEAR)
    return df.loc[mask].reset_index(drop=True)


def replace_dotdot_with_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace 'Dot dot.' (and common variants) with null/NaN.
    Use this when dot dot indicates a missing value in the data.
    """
    # Values to treat as missing (replace with NaN)
    missing_values = [".. ", ".."]
    return df.replace(missing_values, pd.NA)


def non_na_coverage(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    For each specified column, compute the count and percentage of non-NA values.
    Returns a table: Variable | Absolute (count) | Relative (%).
    """
    n_rows = len(df)
    rows = []
    for col in columns:
        count = df[col].notna().sum()
        pct = (count / n_rows * 100) if n_rows > 0 else 0.0
        rows.append({"Variable": col, "Absolute": count, "Relative (%)": round(pct, 2)})
    return pd.DataFrame(rows)


# --- Insert your file path here ---
FILE_PATH = r"C:\Users\user\Desktop\Licenta\work\2\DBCleaned.xlsx"

df = excel_to_dataframe(FILE_PATH)

# Trim by year: set time_column to the name of your time/year column in the Excel file
TIME_COLUMN = "Time"  # e.g. "year", "Year", "date", "time"
df = trim_by_year(df, TIME_COLUMN)

# Replace "Dot dot." (and variants) with null
df = replace_dotdot_with_null(df)

# Non-NA coverage: define the columns you want to check
COLUMNS_TO_CHECK = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]  # replace with your column names
coverage_table = non_na_coverage(df, COLUMNS_TO_CHECK)
print(coverage_table)
print(df.head())

# Save cleaned data in the same folder as the original file
FINAL_DATA_PATH = Path(FILE_PATH).parent / "Final data.xlsx"
df.to_excel(FINAL_DATA_PATH, index=False)
print(f"Saved to: {FINAL_DATA_PATH}")
