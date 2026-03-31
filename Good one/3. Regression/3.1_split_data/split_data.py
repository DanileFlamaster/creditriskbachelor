"""
Split Final data into train (1985-2007), validation (2008-2010), test (2011-2021).
Save the three datasets and an Imbalance table (N obs, N failures, % failures) per period.
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "1. Clean data"
FINAL_DATA_PATH = r"D:\creditriskbachelor\Good one\1. Clean data\Final data.xlsx" 

TIME_COL = "Time"
WORLD_TIME_COL = "Year"
CRISIS_COL = "GFDD.OI.19"

# Period definitions (inclusive)
TRAIN_YEARS = (1985, 2013)
VAL_YEARS = (2014, 2015)
TEST_YEARS = (2016, 2017)


def load_final_data():
    cleaned_df = pd.read_excel(FINAL_DATA_PATH, sheet_name="Cleaned data")
    world_gdp_df = pd.read_excel(FINAL_DATA_PATH, sheet_name="World GDP Growth")
    return cleaned_df, world_gdp_df


def split_by_year(df, time_col, start_year, end_year):
    return df[(df[time_col] >= start_year) & (df[time_col] <= end_year)]


def period_label(period_name, years):
    return f"{period_name} ({years[0]}-{years[1]})"


def period_filename(period_name, years):
    return f"{period_name.lower()}_{years[0]}_{years[1]}.xlsx"


def imbalance_row(df, period_name):
    """N obs, N failures, % failures for a subset (crisis = 1)."""
    n_obs = len(df)
    if CRISIS_COL not in df.columns:
        return {"Period": period_name, "N_obs": n_obs, "N_failures": None, "Pct_failures": None}
    valid = df[CRISIS_COL].dropna()
    n_valid = len(valid)
    n_failures = (valid == 1).sum()
    pct = (n_failures / n_valid * 100) if n_valid > 0 else None
    return {"Period": period_name, "N_obs": n_obs, "N_failures": int(n_failures), "Pct_failures": round(pct, 2) if pct is not None else None}


def main():
    df, world_gdp_df = load_final_data()
    print(f"Loaded: {FINAL_DATA_PATH}")
    print(f"Total rows: {len(df)}")

    train = split_by_year(df, TIME_COL, TRAIN_YEARS[0], TRAIN_YEARS[1])
    val = split_by_year(df, TIME_COL, VAL_YEARS[0], VAL_YEARS[1])
    test = split_by_year(df, TIME_COL, TEST_YEARS[0], TEST_YEARS[1])

    world_train = split_by_year(world_gdp_df, WORLD_TIME_COL, TRAIN_YEARS[0], TRAIN_YEARS[1])
    world_val = split_by_year(world_gdp_df, WORLD_TIME_COL, VAL_YEARS[0], VAL_YEARS[1])
    world_test = split_by_year(world_gdp_df, WORLD_TIME_COL, TEST_YEARS[0], TEST_YEARS[1])

    # Save splits
    train_path = SCRIPT_DIR / period_filename("train", TRAIN_YEARS)
    val_path = SCRIPT_DIR / period_filename("validation", VAL_YEARS)
    test_path = SCRIPT_DIR / period_filename("test", TEST_YEARS)
    with pd.ExcelWriter(train_path) as writer:
        train.to_excel(writer, sheet_name="Cleaned data", index=False)
        world_train.to_excel(writer, sheet_name="World GDP Growth", index=False)
    with pd.ExcelWriter(val_path) as writer:
        val.to_excel(writer, sheet_name="Cleaned data", index=False)
        world_val.to_excel(writer, sheet_name="World GDP Growth", index=False)
    with pd.ExcelWriter(test_path) as writer:
        test.to_excel(writer, sheet_name="Cleaned data", index=False)
        world_test.to_excel(writer, sheet_name="World GDP Growth", index=False)
    print(f"\nSaved:\n  {train_path}\n  {val_path}\n  {test_path}")
    print(f"  Train: {len(train)} rows | Validation: {len(val)} rows | Test: {len(test)} rows")

    # Imbalance table: one row per period
    imbalance_rows = [
        imbalance_row(train, period_label("Train", TRAIN_YEARS)),
        imbalance_row(val, period_label("Validation", VAL_YEARS)),
        imbalance_row(test, period_label("Test", TEST_YEARS)),
    ]
    imbalance_table = pd.DataFrame(imbalance_rows)
    imbalance_path = SCRIPT_DIR / "Imbalance_table.xlsx"
    imbalance_table.to_excel(imbalance_path, index=False)
    print(f"\nImbalance table saved: {imbalance_path}")
    print(imbalance_table.to_string(index=False))


if __name__ == "__main__":
    main()
