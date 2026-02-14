"""
Split Final data into train (1985-2007), validation (2008-2010), test (2011-2021).
Save the three datasets and an Imbalance table (N obs, N failures, % failures) per period.
"""
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "1. Clean data"
FINAL_DATA_PATH = DATA_DIR / "Final data.xlsx"

TIME_COL = "Time"
CRISIS_COL = "GFDD.OI.19"

# Period definitions (inclusive)
TRAIN_YEARS = (1985, 2007)
VAL_YEARS = (2008, 2010)
TEST_YEARS = (2011, 2021)


def load_final_data():
    return pd.read_excel(FINAL_DATA_PATH)


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
    df = load_final_data()
    print(f"Loaded: {FINAL_DATA_PATH}")
    print(f"Total rows: {len(df)}")

    train = df[(df[TIME_COL] >= TRAIN_YEARS[0]) & (df[TIME_COL] <= TRAIN_YEARS[1])]
    val = df[(df[TIME_COL] >= VAL_YEARS[0]) & (df[TIME_COL] <= VAL_YEARS[1])]
    test = df[(df[TIME_COL] >= TEST_YEARS[0]) & (df[TIME_COL] <= TEST_YEARS[1])]

    # Save splits
    train_path = SCRIPT_DIR / "train_1985_2007.xlsx"
    val_path = SCRIPT_DIR / "validation_2008_2010.xlsx"
    test_path = SCRIPT_DIR / "test_2011_2021.xlsx"
    train.to_excel(train_path, index=False)
    val.to_excel(val_path, index=False)
    test.to_excel(test_path, index=False)
    print(f"\nSaved:\n  {train_path}\n  {val_path}\n  {test_path}")
    print(f"  Train: {len(train)} rows | Validation: {len(val)} rows | Test: {len(test)} rows")

    # Imbalance table: one row per period
    imbalance_rows = [
        imbalance_row(train, "Train (1985-2007)"),
        imbalance_row(val, "Validation (2008-2010)"),
        imbalance_row(test, "Test (2011-2021)"),
    ]
    imbalance_table = pd.DataFrame(imbalance_rows)
    imbalance_path = SCRIPT_DIR / "Imbalance_table.xlsx"
    imbalance_table.to_excel(imbalance_path, index=False)
    print(f"\nImbalance table saved: {imbalance_path}")
    print(imbalance_table.to_string(index=False))


if __name__ == "__main__":
    main()
