import importlib.util
import pandas as pd
from pathlib import Path
#list_for_winsorization and list_for_standardization are the columns that will be winsorized and standardized   
list_for_winsorization = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"] # Credit to Deposit, NPL, Z-Score
list_for_standardization = ["GFDD.SI.01"] # Z-Score

SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "standardization", SCRIPT_DIR / "3.3_standardization.py"
)
standardization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(standardization)

DATA_DIR = SCRIPT_DIR.parent / "1. Clean data"
FINAL_DATA_PATH = DATA_DIR / "Final data.xlsx"


def load_final_data() -> pd.DataFrame:
    """Load the final DF produced by 1.1_Data_clean (Final data.xlsx)."""
    return pd.read_excel(FINAL_DATA_PATH)


# Example: load final DF, then winsorize and standardize
if __name__ == "__main__":
    df = load_final_data()
    # df = standardization.winsorize(df, ["col1", "col2"])
    # df = standardization.standardize(df, ["col1", "col2"])

df = standardization.winsorize(df, list_for_winsorization)
df = standardization.standardize(df, list_for_standardization)
df.to_excel('winsorized_standardized_data.xlsx', index=False)