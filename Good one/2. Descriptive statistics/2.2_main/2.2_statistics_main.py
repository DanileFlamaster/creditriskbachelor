"""
Descriptive statistics for the three main variables:
  GFDD.SI.04 - Credit to Deposit (%)
  GFDD.SI.02 - NPL (%)
  GFDD.SI.01 - Z-Score
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "1. Clean data"
FINAL_DATA_PATH = DATA_DIR / "Final data.xlsx"
TIME_COL = "Time"

MAIN_VARS = ["GFDD.SI.04", "GFDD.SI.02", "GFDD.SI.01"]
MAIN_LABELS = {
    "GFDD.SI.04": "Credit to Deposit (%)",
    "GFDD.SI.02": "NPL (%)",
    "GFDD.SI.01": "Z-Score",
}


def plot_mean_std_by_year(df, vars_present):
    """Plot cross-sectional mean and std by year for each variable."""
    if TIME_COL not in df.columns:
        print(f"'{TIME_COL}' column not found; skipping mean/std by year plot.")
        return
    nvars = len(vars_present)
    fig, axes = plt.subplots(nvars, 2, figsize=(12, 4 * nvars))
    if nvars == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(vars_present):
        by_year = df.groupby(TIME_COL)[col].agg(["mean", "std"]).reset_index()
        by_year = by_year.dropna(subset=["mean", "std"])
        label = MAIN_LABELS.get(col, col)
        axes[i, 0].plot(by_year[TIME_COL], by_year["mean"], marker="o", markersize=4)
        axes[i, 0].set_ylabel("Mean")
        axes[i, 0].set_title(f"{label} – mean by year")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].plot(by_year[TIME_COL], by_year["std"], marker="s", markersize=4, color="orange")
        axes[i, 1].set_ylabel("Std")
        axes[i, 1].set_title(f"{label} – std by year")
        axes[i, 1].grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    plt.tight_layout()
    out_path = SCRIPT_DIR / "mean_std_by_year.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    df = pd.read_excel(FINAL_DATA_PATH)
    vars_present = [v for v in MAIN_VARS if v in df.columns]
    if not vars_present:
        print("None of the main variables found in the data.")
        return

    desc = df[vars_present].describe().round(4)
    # Rename columns for display
    desc.columns = [MAIN_LABELS.get(c, c) for c in desc.columns]

    print("=" * 70)
    print("DESCRIPTIVE STATISTICS – Main variables")
    print("=" * 70)
    print(f"Data: {FINAL_DATA_PATH}")
    print(f"Rows: {len(df)}")
    print()
    print(desc)
    print()

    out_path = SCRIPT_DIR / "descriptive_stats_main_variables.xlsx"
    desc.to_excel(out_path)
    print(f"Saved: {out_path}")

    plot_mean_std_by_year(df, vars_present)


if __name__ == "__main__":
    main()
