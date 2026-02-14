"""
Compute class weights for the training set (1985–2007) using
the inverse class frequency method.

Weights are computed for the binary crisis variable GFDD.OI.19:
  w_c = N / (K * n_c)
where:
  - N  = total number of samples in the training set
  - K  = number of classes (here K = 2: crisis / no crisis)
  - n_c = number of samples in class c
"""

import pandas as pd
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPLIT_DIR = SCRIPT_DIR / "split data"
TRAIN_PATH = SPLIT_DIR / "train_1985_2007.xlsx"

CRISIS_COL = "GFDD.OI.19"  # 1 = crisis, 0 = no crisis


def compute_inverse_frequency_weights(df: pd.DataFrame, target_col: str = CRISIS_COL) -> dict:
    """
    Compute class weights using the inverse class frequency formula:

        w_c = N / (K * n_c)

    where:
        - N   = total number of samples with non-missing target
        - K   = number of classes
        - n_c = number of samples in class c

    Returns a dict: {class_value: weight}.
    """
    # Drop missing target values, if any
    y = df[target_col].dropna()

    N = len(y)
    class_counts = y.value_counts().sort_index()
    K = class_counts.shape[0]

    weights = {}
    for cls, n_c in class_counts.items():
        # Inverse class frequency weight
        w_c = N / (K * n_c)
        weights[int(cls)] = float(w_c)

    return weights


def main():
    # Load the training set
    df_train = pd.read_excel(TRAIN_PATH)
    print(f"Loaded training data from: {TRAIN_PATH}")
    print(f"Training rows: {len(df_train)}")

    # Compute class weights
    weights = compute_inverse_frequency_weights(df_train, CRISIS_COL)

    print("\nClass weights (inverse class frequency):")
    print("Formula:  w_c = N / (K * n_c)")
    for cls, w in sorted(weights.items()):
        print(f"  Class {cls}: weight = {w:.4f}")


if __name__ == "__main__":
    main()

