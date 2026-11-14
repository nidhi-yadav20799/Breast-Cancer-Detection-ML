from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "wdbc.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "wdbc_cleaned.csv"

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst",
]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {DATA_PATH}. Place wdbc.csv in data/raw/."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()
    empty_columns = df.columns[df.isna().all()].tolist()
    if empty_columns:
        print("Removing completely empty columns:", empty_columns)
        df = df.drop(columns=empty_columns)

    required = ["id", "diagnosis"] + FEATURES
    missing = [column for column in required if column not in df.columns]
    unexpected = [column for column in df.columns if column not in required]
    if missing or unexpected:
        raise ValueError(
            f"Dataset schema mismatch. Missing: {missing}; unexpected: {unexpected}"
        )

    df = df[required].copy()
    if df.shape != (569, 32):
        raise ValueError(f"Expected 569 rows and 32 columns; found {df.shape}.")
    if df["diagnosis"].isna().any():
        raise ValueError("Diagnosis contains missing values.")
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    if set(df["diagnosis"].unique()) != {"B", "M"}:
        raise ValueError("Diagnosis must contain exactly the labels B and M.")

    if df["id"].isna().any() or not df["id"].is_unique:
        raise ValueError("ID values must be present and unique.")
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    values = df[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictors contain missing, non-numeric or infinite values.")
    if (values < 0).any():
        raise ValueError("WDBC predictor measurements must be non-negative.")
    if df.duplicated().any():
        raise ValueError("Duplicate observations were found.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("WDBC DATA VALIDATION PASSED")
    print("=" * 60)
    print("Cleaned shape:", df.shape)
    print("Diagnosis distribution:")
    print(df["diagnosis"].value_counts())
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print("Saved:", OUTPUT_PATH)

if __name__ == "__main__":
    main()
