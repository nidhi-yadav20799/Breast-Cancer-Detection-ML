
from pathlib import Path
import pandas as pd

# Project configuration
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "wdbc.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("BREAST CANCER DETECTION - DATA VALIDATION")
print("=" * 60)

print("\nOriginal shape:", df.shape)

# Identify and remove completely empty columns
empty_columns = df.columns[df.isna().all()].tolist()

if empty_columns:
    print("\nRemoving completely empty columns:", empty_columns)
    df = df.drop(columns=empty_columns)

print("\nCleaned shape:", df.shape)

# Dataset overview
print("\nFirst 5 records:")
print(df.head())

print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate IDs:", df["id"].duplicated().sum())

print("\nData types:")
print(df.dtypes.value_counts())

# Validation
assert df.shape == (569, 32), "Unexpected dataset dimensions!"
assert set(df["diagnosis"].unique()) == {"M", "B"}
assert df["id"].is_unique
assert df.isnull().sum().sum() == 0
assert df.duplicated().sum() == 0

print("\n" + "=" * 60)
print("SUCCESS: DATASET VALIDATION PASSED!")
print("=" * 60)


# Save cleaned dataset
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "wdbc_cleaned.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", OUTPUT_PATH)