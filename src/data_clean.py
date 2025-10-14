import os
import pandas as pd
import numpy as np


# Load all CSVs
def load_all_weather_data(folder_path="Data_Raw"):
    """Loads and combines all weather CSV files in the given folder."""
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    all_dfs = []

    for file in sorted(all_files):
        file_path = os.path.join(folder_path, file)
        print(f"Reading {file} ...")
        df = pd.read_csv(file_path)

        # Normalize column names (lowercase, strip spaces)
        df.columns = df.columns.str.strip().str.lower()

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined dataset shape: {combined.shape[0]} rows, {combined.shape[1]} columns.")
    print(f"Columns found: {list(combined.columns)}")
    return combined

df = load_all_weather_data("Data_Raw")

def clean_data(df):
    """Clean dataset: build datetime, drop cldc, interpolate prcp/wdir, clip outliers."""

    df = df.copy()
    print("---- Cleaning Data ----")

    # Step 1: Ensure datetime column exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif all(c in df.columns for c in ["year", "month", "day", "hour"]):
        df["datetime"] = pd.to_datetime(
            df[["year", "month", "day", "hour"]],
            errors="coerce"
        )
        df = df.drop(columns=["year", "month", "day", "hour"], errors="ignore")
    else:
        raise KeyError("No valid datetime info found (need 'datetime' or year/month/day/hour).")

    # Move datetime to front
    cols = ["datetime"] + [c for c in df.columns if c != "datetime"]
    df = df[cols]

    print("Datetime column built and moved to front.")

    # Convert others to numeric
    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop cldc column if it exists
    if "cldc" in df.columns:
        df = df.drop(columns=["cldc"])
        print("Dropped 'cldc' column due to high missingness.")

    # Interpolate prcp and wdir if present
    for col in ["prcp", "wdir"]:
        if col in df.columns:
            nulls_before = df[col].isna().sum()
            df[col] = df[col].interpolate(method="nearest", limit_direction="both")
            nulls_after = df[col].isna().sum()
            print("Interpolated '{col}': {nulls_before} → {nulls_after} missing values.")

    # Drop rows with missing datetime or temp
    df = df.dropna(subset=["datetime", "temp"])

    # Sort chronologically
    df = df.sort_values("datetime").reset_index(drop=True)

    # Gentle outlier clipping
    for col in ["temp", "rhum", "prcp", "wdir"]:
        if col in df.columns:
            lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    print("Cleaned dataset has {len(df)} rows and {len(df.columns)} columns.")
    print("-----------------------")

    # Final sanity check: drop any row that still has NaN
    before = len(df)
    df = df.dropna()
    after = len(df)
    print("Sanity check: dropped {before - after} rows with remaining NaN values.")

    return df

# cleaned_df = clean_data(df)

# print("\nPreview of cleaned data:")
# print(cleaned_df.head())
