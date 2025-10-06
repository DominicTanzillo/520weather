import os
import pandas as pd
import numpy as np

def load_and_clean_weather(
    folder_path="Data_Raw",
    station_id="72306",
    start_year=2015,
    end_year=2025,
    output_file="cleaned_weather.csv",
    keep_cols=["year","month","day","hour","temp","rhum","prcp","wdir","cldc"]
):
    """
    Load weather data files named like <station_id>-YYYY.csv for a given year range,
    clean them, and save as one combined CSV.
    """

    all_dfs = []

    for year in range(start_year, end_year + 1):
        file_name = f"{station_id}-{year}.csv"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.exists(file_path):
            continue

        print(f"Reading {file_name} ...")
        df = pd.read_csv(file_path)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Keep only relevant columns
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

        # Convert numeric columns
        for col in df.columns:
            if col not in ["year","month","day","hour"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Build datetime column (more robust than string formatting)
        if all(col in df.columns for col in ["year","month","day","hour"]):
            df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]], errors="coerce")

        all_dfs.append(df)

    if not all_dfs:
        raise FileNotFoundError("No weather files found for the specified range.")

    # Combine everything
    combined = pd.concat(all_dfs, ignore_index=True)

    # Drop rows missing datetime or temp
    combined = combined.dropna(subset=["datetime","temp"])

    # Sort by datetime
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # Gentle outlier clipping
    for col in ["temp","rhum","prcp","wdir","cldc"]:
        if col in combined.columns:
            lower, upper = combined[col].quantile(0.01), combined[col].quantile(0.99)
            combined[col] = combined[col].clip(lower, upper)

    # Save to CSV
    #output_path = os.path.join(folder_path, output_file)
    #combined.to_csv(output_path, index=False)

    print(f"Final dataset: {combined.shape[0]} rows, {combined.shape[1]} columns")
    print(f"Date range: {combined['datetime'].min()} → {combined['datetime'].max()}")
    #print(f"Saved cleaned CSV to: {output_path}")

    return combined
