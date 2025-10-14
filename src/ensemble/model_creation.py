from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import os
import joblib



def split_data_by_date(df, target="temp", forecast_start="2025-09-17", forecast_end="2025-09-30"):
    """
    Splits cleaned dataframe into train/test sets based on datetime.
    Training = all data before forecast_start
    Test = data between forecast_start and forecast_end
    """
    df = df.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    # Define forecast window
    mask_test = (df["datetime"] >= forecast_start) & (df["datetime"] <= forecast_end)
    mask_train = df["datetime"] < forecast_start

    train_df = df[mask_train]
    test_df = df[mask_test]

    # Features / labels
    X_train = train_df.drop(columns=[target, "datetime"], errors="ignore")
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target, "datetime"], errors="ignore")
    y_test = test_df[target] if target in test_df else None

    return X_train, y_train, X_test, y_test

def synthetic_september(df, forecast_start="2025-09-17", forecast_end="2025-09-30"):
    """
    Creates a synthetic forecast window (Sept 17–30 for the forecast year)
    using hourly averages of all September 17–30 data from 2015 to 2024.
    """
    df = df.copy()
    df["month_day"] = df["datetime"].dt.strftime("%m-%d")

    #Historical data for Sept 17–30 only (all years 2015–2024)
    sept_df = df[
        (df["month_day"] >= "09-17") &
        (df["month_day"] <= "09-30") &
        (df["year"] >= 2015) &
        (df["year"] <= 2024)
        ]

    # Average for each day/hour combo
    features = ["temp", "rhum", "prcp", "wdir"]
    avg_df = (
        sept_df.groupby(["month", "day", "hour"], as_index=False)[features]
        .mean(numeric_only=True) # Drops NaNs
    )

    # Create synthetic datetimes for forecast window (e.g. 2025)
    future_dates = pd.date_range(start=forecast_start, end=forecast_end, freq="h")
    future_df = pd.DataFrame({
        "datetime": future_dates,
        "year": future_dates.year,
        "month": future_dates.month,
        "day": future_dates.day,
        "hour": future_dates.hour
    })

    # Merge synthetic dates with historical averages on month/day/hour
    synthetic_df = pd.merge(future_df, avg_df, on=["month", "day", "hour"], how="left")

    # Create X_test (drop temp since it’s the target)
    X_test = synthetic_df.drop(columns=["temp", "datetime"], errors="ignore")

    return X_test, synthetic_df

def build_ensemble():
    """Build an ensemble model with RF, GB, and SVR."""
    rf = RandomForestRegressor(n_estimators=200, random_state=0)
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=0)
    svr = SVR(kernel="rbf", C=10, gamma=0.1)

    return VotingRegressor(estimators=[
        ("rf", rf),
        ("gb", gb),
        ("svr", svr)
    ])


def run_pipeline(weather_filename="Data_Cleaned/cleaned_weather.csv",
                 forecast_start="2025-09-17", forecast_end="2025-09-30"):
    """
    Train on all data before forecast_start, then predict for forecast window.
    """
    output_folder = "Nonlinear_Forecast_Final"

    df = pd.read_csv(weather_filename, parse_dates=["datetime"])
    df = df.sort_values("datetime")

    X_train, y_train, _, y_test = split_data_by_date(
        df, forecast_start=forecast_start, forecast_end=forecast_end
    )

    # Builds Synthetic September Data Set
    X_test, synthetic_df = synthetic_september(df, forecast_start, forecast_end)

    model = build_ensemble()
    print("Fitting ensemble...")
    model.fit(X_train, y_train)

    model_path = os.path.join(output_folder, "trained_model.joblib")
    X_test_path = os.path.join(output_folder, "X_test.csv")

    joblib.dump(model, model_path) ## In case runtime error and do not want to wait another 28 minutes.
    X_test.to_csv(X_test_path, index=False)

    # Predictions for the forecast window
    y_pred = model.predict(X_test)

    return X_train, y_train, X_test, y_test, model, y_pred
