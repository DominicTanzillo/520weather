from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
import pandas as pd
import numpy as np

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
    df = pd.read_csv(weather_filename, parse_dates=["datetime"])
    df = df.sort_values("datetime")

    X_train, y_train, X_test, y_test = split_data_by_date(
        df, forecast_start=forecast_start, forecast_end=forecast_end
    )

    model = build_ensemble()
    print("Fitting ensemble...")
    model.fit(X_train, y_train)

    # Predictions for the forecast window
    y_pred = model.predict(X_test)

    return X_train, y_train, X_test, y_test, model, y_pred
