import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def model_eval(y_test: pd.Series, y_pred: pd.Series, time_index: pd.Series):
    """
    Evaluate model predictions and compute residual statistics.
    """
    print("Model Evaluation")

    residuals = y_test - y_pred
    abs_residuals = residuals.abs()

    print("Residual Summary")
    print("Mean Residual:", residuals.mean())
    print("Median Residual:", residuals.median())
    print("Residual Std Dev:", residuals.std())
    print("Mean Absolute Error:", residuals.abs().mean())
    print("Skewness:", residuals.skew())
    print("Kurtosis:", residuals.kurtosis())

    # Simple rolling mean absolute error
    window = 24
    rolling_mae = abs_residuals.rolling(window=window).mean()

    plt.figure(figsize=(14, 5))
    plt.plot(rolling_mae.values, color="red", linewidth=2)
    plt.title("Rolling Mean Absolute Error (Drift Over Time)")
    plt.xlabel("Hour Index")
    plt.ylabel("Rolling MAE")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return residuals

def graph_making(y_test: pd.Series, y_pred: pd.Series, time_index: pd.Series) -> None:
    """Generate residual diagnostic plots assuming time_index starts at 0 = Sept 17, 2025 00:00."""

    # Anchor base date
    base_date = pd.to_datetime("2025-09-17 00:00")
    time_index = base_date + pd.to_timedelta(time_index, unit="h")

    residuals = y_test - y_pred

    # Graph 1: Actual vs Predicted
    plt.figure(figsize=(14, 5))
    plt.plot(time_index, y_test, label="Actual", linewidth=2)
    plt.plot(time_index, y_pred, label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted Temperatures")
    plt.xlabel("Date")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(time_index[::24], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Graph 2: Residuals Over Time
    plt.figure(figsize=(14, 5))
    plt.plot(time_index, residuals, color="gray", linewidth=1.5)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residuals Over Forecast Horizon")
    plt.xlabel("Date")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.3)
    plt.xticks(time_index[::24], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Graph 3: Absolute Residuals
    plt.figure(figsize=(14, 5))
    plt.plot(time_index, np.abs(residuals), color="red", linewidth=1.5)
    plt.title("Absolute Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.grid(alpha=0.3)
    plt.xticks(time_index[::24], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Graph 4: Residual Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, color="gray", edgecolor="black", alpha=0.8)
    plt.title("Residual Distribution")
    plt.xlabel("Residual (C)")
    plt.ylabel("Frequency")
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.show()