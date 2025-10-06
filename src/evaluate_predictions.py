from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_preds(y_true, y_pred):
    """Compute common regression metrics given actual and predicted values."""
    if y_true is None or len(y_true) == 0:
        print("No actual values provided.")
        return None

    metrics = {}
    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["RMSE"] = np.sqrt(metrics["MSE"])
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    metrics["R2"] = r2_score(y_true, y_pred)

    print("Evaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    return metrics