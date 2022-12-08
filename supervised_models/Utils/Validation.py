from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr
from .SignificantDigits import significant_digits
import matplotlib.pyplot as plt
import pandas as pd


def calculate_regression_metrics(
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        predicted_uncertainty: np.ndarray
) -> Dict[str, float]:
    """
    Analyzes a set of prediction made by computing relevant correlation metrics.
        - Determination Coefficient R^2
        - Mean Absolute Error
        - Root Mean Square Error
        - Pearson Correlation Coefficient R

    Args:
        true_values: Numpy array of true experimental target values
        predicted_values: Numpy array of the ML-predicted target values
        predicted_uncertainty: Numpy array of the ML prediction uncertainty.

    Returns:
        dict: Dictionary of metrics names and their values.
    """
    true_values = true_values.flatten()
    predicted_values = predicted_values.flatten()
    predicted_uncertainty = predicted_uncertainty.flatten()

    # Filter out any data points where the prediction is nan (for whatever reason)
    true_values = true_values[np.invert(np.isnan(predicted_values) | np.isinf(predicted_values))]
    predicted_values = predicted_values[np.invert(np.isnan(predicted_values) | np.isinf(predicted_values))]
    predicted_uncertainty = predicted_uncertainty[np.invert(np.isnan(predicted_uncertainty) | np.isinf(predicted_uncertainty))]

    metrics: dict = {
        "R^2": significant_digits(r2_score(true_values, predicted_values), 3),
        "MAE": significant_digits(mean_absolute_error(true_values, predicted_values), 3),
        "RMSE": significant_digits(mean_squared_error(true_values, predicted_values, squared=False), 3),
        "R": significant_digits(pearsonr(true_values, predicted_values)[0], 3),
        "rho": significant_digits(spearmanr(true_values, predicted_values)[0], 3),
        "Uncertainty": significant_digits(float(np.mean(predicted_uncertainty)), 3) if predicted_uncertainty.size > 0 else np.nan,
        "Uncertainty_Correlation": significant_digits(pearsonr(predicted_uncertainty, abs(true_values - predicted_values))[0], 3) if predicted_uncertainty.size == predicted_values.size else np.nan
    }

    return metrics


def calculate_classification_metrics(
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        predicted_uncertainty: np.ndarray,
) -> Dict[str, float]:
    """
    Calculates performance metrics for binary classification tasks.

    Args:
        true_labels: Numpy array of true experimental target labels
        predicted_labels: Numpy array of predicted experimental labels
        predicted_uncertainty: Numpy array of the ML prediction uncertainty.

    Returns:
        dict: Dictionary of metrics names and their values
    """
    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    predicted_uncertainty = predicted_uncertainty.flatten()  # Currently unused

    metrics: dict = {
        "accuracy": significant_digits(accuracy_score(true_labels, predicted_labels), 3),
        "roc_auc": significant_digits(roc_auc_score(true_labels, predicted_labels), 3),
        "f1_score": significant_digits(fbeta_score(true_labels, predicted_labels, beta=1), 3),
        "f2_score": significant_digits(fbeta_score(true_labels, predicted_labels, beta=2), 3),
        "cohen_kappa": significant_digits(cohen_kappa_score(true_labels, predicted_labels), 3)
    }

    return metrics


def calculate_average_metrics(
        metrics_per_iteration: List[Dict[str, Dict[str, float]]],
        eval_metric: str = "R^2"
) -> Tuple[dict, float]:
    """
    Averages the metrics obtained in multiple validation cycles.

    Args:
        metrics_per_iteration: List of performance metrics per iteration
        eval_metric: Name of the metric to be returned as a loss

    Returns:
        dict: Calculated average metrics (mean, standard deviation)
        float: Value of the evaluation metric.
    """
    metrics: list = list(metrics_per_iteration[0]["train"].keys())

    average_performance: dict = {}
    for partition in ("train", "test"):
        average_performance[partition] = {}
        for metric in metrics:
            average_performance[partition][metric] = [
                significant_digits(float(np.mean([iteration[partition][metric] for iteration in metrics_per_iteration])), 4),
                significant_digits(float(np.std([iteration[partition][metric] for iteration in metrics_per_iteration])), 4)
            ]

    return average_performance, average_performance["test"][eval_metric][0]


def plot_regression(data_path: Path, file_pattern: str, metric_name: str, metric_value: float) -> None:
    """
    Plots the predictions loaded from csv file(s) with predictive performance(s).
    Each file must contain the following columns: True Values, Predicted Values, PredictionUncertainty

    Args:
        data_path: Path to the folder where the prediction data is stored.
        file_pattern: Common pattern for all data-containing files (passed to Path.glob()).
        metric_name: Name of the evaluation metric to be displayed in the plot.
        metric_value: Value of the evaluation metric to be displayed in the plot.
    """
    min_value, max_value = 0, 0
    fig, ax = plt.subplots()
    for file in data_path.glob(file_pattern):
        prediction_data: pd.DataFrame = pd.read_csv(file).fillna(0.0)
        ax.errorbar(
            prediction_data["True Values"],
            prediction_data["Predicted Values"],
            yerr=prediction_data["Prediction Uncertainty"],
            label=file.stem,
            fmt="o"
        )
        min_value = min(min_value, np.amin(prediction_data["True Values"]), np.amin(prediction_data["Predicted Values"]))
        max_value = max(max_value, np.amax(prediction_data["True Values"]), np.amax(prediction_data["Predicted Values"]))

    ax.plot([min_value, max_value], [min_value, max_value], 'k--')
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{metric_name} = {metric_value}")
    ax.legend()
    fig.savefig(data_path / "Results.png")
    plt.close(fig)

