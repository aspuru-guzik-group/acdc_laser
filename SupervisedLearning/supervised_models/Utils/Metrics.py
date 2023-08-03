from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, fbeta_score, roc_auc_score, cohen_kappa_score
from scipy.stats import pearsonr, spearmanr, kendalltau
from .SignificantDigits import significant_digits


def calculate_regression_metrics(
        true_values: np.ndarray,
        predicted_values: np.ndarray,
        predicted_uncertainty: np.ndarray,
        n_targets: int = 1
) -> Dict[str, List[float]]:
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
        n_targets: Number of targets (default: 1)

    Returns:
        dict: Dictionary of metrics names and a list of values for each target.
    """
    methods = {
        "R^2": lambda true, predicted, uncertainty: significant_digits(r2_score(true, predicted), 3),
        "MAE": lambda true, predicted, uncertainty: significant_digits(mean_absolute_error(true, predicted), 3),
        "RMSE": lambda true, predicted, uncertainty: significant_digits(mean_squared_error(true, predicted, squared=False), 3),
        "R": lambda true, predicted, uncertainty: significant_digits(pearsonr(true, predicted)[0], 3),
        "rho": lambda true, predicted, uncertainty: significant_digits(spearmanr(true, predicted)[0], 3),
        "tau": lambda true, predicted, uncertainty: significant_digits(kendalltau(true, predicted)[0], 3),
        "Uncertainty": lambda true, predicted, uncertainty: significant_digits(float(np.mean(uncertainty)), 3) if uncertainty.size > 0 else np.nan,
        "Uncertainty_Correlation": lambda true, predicted, uncertainty: significant_digits(pearsonr(uncertainty, abs(true - predicted))[0], 3) if uncertainty.size == predicted.size else np.nan
    }

    metrics = {name: [] for name in methods.keys()}

    for i in range(n_targets):

        true_values_target = true_values[:, i].flatten()
        predicted_values_target = predicted_values[:, i].flatten()
        predicted_uncertainty_target = predicted_uncertainty[:, i].flatten()

        # Filter out any data points where the prediction is nan (for whatever reason)
        indices_to_consider = np.invert(np.isnan(predicted_values_target) | np.isinf(predicted_values_target) | np.isnan(true_values_target) | np.isinf(true_values_target))
        true_values_target = true_values_target[indices_to_consider]
        predicted_values_target = predicted_values_target[indices_to_consider]
        predicted_uncertainty_target = predicted_uncertainty_target[indices_to_consider]

        for name, method in methods.items():
            metrics[name].append(method(true_values_target, predicted_values_target, predicted_uncertainty_target))

    return metrics


def calculate_classification_metrics(
        true_labels: np.ndarray,
        predicted_labels: np.ndarray,
        predicted_uncertainty: np.ndarray,
        n_targets: int = 1
) -> Dict[str, float]:
    """
    Calculates performance metrics for binary classification tasks.

    Args:
        true_labels: Numpy array of true experimental target labels
        predicted_labels: Numpy array of predicted experimental labels
        predicted_uncertainty: Numpy array of the ML prediction uncertainty.
        n_targets: Number of targets (default: 1)

    Returns:
        dict: Dictionary of metrics names and their values
    """
    methods = {
        "accuracy": lambda true, predicted, uncertainty: significant_digits(accuracy_score(true, predicted), 3),
        "roc_auc": lambda true, predicted, uncertainty: significant_digits(roc_auc_score(true, predicted), 3),
        "f1_score": lambda true, predicted, uncertainty: significant_digits(fbeta_score(true, predicted, beta=1), 3),
        "f2_score": lambda true, predicted, uncertainty: significant_digits(fbeta_score(true, predicted, beta=2), 3),
        "cohen_kappa": lambda true, predicted, uncertainty: significant_digits(cohen_kappa_score(true, predicted), 3)
    }

    metrics = {name: [] for name in methods.keys()}

    for i in range(n_targets):

        true_labels_target = true_labels[:, i].flatten()
        predicted_labels_target = predicted_labels[:, i].flatten()
        predicted_uncertainty_target = predicted_uncertainty[:, i].flatten()  # Currently unused

        for name, method in methods.items():
            metrics[name].append(method(true_labels_target, predicted_labels_target, predicted_uncertainty_target))

    return metrics


def calculate_average_metrics(
        metrics_per_iteration: List[Dict[str, Dict[str, List[float]]]],
        eval_metric: str = "R^2",
        n_targets: int = 1,
        target_index: Optional[int] = None
) -> Tuple[dict, float]:
    """
    Averages the metrics obtained in multiple validation cycles.

    Args:
        metrics_per_iteration: List of performance metrics per iteration
        eval_metric: Name of the metric to be returned as a loss
        target_index: Index of the target to be used for the evaluation (default: None -> averaged over all targets)

    Returns:
        dict: Calculated average metrics (mean, standard deviation)
        float: Value of the evaluation metric.
    """
    metrics: list = list(metrics_per_iteration[0]["train"].keys())

    average_performance: dict = {}
    for partition in ("train", "test"):
        average_performance[partition] = {}
        for metric in metrics:
            average_performance[partition][metric] = []
            for idx in range(n_targets):
                average_performance[partition][metric].append(
                    [
                        significant_digits(float(np.mean([iteration[partition][metric][idx] for iteration in metrics_per_iteration])), 4),
                        significant_digits(float(np.std([iteration[partition][metric][idx] for iteration in metrics_per_iteration])), 4)
                    ]
                )

    try:
        if target_index is None:
            return average_performance, np.mean([average_performance["test"]["eval_metric"][i][0] for i in range(n_targets)])
        else:
            return average_performance, average_performance["test"][eval_metric][target_index][0]

    except (KeyError, IndexError):
        return average_performance, -100.0
        # TODO: Implement a default "bad" value for each type of eval_metrc
