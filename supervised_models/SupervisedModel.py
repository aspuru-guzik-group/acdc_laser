import time
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Optional
from pathlib import Path
import logging
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from hyperopt import fmin, tpe, space_eval, STATUS_OK, early_stop

from .Utils import save_csv, save_json
from .Utils import calculate_metrics, calculate_average_metrics, plot_predictions


class SupervisedModel(metaclass=ABCMeta):

    """
    Metaclass for supervised learning models with uncertainty prediction.
    Provides wrappers for model (cross) validation and hyperparameter optimization.
    """

    name = ""

    def __init__(
            self,
            output_dir: Path,
            hyperparameters_fixed: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            **kwargs
    ):
        """
        Instantiates the SupervisedModel object by setting a range of private attributes.

        Args:
             output_dir: Path to the directory where output data should be saved.
             hyperparameters_fixed: Dictionary of all model hyperparameters that should not be varied.
             hyperparameters: Dictionary of all model hyperparameters (name and hyperopt object).
             kwargs: Further keyword arguments
                        - "feature_scaler": Instance of a scaler (default: sklearn.preprocessing.StandardScaler).
                        - "target_scaler": Instance of a scaler (default: sklearn.preprocessing.StandardScaler).
                        - "random_state": Random state variable (default: 42)
                        - kwargs to be passed to the constructor of the specific model.
        """
        self._model: Any = None
        self._output_dir = output_dir
        self._hp_options = hyperparameters if hyperparameters else {}
        self._current_hp = hyperparameters_fixed if hyperparameters_fixed else {}
        self._feature_scaler = kwargs.pop("feature_scaler") if kwargs.get("feature_scaler") else StandardScaler()
        self._target_scaler = kwargs.pop("target_scaler") if kwargs.get("target_scaler") else StandardScaler()
        self._performance_metrics: dict = {}
        self._random_state = kwargs.pop("random_state") if kwargs.get("random_state") else 42
        self._kwargs: dict = kwargs

    @property
    def hyperparameters(self) -> dict:
        """
        Getter for the current hyperparameters.

        Returns:
            dict: Dictionary of current hyperparameters.
        """
        return self._current_hp

    @hyperparameters.setter
    def hyperparameters(self, parameters: dict):
        """
        Setter for some or all of the current hyperparameters. .

        Args:
             parameters: Key-value pairs for the current hyperparameters to be updated.

        Raises:
            KeyError: If a required hyperparameter is not defined.
            KeyError: If a hyperparameter is passed that is unknown for the current model.
        """
        for parameter, param_value in parameters.items():
            if parameter not in self._hp_options:
                raise KeyError(f"Error when setting the current hyperparamters. The parameter {parameter} is unknown!")

            # TODO: Put in general type checking / type conversion here, not this hacky workaround implementation
            if type(param_value) is np.int64:
                param_value = int(param_value)

            if type(param_value) is dict:
                self._current_hp.update(param_value)

            else:
                self._current_hp[parameter] = param_value

        # Checks if all required hyperparameters (keys in self._hp_options) are currently set.
        missing_hp: set = set(self._hp_options.keys()) - set(self._current_hp.keys())
        if len(missing_hp) > 0:
            raise KeyError(f"The hyperparameters {missing_hp} are undefined!")

    def train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Scales the features and targets, and calls the private train method.

        Args:
             features: Numpy ndarray (n_samples, n_features) of training features.
             targets: Numpy ndarray (n_samples, 1) of training targets.
        """
        features_scaled = self._feature_scaler.fit_transform(features)
        targets_scaled = self._target_scaler.fit_transform(targets)
        self._train(features_scaled, targets_scaled)

    @abstractmethod
    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Instantiates the model, and trains it with the passed training data.

        Args:
             features: Numpy ndarray (n_samples, n_features) of training features.
             targets: Numpy ndarray (n_samples, 1) of training targets.
        """
        raise NotImplementedError

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scales the features and calls the private predict method.

        Args:
             features: Ndarray (n_samples, n_features) of the features of the data points to predict.

        Returns:
            np.ndarray: Ndarray (n_samples, 1) of predicted values for each test datapoint.
            np.ndarray: Ndarray (n_samples, 1) of uncertainty for each test datapoint (np.nan if not implemented).
        """
        if not self._model:
            raise KeyError("The model has not been instantiated and trained.")

        features_scaled: np.ndarray = self._feature_scaler.transform(features)
        targets_scaled, uncertainty_scaled = self._predict(features_scaled)
        return self._target_scaler.inverse_transform(targets_scaled), self._target_scaler.inverse_transform(uncertainty_scaled)

    @abstractmethod
    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses self.model to predict the target values and uncertainties for the passed data points.

        Args:
             features: Ndarray (n_samples, n_features) of the features of the data points to predict.

        Returns:
            np.ndarray: Ndarray (n_samples, 1) of predicted values for each test datapoint.
            np.ndarray: Ndarray (n_samples, 1) of uncertainty for each test datapoint (np.nan if not implemented).
        """
        raise NotImplementedError

    def run(
            self,
            features: np.ndarray,
            targets: np.array,
            splitting_scheme: str,
            splitting_scheme_params: dict,
            train_expansion_features: Optional[np.ndarray] = None,
            train_expansion_targets: Optional[np.ndarray] = None,
            eval_metric: str = "R^2",
            max_evaluations: int = 100
    ) -> None:
        """
        Runs the model by performing hyperparameter optimization following the defined train-test splitting scheme.
        Evaluates the performance of the optimized model and saves the obtained predictions.

        Args:
            features: Numpy ndarray of the features of all data points.
            targets: Numpy ndarray of the corresponding targets for all data points.
            splitting_scheme: Name of the splitting scheme (must correspond to class name from sklearn.model_selection)
            splitting_scheme_params: Kwargs for the constructor of the splitting scheme.
            train_expansion_features: Ndarray of additional data points to be added to the training data (features).
            train_expansion_targets: Ndarray of additional data points to be added to the training data (targets).
            eval_metric: Name of the metric to be used as the loss function for hyperparameter optimization.
            max_evaluations: Maximum number of evaluations for hyperparameter optimiazation.
        """
        if self._hp_options:

            def objective(hyperparameters):
                """Callable object to be passed to the hyperopt fmin function."""
                self.hyperparameters = hyperparameters
                loss = self.evaluate_performance(
                    features,
                    targets,
                    splitting_scheme=splitting_scheme,
                    splitting_scheme_params=splitting_scheme_params,
                    name=time.strftime("%H%M%S", time.localtime()),
                    train_expansion_features=train_expansion_features,
                    train_expansion_targets=train_expansion_targets,
                    eval_metric=eval_metric
                )
                return {"loss": -loss, "status": STATUS_OK}

            optimal_params = fmin(
                objective,
                space=self._hp_options,
                algo=tpe.suggest,
                max_evals=max_evaluations,
                early_stop_fn=early_stop.no_progress_loss(int(0.2*max_evaluations))
            )

            self.hyperparameters = space_eval(self._hp_options, optimal_params)

        self.evaluate_performance(
            features,
            targets,
            splitting_scheme=splitting_scheme,
            splitting_scheme_params=splitting_scheme_params,
            name="Optimized_Model",
            train_expansion_features=train_expansion_features,
            train_expansion_targets=train_expansion_targets,
            eval_metric=eval_metric,
            save_predictions=True
        )

        save_json(self._performance_metrics, self._output_dir / "hyperparameter_optimization.json")

    def evaluate_performance(
            self,
            features: np.ndarray,
            targets: np.array,
            splitting_scheme: str,
            splitting_scheme_params: dict,
            name: str,
            train_expansion_features: Optional[np.ndarray] = None,
            train_expansion_targets: Optional[np.ndarray] = None,
            eval_metric: str = "R^2",
            save_predictions: bool = False,
    ) -> float:
        """
        Evaluates the predictive performance of the model with the current settings / hyperparameters using a defined
        splitting scheme.

        Args:
             features: Numpy ndarray of the features of all data points.
             targets: Numpy ndarray of the corresponding targets for all data points.
             splitting_scheme: Name of the splitting scheme (must correspond to class name from sklearn.model_selection)
             splitting_scheme_params: Kwargs for the constructor of the splitting scheme.
             name: Base name of the current evaluation run.
             train_expansion_features: Ndarray of additional data points to be added to the training data (features).
             train_expansion_targets: Ndarray of additional data points to be added to the training data (targets).
             eval_metric: Name of the metric returned as a loss
             save_predictions: True if the predictions should be saved.

        Returns:
            float: Value of the metric specified in the eval_metric arg.
        """
        logging.info(f"Performing Model Evaluation Using {splitting_scheme}.")
        performance_metrics: list = []

        # Create folder if prediction should be saved
        if save_predictions:
            local_path: Path = self._output_dir / name
            local_path.mkdir(parents=True, exist_ok=True)

        # Create empty dummy arrays if no training data expansion is provided
        if train_expansion_features is None:
            train_expansion_features: np.ndarray = np.empty((0, features.shape[1]))
            train_expansion_targets: np.ndarray = np.empty((0, 1))

        # Instantiate splitter from sklearn
        splitter = eval(splitting_scheme)(**splitting_scheme_params, random_state=self._random_state)
        for i, (train, test) in enumerate(splitter.split(features)):

            # Merge train data with expansion data
            train_features = np.vstack((features[train, :], train_expansion_features))
            train_targets = np.vstack((targets[train, :], train_expansion_targets))

            # Train the model, obtain predictions for train and test data
            self.train(features=train_features, targets=train_targets)
            train_prediction, train_uncertainty = self.predict(features=train_features)
            test_prediction, test_uncertainty = self.predict(features=features[test, :])
            performance_metrics.append(
                {
                    "train": calculate_metrics(train_targets, train_prediction, train_uncertainty),
                    "test": calculate_metrics(targets[test, :], test_prediction, test_uncertainty)
                }
            )

            # If specified, save observed and predicted values as csv file
            if save_predictions:
                save_csv(
                    np.vstack((train_targets.flatten(), train_prediction.flatten(), train_uncertainty.flatten())).T,
                    colnames=["True Values", "Predicted Values", "Prediction Uncertainty"],
                    file_path=local_path / f"Train_{i}.csv"
                )
                save_csv(
                    np.vstack((targets[test, :].flatten(), test_prediction.flatten(), test_uncertainty.flatten())).T,
                    colnames=["True Values", "Predicted Values", "Prediction Uncertainty"],
                    file_path=local_path / f"Test_{i}.csv"
                )

        # Calculate average performance metrics and save them to _performance_metrics attribute
        average_metrics, performance_metric = calculate_average_metrics(performance_metrics, eval_metric)
        self._performance_metrics[name] = {
            "hyperparameters": self.hyperparameters,
            "all_metrics": performance_metrics,
            "average": average_metrics
        }
        if save_predictions:
            plot_predictions(local_path, "Test_*.csv", eval_metric, performance_metric)

        return performance_metric
