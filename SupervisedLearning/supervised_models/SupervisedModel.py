import time
from abc import ABCMeta, abstractmethod
from typing import Any, Tuple, Optional, Dict, List, Union
from pathlib import Path
import logging
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *
from hyperopt import fmin, tpe, space_eval, STATUS_OK, early_stop

from .Utils import save_csv, save_json
from .Utils import calculate_regression_metrics, calculate_classification_metrics, calculate_average_metrics
from .Utils import IdentityScaler
from .Utils import ECFPSplitter


class SupervisedModel(metaclass=ABCMeta):

    """
    Metaclass for supervised learning models with uncertainty prediction.
    Provides wrappers for model (cross) validation and hyperparameter optimization.
    """

    name = ""

    _prediction_types: dict = {
        "regression": {
            "default_target_scaler": StandardScaler,
            "metrics": calculate_regression_metrics,
        },
        "classification": {
            "default_target_scaler": IdentityScaler,
            "metrics": calculate_classification_metrics,
        }
    }

    def __init__(
            self,
            prediction_type: str,
            output_dir: Path,
            n_tasks: int = 1,
            verbose: bool = False,
            hyperparameters_fixed: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            **kwargs
    ):
        """
        Instantiates the SupervisedModel object by setting a range of private attributes.

        Args:
            prediction_type: "regression", "classification"
            output_dir: Path to the directory where output data should be saved.
            n_tasks: Number of tasks for a possible multitask model (e.g. number of properties to predict).
            verbose: If True, prints additional information during model evaluation.
            hyperparameters_fixed: Dictionary of all model hyperparameters that should not be varied.
            hyperparameters: Dictionary of all model hyperparameters (name and hyperopt object).
            kwargs: Further keyword arguments
                        - "feature_scaler": Instance of a scaler (default: sklearn.preprocessing.StandardScaler).
                        - "target_scaler": Instance of a scaler (default: sklearn.preprocessing.StandardScaler).
                        - "random_state": Random state variable (default: 42).
                        - "verbose": If True, prints additional information to the console.
                        - kwargs to be passed to the constructor of the specific model.
        """
        self._prediction_type: str = prediction_type
        self._prediction_tools: dict = self._prediction_types[prediction_type]
        self._n_tasks: int = n_tasks
        self._model: Any = None
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._hp_options = hyperparameters
        self._hyperparameters_fixed: dict = hyperparameters_fixed if hyperparameters_fixed else {}
        self._current_hp = copy.deepcopy(self._hyperparameters_fixed)
        self._feature_scaler = kwargs.pop("feature_scaler", StandardScaler)()
        self._target_scaler = kwargs.pop("target_scaler", self._prediction_tools["default_target_scaler"])()
        self._performance_metrics: dict = {}
        self._random_state = kwargs.pop("random_state", 42)
        self._kwargs: dict = kwargs

        # Sets up the logger
        file_handler = logging.FileHandler(self._output_dir / f"Output.log")
        formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self._logger = logging.getLogger(time.strftime("%H%M%S%f", time.localtime()))
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(file_handler)
        if verbose:
            self._logger.setLevel(logging.DEBUG)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self._logger.addHandler(stream_handler)

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
        self._current_hp = copy.deepcopy(self._hyperparameters_fixed)

        for parameter, param_value in parameters.items():
            if parameter not in self._hp_options:
                self._logger.error(f"Error when setting the current hyperparamters. The parameter {parameter} is unknown!")
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
            self._logger.error(f"The hyperparameters {missing_hp} are undefined!")
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
            self._logger.error("The model has not been instantiated and trained.")
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
            targets: np.ndarray,
            train_test_splitting: type,
            train_test_splitting_params: dict,
            validation_splitting: Optional[type] = None,
            validation_splitting_params: Optional[dict] = None,
            validation_metric: str = "R^2",
            max_evaluations: int = 100,
            validation_target_index: Optional[int] = None,
            expansion_features: Optional[np.ndarray] = None,
            expansion_targets: Optional[np.ndarray] = None,
            smiles: Optional[np.array] = None
    ):
        """
        Main function to run supervised learning on a given dataset (passed as features and targets).

            1. Performs train-test splitting of the data, using the specified train-test splitting scheme.
            2. For each train-test split, performs nested hyperparameter optimization on the training data.
               (uses the optimize_hyperparameters method)
            3. Evaluates the performance of the optimized model on each train-test split.
               Saves the obtained predictions

        Args:
            features: Numpy ndarray of the features of all data points.
            targets: Numpy ndarray of the corresponding targets for all data points.
            train_test_splitting: Type of train–test splitter.
            train_test_splitting_params: Kwargs for the constructor of the train-test splitter.
            validation_splitting: Type of the splitter for validation / hyperparameter optimization.
            validation_splitting_params: Kwargs for the constructor of the validation splitter.
            validation_metric: Name of the metric used for validation / hyperparameter optimization.
            max_evaluations: Maximum number of evaluations for validation.
            validation_target_index: Index of the target to be used for calculating the validation metrics.
                                     If None, the mean of the metrics for all targets is used.
            expansion_features: Ndarray of additional data points to be added to the training data (features).
            expansion_targets: Ndarray of additional data points to be added to the training data (targets).
            smiles: 1D Numpy array of SMILES for all data points (only required for ECFPSplitter).
        """
        if train_test_splitting == ECFPSplitter and smiles is not None:
            train_test_splitting_params["molecule_smiles"] = smiles.flatten()

        train_test_splitter = train_test_splitting(**train_test_splitting_params, random_state=self._random_state)
        split_indices: list = [split for split in train_test_splitter.split(features)]

        for i, (train, test) in enumerate(split_indices):

            self._logger.info(f"Train-Test Split {i + 1}/{len(split_indices)}")

            self.optimize_hyperparameters(
                features=features[train, :],
                targets=targets[train, :],
                n_targets=self._n_tasks,
                validation_splitting=validation_splitting,
                validation_splitting_params=validation_splitting_params,
                outer_split_id=f"Train_{i}_Validation",
                eval_metric=validation_metric,
                eval_target_index=validation_target_index,
                max_evaluations=max_evaluations,
                expansion_features=expansion_features,
                expansion_targets=expansion_targets,
                smiles=smiles[train] if smiles is not None else None
            )

            if test is None:
                continue

            data_dir: Path = self._output_dir / "Test"
            data_dir.mkdir(parents=True, exist_ok=True)

            test_metrics = self.evaluate_single_split(
                train_features=features[train, :],
                train_targets=targets[train, :],
                test_features=features[test, :],
                test_targets=targets[test, :],
                n_targets=self._n_tasks,
                data_dir=data_dir,
                split_id=i
            )
            self._save_metrics(f"Test_{i}", test_metrics)

        all_metrics: list = [self._performance_metrics.get(f"Test_{i}", {}).get("all_metrics") for i in range(len(split_indices))]
        average_metrics, _ = calculate_average_metrics(all_metrics, validation_metric, self._n_tasks, validation_target_index)
        self._save_metrics("Test", {}, "All_Results", average_metrics=average_metrics)

    def optimize_hyperparameters(
            self,
            features: np.ndarray,
            targets: np.array,
            n_targets: int,
            validation_splitting: Optional[type],
            validation_splitting_params: Optional[dict],
            outer_split_id: Union[str, int],
            eval_metric: str = "R^2",
            eval_target_index: Optional[int] = None,
            max_evaluations: int = 100,
            expansion_features: Optional[np.ndarray] = None,
            expansion_targets: Optional[np.ndarray] = None,
            smiles: Optional[np.array] = None
    ) -> None:
        """
        Runs the model by performing hyperparameter optimization following the defined train-test splitting scheme.
        Evaluates the performance of the optimized model and saves the obtained predictions.

        Args:
            features: Numpy ndarray of the features of all data points.
            targets: Numpy ndarray of the corresponding targets for all data points.
            n_targets: Number of targets to be predicted.
            validation_splitting: Name of the splitting scheme (must correspond to class name from sklearn.model_selection)
            validation_splitting_params: Kwargs for the constructor of the splitting scheme.
            outer_split_id: Name of the outer split for which hyperparameters should be optimized.
            eval_metric: Name of the metric to be used as the loss function for hyperparameter optimization.
            eval_target_index: Index of the target to be used for calculating the validation metrics.
                               If None, the mean of the metrics for all targets is used.
            max_evaluations: Maximum number of evaluations for hyperparameter optimiazation.
            expansion_features: Ndarray of additional data points to be added to the training data (features).
            expansion_targets: Ndarray of additional data points to be added to the training data (targets).
            smiles: 1D Numpy array of SMILES for all data points (only required for ECFPSplitter).
        """
        if validation_splitting is None or not self._hp_options:
            return

        if validation_splitting == ECFPSplitter and smiles is not None:
            validation_splitting_params["molecule_smiles"] = smiles
        validation_splitter = validation_splitting(**validation_splitting_params, random_state=self._random_state)
        validation_splits: list = [split for split in validation_splitter.split(features)]

        if self._hp_options:

            def objective(hyperparameters):
                """Callable object to be passed to the hyperopt fmin function."""
                self.hyperparameters = hyperparameters
                loss = self.run_validation(
                    features,
                    targets,
                    n_targets=n_targets,
                    split_indices=validation_splits,
                    outer_split_id=outer_split_id,
                    name=time.strftime("%H%M%S", time.localtime()),
                    eval_metric=eval_metric,
                    eval_target_index=eval_target_index,
                    expansion_features=expansion_features,
                    expansion_targets=expansion_targets
                )
                self._logger.info(f"Hyperparameter optimization round completed -> Score = {loss}.")
                return {"loss": -loss, "status": STATUS_OK}

            optimal_params = fmin(
                objective,
                space=self._hp_options,
                algo=tpe.suggest,
                max_evals=max_evaluations,
                early_stop_fn=early_stop.no_progress_loss(max(10, int(0.2*max_evaluations))),
                show_progressbar=False
            )
            # stops if no improvement is observed for 20% of all iterations (min. 10)

            self.hyperparameters = space_eval(self._hp_options, optimal_params)
            self._logger.info("Hyperparameter optimization finished.")

        self.run_validation(
            features,
            targets,
            n_targets=n_targets,
            split_indices=validation_splits,
            outer_split_id=outer_split_id,
            name="Final_Model",
            eval_metric=eval_metric,
            eval_target_index=eval_target_index,
            expansion_features=expansion_features,
            expansion_targets=expansion_targets,
            save_predictions=True
        )

    def run_validation(
            self,
            features: np.ndarray,
            targets: np.array,
            n_targets: int,
            split_indices: List[Tuple[np.array, np.array]],
            outer_split_id: Union[str, int],
            name: str,
            eval_metric: str = "R^2",
            eval_target_index: Optional[int] = None,
            expansion_features: Optional[np.ndarray] = None,
            expansion_targets: Optional[np.ndarray] = None,
            save_predictions: bool = False,
    ) -> float:
        """
        Evaluates the predictive performance of the model with the current settings / hyperparameters using a defined
        splitting scheme.

        Args:
             features: Numpy ndarray of the features of all data points.
             targets: Numpy ndarray of the corresponding targets for all data points.
             n_targets: Number of targets to be predicted.
             split_indices: List of split indices (train, validation) for all validation splits.
             outer_split_id: Identifier of the outer train/test split.
             name: Base name of the current evaluation run.
             expansion_features: Ndarray of additional data points to be added to the training data (features).
             expansion_targets: Ndarray of additional data points to be added to the training data (targets).
             eval_metric: Name of the metric returned as a loss
             eval_target_index: Index of the target to be used for calculating the validation metrics.
                                If None, the mean of the metrics for all targets is used.
             save_predictions: True if the predictions should be saved.

        Returns:
            float: Value of the metric specified in the eval_metric arg.
        """
        self._logger.debug(f"Evaluating model performance with hyperparameters {self.hyperparameters}.")

        performance_metrics: list = []

        # Create folder where prediction should be saved
        if save_predictions:
            local_path: Path = self._output_dir / name
            local_path.mkdir(parents=True, exist_ok=True)

        # Create empty dummy arrays if no training data expansion is provided
        if expansion_features is None:
            expansion_features: np.ndarray = np.empty((0, features.shape[1]))
            expansion_targets: np.ndarray = np.empty((0, targets.shape[1]))

        # Iterate over all splits
        for i, (train, validation) in enumerate(split_indices):

            # Merge train data with expansion data
            train_features = np.vstack((features[train, :], expansion_features))
            train_targets = np.vstack((targets[train, :], expansion_targets))

            # Train the model, obtain predictions for train and test data
            metrics = self.evaluate_single_split(
                train_features=train_features,
                train_targets=train_targets,
                test_features=features[validation, :],
                test_targets=targets[validation, :],
                n_targets = n_targets,
                data_dir=local_path if save_predictions else None,
                split_id=i
            )
            performance_metrics.append(metrics)

        # Calculate average performance metrics and save them to _performance_metrics attribute
        average_metrics, performance_metric = calculate_average_metrics(performance_metrics, eval_metric, n_targets, eval_target_index)
        self._save_metrics(outer_split_id, performance_metrics, f"Validation_{name}", average_metrics)

        self._logger.debug(f"Model evaluation completed. Average Score: {performance_metric}.")

        return performance_metric

    def evaluate_single_split(
            self,
            train_features: np.ndarray,
            train_targets: np.ndarray,
            test_features: np.ndarray,
            test_targets: np.ndarray,
            n_targets: int,
            data_dir: Optional[Path] = None,
            split_id: Optional[Any] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Builds a model based on the training data and evaluates it on the test data

        Args:
            train_features: Features of the training data (n_datapoints x n_features)
            train_targets: Targets of the training data (n_datapoints x 1)
            test_features: Features of the test data (n_datapoints x n_features)
            test_targets: Targets of the test data (n_datapoints x n_features).
            n_targets: Number of targets to be predicted.
            data_dir: Path to the folder where the prediction should be saved.
            split_id: Identifier of the train-test split (to name the saved file).

        Returns:
            Dict[str, Dict[str, float]]: Dictionary of prediction performance metrics (separate for "train" and "test).
        """
        try:
            self.train(features=train_features, targets=train_targets)
            train_prediction, train_uncertainty = self.predict(features=train_features)
            test_prediction, test_uncertainty = self.predict(features=test_features)

        except RuntimeError as e:
            # If the model crashes during training, return empty metrics
            self._logger.error(f"Runtime error during model training: {e}")
            return {
                "train": {},
                "test": {}
            }

        metrics: dict = {
            "train": self._prediction_tools["metrics"](train_targets, train_prediction, train_uncertainty, n_targets),
            "test": self._prediction_tools["metrics"](test_targets, test_prediction, test_uncertainty, n_targets)
        }

        if data_dir is not None and split_id is not None:
            self._save_prediction(train_targets, train_prediction, train_uncertainty, data_dir / f"Train_{split_id}.csv")
            self._save_prediction(test_targets, test_prediction, test_uncertainty, data_dir / f"Test_{split_id}.csv")

        return metrics

    @staticmethod
    def _save_prediction(
            true_targets: np.ndarray,
            predicted_targets: np.ndarray,
            predicted_uncertainty: np.ndarray,
            file_name: Path
    ):
        """
        Saves the true and predicted values into a csv file.

        Args:
            true_targets
            predicted_targets
            predicted_uncertainty
            file_name: Path to the .csv file
        """
        for i in range(true_targets.shape[1]):
            file = file_name.parent / f"{file_name.stem}_{i}{file_name.suffix}"

            save_csv(
                np.vstack((true_targets[:, i].flatten(), predicted_targets[:, i].flatten(), predicted_uncertainty[:, i].flatten())).T,
                colnames=["True Values", "Predicted Values", "Prediction Uncertainty"],
                file_path=file
            )

    def _save_metrics(
            self,
            outer_split_id: Union[int, str],
            all_metrics: Union[dict, list],
            inner_split_id: Optional[Union[int, str]] = None,
            average_metrics: Optional[dict] = None
    ) -> None:
        """
        Saves all metrics into the _performance_metrics dict and updates the output json file (logging of the progress
        in case of failure).

        Args:
            outer_split_id: Identifier of the outer (train-test) split.
            all_metrics: Dictionary of all metrics
            inner_split_id: Identifier of the inner (validation) split.
            average_metrics: Dictionary of average performance metrics
        """

        if outer_split_id not in self._performance_metrics:
            self._performance_metrics[outer_split_id] = dict()

        if inner_split_id is not None:
            self._performance_metrics[outer_split_id][inner_split_id] = {
                "hyperparameters": self.hyperparameters,
                "all_metrics": all_metrics,
                "average": average_metrics
            }

        else:
            self._performance_metrics[outer_split_id] = {
                "hyperparameters": self.hyperparameters,
                "all_metrics": all_metrics
            }

        save_json(self._performance_metrics, self._output_dir / "hyperparameter_optimization.json")
