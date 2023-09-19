from typing import List, Tuple, Type, Optional
from pathlib import Path
import logging
import numpy as np
import time
import torch
from sklearn.preprocessing import StandardScaler
from .SurrogateModels import SurrogateModel
from .EnsembleAcquisition import DQEnsembleAcquisition
from .AcquisitionFunctions import DiscreteQAcquisitionFunction
from .Utils import IdentityScaler
from . import DEVICE


class DiscreteGridOptimizer(object):
    """
    Implementation of a Bayesian Optimizer for optimization in a space of discrete data points.

    Args:
        data_dir: Path to the directory where the data is stored.
        surrogate_model: SurrogateModel type. Must implement a train_model() and a posterior() method.
        surrogate_params: Dictionary of parameters for the surrogate model.
        iteration_number: Number of the current iteration.
        acquisition_functions: List of DiscreteQAcquisitionFunction types to be used in the portfolio.
        acquisition_function_params: List of dictionaries of parameters for the acquisition functions.
        feature_scaler: Scaler type for scaling the feature values (following the scikit-learn API).
        target_scaler: Scaler type for scaling the target values (following the scikit-learn API).
        **kwargs: Additional keyword arguments to be passed to the acquisition function ensemble.

    Attributes:
        surrogate_model: SurrogateModel object.
    """

    def __init__(
            self,
            data_dir: Path,
            surrogate_model: Type[SurrogateModel],
            surrogate_params: dict = None,
            iteration_number: int = 0,
            acquisition_functions: List[Type[DiscreteQAcquisitionFunction]] = None,
            acquisition_function_params: List[dict] = None,
            feature_scaler: type = IdentityScaler,
            target_scaler: type = StandardScaler,
            logger=None,
            **kwargs
    ):
        self._data_dir = data_dir

        self.surrogate_model = None
        self._surrogate_type = surrogate_model
        self._surrogate_hyperparameters = surrogate_params if surrogate_params is not None else {}

        self._acquisition_functions = acquisition_functions if acquisition_functions is not None else []
        self._acquisition_parameters = acquisition_function_params if acquisition_function_params is not None else [{} for _ in acquisition_functions]
        self._best_f = -np.inf
        self._kwargs = kwargs

        self._iteration_number = iteration_number

        self._feature_scaler = feature_scaler()
        self._target_scaling = target_scaler()

        self._logger = logger if logger is not None else logging.getLogger()
        self._logger.info(f"Instantiated DiscreteGridOptimizer with a {surrogate_model.name} surrogate "
                          f"and an ensemble of {len(acquisition_functions)} acquisition functions on {DEVICE}.\n")

    def __call__(
            self,
            observations_features: np.ndarray,
            observations_targets: np.ndarray,
            objective_index: Optional[int],
            search_space_features: np.ndarray,
            pending_data_features: Optional[np.ndarray] = None,
            batch_size: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs a single optimization step. Scales features and targets, trains the surrogate model and returns the indices
        of the recommended data points.

        Args:
            observations_features: Numpy ndarray (n_observations x n_features) of all features of observations.
            observations_targets: Numpy ndarray (n_observations x 1) of all target values of the observations.
            objective_index: Index of the objective to be optimized.
            search_space_features: Numpy ndarray (n_search_space_points x n_features) of the full search space.
            pending_data_features: Numpy ndarray (n_pending_data_points x n_features) of all pending data points.
            batch_size: Number of data points to recommend.

        Returns:
            np.ndarray: Indices (batch_size, ) of all recommended data points.
            np.ndarray: Acquisition function values (batch_size, ) of all recommended data points.
        """
        search_space_features_scaled = self._feature_scaler.fit_transform(search_space_features)
        obs_features_scaled = self._feature_scaler.transform(observations_features)
        obs_targets_scaled = self._target_scaling.fit_transform(observations_targets)

        if pending_data_features is not None:
            if pending_data_features.size > 0:
                pending_data_features_scaled = self._feature_scaler.transform(pending_data_features)
            else:
                pending_data_features_scaled = np.empty((0, search_space_features.shape[1]))
        else:
            pending_data_features_scaled = np.empty((0, search_space_features.shape[1]))

        self._best_f = np.nanmax(obs_targets_scaled[:, objective_index])

        self._train_surrogate(obs_features_scaled, obs_targets_scaled)

        return self._get_recommendations(
            objective_index,
            search_space_features_scaled,
            pending_data_features_scaled,
            batch_size
        )

    def _train_surrogate(
            self,
            observations_features: np.ndarray,
            observations_targets: np.ndarray,
    ) -> None:
        """
        Train the surrogate model.

        Args:
            observations_features: Numpy ndarray (n_observations x n_features) of all features of observations.
            observations_targets: Numpy ndarray (n_observations x n_targets) of all target values of the observations.
        """
        self._logger.info(f"Training surrogate model on {observations_features.shape[0]} observations "
                          f"with {observations_targets.shape[1]} targets.")

        start_time = time.time()

        train_x, train_y = torch.from_numpy(observations_features), torch.from_numpy(observations_targets)

        self.surrogate_model = self._surrogate_type(
            train_features=train_x,
            train_targets=train_y,
            **self._surrogate_hyperparameters
        )

        self.surrogate_model.train_model()

        self._logger.info(f"Surrogate model training completed after {round(time.time() - start_time)} sec.\n")

    def _get_recommendations(
            self,
            objective_index: Optional[int],
            search_space_features: np.ndarray,
            pending_data_features: np.ndarray,
            batch_size: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the indices of the recommended data points.

        Args:
            objective_index: Index of the objective to be optimized.
            search_space_features: Numpy ndarray (n_search_space_points x n_features) of the full search space.
            batch_size: Number of data points to recommend.

        Returns:
            np.ndarray: Indices (batch_size, ) of all recommended data points.
            np.ndarray: Acquisition function values (batch_size, ) of all recommended data points.
        """
        test_x = torch.from_numpy(search_space_features)
        pending_x = torch.from_numpy(pending_data_features)

        acquisition_function = DQEnsembleAcquisition(
            iteration=self._iteration_number,
            directory=self._data_dir,
            model=self.surrogate_model,
            acquisition_functions=self._acquisition_functions,
            parameters=self._acquisition_parameters,
            objective_index=objective_index,
            best_f=self._best_f,
            logger=self._logger,
            **self._kwargs
        )

        optimal_indices, acqf_values = acquisition_function.optimize_discrete(
            test_x=test_x,
            pending_x=pending_x,
            q=batch_size
        )

        # sort the indices by the acqf values (descending)
        optimal_indices = optimal_indices[torch.argsort(acqf_values, descending=True)]
        acqf_values = acqf_values[torch.argsort(acqf_values, descending=True)]

        return optimal_indices.cpu().detach().numpy().flatten(), acqf_values.cpu().detach().numpy().flatten()
