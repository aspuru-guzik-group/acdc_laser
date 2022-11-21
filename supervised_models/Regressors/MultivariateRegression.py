import copy
from typing import Tuple
import numpy as np
from sklearn import linear_model, kernel_ridge
from supervised_models.SupervisedModel import SupervisedModel


class MultivariateRegression(SupervisedModel):
    """
    Instance of the SupvervisedModel metaclass using multivariate regression approaches (including optional Kernel
    distance metrics and regularization / feature normalization approaches).
    """

    name = "MultivariateRegression"

    models: dict = {
        "linear":
            {
                None: linear_model.LinearRegression,
                "ridge": linear_model.Ridge,
                "lasso": linear_model.Lasso
            },
        "kernel":
            {
                "ridge": kernel_ridge.KernelRidge
            }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_type: str = self._kwargs.pop("regression_type")

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Selects the model based on the regularization type (passed either as a fixed value in self._kwargs, or as
        an optimizable hyperparameter in self.hyperparameters). Trains the model to fit the targets.

        Args:
            features: Numpy ndarray (n_samples x n_features)
            targets: Numpy ndarray (n_samples x 1)
        """
        hyperparameters = copy.deepcopy(self.hyperparameters)
        kwargs = copy.deepcopy(self._kwargs)
        try:
            regularization: str = kwargs.pop("regularization")
        except KeyError:
            regularization: str = hyperparameters.pop("regularization", None)

        self._model = self.models[self._model_type][regularization](**kwargs, **hyperparameters)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions: np.ndarray = self._model.predict(features).reshape(-1, 1)
        uncertainties: np.ndarray = np.empty((features.shape[0],)).reshape(-1, 1)
        uncertainties[:] = np.nan  # no easy way to implement uncertainties for linear regression, just returns nans
        return predictions, uncertainties
