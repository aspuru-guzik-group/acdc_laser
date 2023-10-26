import copy
from typing import Tuple
import numpy as np
from sklearn import linear_model
from ..SupervisedModel import SupervisedModel


class LinearModel(SupervisedModel):
    """
    Instance of the SupvervisedModel metaclass using multivariate regression approaches (including optional Kernel
    distance metrics and regularization / feature normalization approaches).
    """

    name = "LinearModel"

    _available_models: dict = {
        "regression": {
            None: linear_model.LinearRegression,
            "ridge": linear_model.Ridge,
            "lasso": linear_model.Lasso,
            "elasticnet": linear_model.ElasticNet
        },
        "classification": {
            None: lambda *args, **kwargs: linear_model.LogisticRegression(penalty="none", *args, **kwargs),
            "ridge": lambda *args, **kwargs: linear_model.LogisticRegression(penalty="l2", *args, **kwargs),
            "lasso": lambda *args, **kwargs: linear_model.LogisticRegression(penalty="l1", solver="saga", *args, **kwargs),
            "elasticnet": lambda *args, **kwargs: linear_model.LogisticRegression(penalty="elasticnet", solver="saga", *args, **kwargs)
        }
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self._model = self._available_models[self._prediction_type][regularization](**kwargs, **hyperparameters)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of the _predict method from the SupervisedModel metaclass.
        Returns uncertainties as array of np.nan, since there is no native uncertainty in linear regression.

        Args:
            features: Numpy ndarray (n_samples x n_features)

        Returns:
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted targets.
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted uncertainties (np.nan in this case).
        """
        predictions: np.ndarray = self._model.predict(features).reshape(-1, 1)
        uncertainties: np.ndarray = np.empty((features.shape[0],)).reshape(-1, 1)
        uncertainties[:] = np.nan  # no easy way to implement uncertainties for linear regression, just returns nans
        # TODO: For logistic regression, there is some kind of uncertainty estimate –> implement it here?
        return predictions, uncertainties
