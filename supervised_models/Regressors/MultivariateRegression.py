from typing import Tuple, Optional
import numpy as np
from sklearn import linear_model, kernel_ridge
from scipy.spatial.distance import jaccard
from supervised_models.SupervisedModel import SupervisedModel
from rdkit import DataStructs


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
        self._model_type: type = self.models[self._kwargs.pop("regression_type")][self._kwargs.pop("regularization", None)]

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._model_type(**self._kwargs, **self.hyperparameters)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions: np.ndarray = self._model.predict(features)
        uncertainties: np.ndarray = np.empty((features.shape[0],))
        uncertainties[:] = np.nan  # no easy way to implement uncertainties for linear regression, just returns nans
        return predictions, uncertainties


def tanimoto_distance(fp1: np.array, fp2: np.array, **kwargs) -> float:
    """
    Computes the Tanimoto distance between to bit vectors (fp1 and fp2).

    Args:
        fp1, fp2: Numpy 1D array of the fingerprint.

    Returns:
        float: Tanimoto similarity
    """
    return 1 - jaccard(fp1, fp2)
