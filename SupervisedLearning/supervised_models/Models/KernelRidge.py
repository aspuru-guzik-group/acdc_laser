from typing import Tuple, Optional
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator
from ..SupervisedModel import SupervisedModel
from ..Utils import CategoricalTransformer
from scipy.spatial.distance import jaccard


class KernelRidgeClassifier(BaseEstimator):
    """
    Minimalistic implementation of a binary classifier based on sklearn.kernel_ridge.KernelRidge.
    """

    def __init__(self, *args, **kwargs):
        self._class_label_transformer = CategoricalTransformer(categories=(-1, 1), binary=True)
        self._regressor: KernelRidge = KernelRidge(*args, **kwargs)
        self._trained: bool = False
        self._unique_values: Optional[np.array] = None

    def fit(self, X: np.ndarray, y: np.array) -> None:
        """
        Trains the internal regressor on class labels (-1, 1).

        Args:
             X: Numpy ndarray (n_samples, n_features) of training features.
             y: Numpy 1D array (n_samples) of training class labels.
        """
        y: np.array = self._class_label_transformer.fit_transform(y)
        self._regressor.fit(X, y)
        self._trained = True

    def predict(self, X: np.ndarray) -> np.array:
        """
        Predicts class labels based on given features.

        Args:
            X: Numpy ndarray (n_samples, n_features) of data points to evaluate.

        Returns:
            np.array: Numpy 1D array (n_samples) of predicted class labels.

        Raises:
            ValueError: If the method is called before training the model.

        """
        if not self._trained:
            raise ValueError("The Classifier has not been trained yet.")

        probabilities = self._regressor.predict(X)
        probabilities[probabilities > 0] = 1
        probabilities[probabilities < 0] = -1

        return self._class_label_transformer.inverse_transform(probabilities)


class KernelRidgeModel(SupervisedModel):
    """
    Instance of the SupvervisedModel metaclass using KernelRidge models.
    """

    name = "KernelRidge"

    _available_models: dict = {
        "regression": KernelRidge,
        "classification": KernelRidgeClassifier
    }

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._available_models[self._prediction_type](**self._kwargs, **self.hyperparameters)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of the _predict method from the SupervisedModel metaclass.
        Returns uncertainties as array of np.nan, since there is no native uncertainty in KernelRidge.

        Args:
            features: Numpy ndarray (n_samples x n_features)

        Returns:
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted targets.
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted uncertainties (np.nan in this case).
        """
        predictions: np.ndarray = self._model.predict(features).reshape(-1, 1)
        uncertainties: np.ndarray = np.empty((features.shape[0],)).reshape(-1, 1)
        uncertainties[:] = np.nan  # no easy way to implement uncertainties for ridge regression, just returns nans
        return predictions, uncertainties


def tanimoto_distance(fp1: np.array, fp2: np.array, **kwargs) -> float:
    """
    Simple implementation of a Tanimoto Kernel for use with sklearn's KernelRidge models (following the concept
    described in GAUCHE, Rhys-Griffiths et al., 2023). Uses scipy's jaccard implementation to compute the Tanimoto
    distance between to bit vectors (fp1 and fp2).

    Args:
        fp1, fp2: Numpy 1D array of the fingerprint.

    Returns:
        float: Tanimoto similarity
    """
    return 1 - jaccard(fp1, fp2)
