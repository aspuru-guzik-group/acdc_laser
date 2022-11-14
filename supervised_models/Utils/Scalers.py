import numpy as np


class IdentityScaler(object):
    """
    Dummy feature scaler that returns unscaled features.
    To use as a scaler if no scaling is desired.
    """
    @staticmethod
    def fit_transform(values: np.ndarray) -> np.ndarray:
        return values

    @staticmethod
    def transform(values: np.ndarray) -> np.ndarray:
        return values


class FeatureNormalizer(object):
    """
    Scaler to normalize each feature by dividing all values through the maximum absolute value.
    Follows the same API syntax as the scalers provided by sklearn.preprocessing.
    """
    def __init__(self):
        self._max_values = None

    def fit(self, x: np.ndarray) -> None:
        self._max_values = np.max(abs(x), axis=0)
        self._max_values[self._max_values == 0.0] = 1.0  # No scalarization if all values are 0 for a certain feature.

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._max_values is None:
            raise AttributeError("The scaler has not been trained yet.")
        return x[:, ] / self._max_values

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
