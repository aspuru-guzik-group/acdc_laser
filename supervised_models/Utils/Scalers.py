from typing import Callable
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler


class IdentityScaler(object):
    """
    Dummy feature scaler that returns unscaled features.
    To use as a scaler if no scaling is desired.
    """
    @staticmethod
    def fit(x):
        pass

    @staticmethod
    def transform(values: np.ndarray) -> np.ndarray:
        return values

    @staticmethod
    def inverse_transform(values: np.ndarray) -> np.ndarray:
        return values

    @staticmethod
    def fit_transform(values: np.ndarray) -> np.ndarray:
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

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x[:, ] * self._max_values

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class PowerTransformScaler(object):
    """
    Scaler using the PowerTransformer() from sklearn.
    Circumvents the numerical instability of the PowerTransformer() for small values by coupling it to a
    FeatureNormalizer.
    """
    def __init__(self):
        self._normalizer = FeatureNormalizer()
        self._powertransformer = PowerTransformer()

    def fit(self, x: np.ndarray) -> None:
        x_normalized: np.ndarray = self._normalizer.fit_transform(x)
        self._powertransformer.fit(x_normalized)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_normalized: np.ndarray = self._normalizer.transform(x)
        return self._powertransformer.transform(x_normalized)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x_power_inverted: np.ndarray = self._powertransformer.inverse_transform(x)
        return self._normalizer.inverse_transform(x_power_inverted)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class FunctionTransformScaler(object):
    """
    Scaler combining a FunctionTransformer from sklearn with a post-scaling of the target values.
    """
    def __init__(self, func: Callable, inverse_func: Callable, scaler: object = StandardScaler()):
        self._func_transformer = FunctionTransformer(func=func, inverse_func=inverse_func)
        self._scaler = scaler

    def fit(self, x: np.ndarray) -> None:
        x_transformed: np.ndarray = self._func_transformer.fit_transform(x)
        self._scaler.fit(x_transformed)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_transformed: np.ndarray = self._func_transformer.transform(x)
        return self._scaler.transform(x_transformed)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x_unscaled: np.ndarray = self._scaler.inverse_transform(x)
        return self._func_transformer.inverse_transform(x_unscaled)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
