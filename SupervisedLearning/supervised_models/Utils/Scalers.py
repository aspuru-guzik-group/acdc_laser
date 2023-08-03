from typing import Callable, Iterable, Optional, Union
import numpy as np
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, StandardScaler, QuantileTransformer


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


class QuantileScaler(object):
    """
    Performs a quantile scaling of features into n quantiles.
    Coupled to a FeatureNormalizer to circumvent numerical instability of the QuantileTransformer for small values.
    """
    def __init__(self, n_quantiles: int = 20, output_distribution: str = "normal"):
        self._feature_normalizer = FeatureNormalizer()
        self._quantile_scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=output_distribution)

    def fit(self, x: np.ndarray) -> None:
        x_standardized: np.ndarray = self._feature_normalizer.fit_transform(x)
        self._quantile_scaler.fit(x_standardized)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_standardized: np.ndarray = self._feature_normalizer.transform(x)
        return self._quantile_scaler.transform(x_standardized)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x_unscaled: np.ndarray = self._quantile_scaler.inverse_transform(x)
        return self._feature_normalizer.inverse_transform(x_unscaled)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self. transform(x)


class CategoricalTransformer(object):

    def __init__(self, categories: Iterable, binary: bool = False):
        self._categories = categories
        self._unique_values: Optional[np.array] = None
        self._binary = binary

    def fit(self, x: np.array):
        """
        "Fits" the categorical transformer by identifying unique values in the data and assigning them to the
        _unique values attribute. Matching to category names is done by index.
        """
        self._unique_values: np.array = np.unique(x)

        if self._binary is True and len(self._unique_values) > 2:
            raise ValueError(f"{len(x)}  classes were found for a binary classification task.")
        if len(self._unique_values) > len(self._categories):
            raise ValueError("There are more class options than class labels given.")

    def transform(self, x: np.array) -> np.array:
        """
        Transforms original class labels to the categories defined in the _categories attribute.

        Args:
            x: Numpy array (n_samples) of original binary class labels.

        Returns:
             np.array: Numpy array (n_samples) of transformed class labels.
        """
        if self._unique_values is None:
            raise ValueError("The transformer has not been trained yet.")

        x_new = np.empty(len(x), dtype="int32")  # TODO: make data type attribute

        for original_value, new_value in zip(self._unique_values, self._categories):
            x_new[x == original_value] = new_value

        return x_new

    def fit_transform(self, x: np.array) -> np.array:
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x: np.array) -> np.array:
        """
        Transforms the predicted class labels (from self._categories) back to original class labels.

        Args:
             x: Numpy array (n_samples) of predicted class labels

        Returns:
            np.array: Numpy array (n_samples) of predicted original class labels.
        """
        for predicted_value, original_label in zip(self._categories, self._unique_values):
            x[x == predicted_value] = original_label

        return x