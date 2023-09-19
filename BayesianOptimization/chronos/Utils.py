import numpy as np
import torch
from . import DEVICE
import gc


class IdentityScaler(object):
    """
    Dummy feature scaler that returns unscaled features.
    To use as a scaler if no scaling is desired.
    Follows the sklearn scaler interface.
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


def clear_cache():
    """
    Clears the cache of GPU memory.
    """
    if DEVICE.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()