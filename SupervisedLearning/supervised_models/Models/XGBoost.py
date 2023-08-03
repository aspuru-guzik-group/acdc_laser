import numpy as np
from typing import Tuple
from xgboost import XGBRegressor, XGBClassifier
from supervised_models.SupervisedModel import SupervisedModel


class XGBoost(SupervisedModel):
    """
    Instance of the SupervisedModel metaclass using XGBoost as the Supervised Learning Model.
    """

    name = "XGBoost"

    _available_models = {
        "regression": XGBRegressor,
        "classification": XGBClassifier
    }

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._available_models[self._prediction_type](**self.hyperparameters, verbosity=0, random_state=self._random_state)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of the _predict method from the SupervisedModel metaclass.
        Returns uncertainties as array of np.nan, since there is no native uncertainty in XGBoost.

        Possible Extension for Uncertainty Quantification:
        https://towardsdatascience.com/regression-prediction-intervals-with-xgboost-428e0a018b
        Goes at the expense of being able to use custom loss functions.

        Args:
            features: Numpy ndarray (n_samples x n_features)

        Returns:
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted targets.
             np.ndarray: Numpy ndarray (n_samples x 1) of predicted uncertainties (np.nan in this case).
        """
        predictions: np.ndarray = self._model.predict(features).reshape(-1, 1)
        uncertainties: np.ndarray = np.empty((features.shape[0],)).reshape(-1, 1)
        uncertainties[:] = np.nan  # no easy way to implement uncertainties for XGBoost, just returns nans
        return predictions, uncertainties
