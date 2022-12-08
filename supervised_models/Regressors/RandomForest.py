from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from supervised_models.SupervisedModel import SupervisedModel


class RandomForest(SupervisedModel):
    """
    Instance of the SupervisedModel metaclass using a Random Forest Regressor (as implemented in scikit-learn)
    as the supervised learning model.
    """

    name = "RandomForest"

    _available_models: dict = {
        "regression": RandomForestRegressor,
        "classification": RandomForestClassifier
    }

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._available_models[self._prediction_type](**self.hyperparameters, random_state=self._random_state, verbose=0)
        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions: np.ndarray = self._model.predict(features).reshape(-1, 1)
        variances: np.ndarray = np.var([estimator.predict(features) for estimator in self._model.estimators_], axis=0).reshape(-1, 1)
        return predictions, variances
