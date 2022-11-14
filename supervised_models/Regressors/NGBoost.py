import numpy as np
from typing import Tuple
from ngboost import NGBRegressor
from supervised_models.SupervisedModel import SupervisedModel


class NGBoost(SupervisedModel):

    """
    Instance of the SupervisedModel metaclass using an NGBRegressor as the Supervised Learning Model.
    """

    name = "NGBoost"

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = NGBRegressor(**self.hyperparameters, verbose=False)
        self._model.fit(features, targets)

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = self._model.pred_dist(features)
        return predictions.loc, predictions.var
