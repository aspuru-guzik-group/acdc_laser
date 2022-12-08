import numpy as np
from typing import Tuple
from ngboost import NGBRegressor, NGBClassifier
from supervised_models.SupervisedModel import SupervisedModel
from supervised_models.Utils import CategoricalTransformer


class NGBoost(SupervisedModel):
    """
    Instance of the SupervisedModel metaclass using an NGBRegressor as the Supervised Learning Model.
    """

    name = "NGBoost"

    _available_models = {
        "regression": NGBRegressor,
        "classification": NGBClassifier
    }

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        self._model = self._available_models[self._prediction_type](**self.hyperparameters, verbose=False)

        # Currently, this is a little hacky workaround for the fact that the NGBClassifier can only work with integer-
        # type category labels. Should be cleaned up at some point (factory pattern or two separate subclasses for
        # regression and classification)
        if self._prediction_type == "classification":
            self._category_scaler = CategoricalTransformer(categories=range(len(np.unique(targets.flatten()))))
            targets = self._category_scaler.fit_transform(targets.flatten())

        self._model.fit(features, targets.flatten())

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self._prediction_type == "classification":
            class_labels: np.array = self._model.predict(features)
            uncertainties: np.array = np.sum(self._model.predict_proba(features), axis=1)
            ret: tuple = class_labels, uncertainties
        else:
            predictions = self._model.pred_dist(features)
            ret: tuple = predictions.loc.reshape(-1, 1), predictions.var.reshape(-1, 1)

        return ret
