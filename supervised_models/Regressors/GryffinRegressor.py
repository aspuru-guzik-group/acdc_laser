from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
from supervised_models.SupervisedModel import SupervisedModel
from gryffin import Gryffin


class GryffinRegressor(SupervisedModel):

    name = "gryffin"

    def __init__(
            self,
            output_dir: Path,
            hyperparameters: dict,
            gryffin_config: dict,
            all_data_points: np.ndarray,
            descriptors: Optional[dict] = None,
    ):
        super().__init__(output_dir, hyperparameters)
        self.gryffin_config = gryffin_config

        if descriptors:
            self.descriptors = descriptors
        else:
            self.descriptors = dict()

        self.parameters = self._prepare_parameters(all_data_points)

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Instantiates Gryffin in self._model, and trains the surrogate on all available observations.

        Args:
            features: Categorical options belonging to each data point.
            targets: 1D array of targets.
        """
        observations: List[dict] = self._prepare_observations(features, targets)

        self._model = Gryffin(
            config_dict={
                "general": self.gryffin_config,
                "parameters": self.parameters,
                "objectives": [{"name": "obj", "goal": "max"}]
            }
        )

        self._model.build_surrogate(observations)

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uses Gryffin as a regressor to predict the value for each unknown observation.
        Attention: Does only return dummy uncertainties (0 for each data point).

        Args:
            features: Categorical options belonging to each test datapoint.

        Returns:
            np.ndarray: Predicted values for each observation.
            np.ndarray: Zero array (dummy array)
        """
        test_observations: List[dict] = self._prepare_observations(features)
        predictions = self._model.get_regression_surrogate(test_observations)

        return predictions, np.zeros(predictions.shape[0])

    @staticmethod
    def _prepare_observations(data_points: np.ndarray, targets: Optional[np.array] = None) -> List[dict]:
        """
        Prepares the observations in the format required by Gryffin.

        Args:
            data_points: Categorical options belonging to each data point.
            targets: (optional) Numpy 1D array of target values.

        Returns:
            List[dict]: Observations as required by Gryffin.
        """
        observations = []
        for i in range(data_points.shape[0]):
            data = {f"x_{j}": data_points[i, j] for j in range(data_points.shape[1])}
            if targets is not None:
                data["obj"] = targets[i]
            observations.append(data)

        return observations

    def _prepare_parameters(self, all_data_points: np.ndarray) -> List[dict]:
        """
        Prepares the parameter space in the format required from Gryffin.

        Args:
            all_data_points: Categorical options belonging to each data point.

        Returns:
            List[dict]: Parameters as required by Gryffin.
        """
        parameters = [
            {
                "name": f"x_{i}",
                "type": "categorical",
                "options": list({all_data_points[j, i] for j in range(all_data_points.shape[0])}),
                "category_details": {all_data_points[j, i]: self.descriptors.get(all_data_points[j, i]) for j in range(all_data_points.shape[0])}
            }
            for i in range(all_data_points.shape[1])
        ]

        return parameters


