import numpy as np
from pathlib import Path
from typing import Tuple, Callable, Optional
import torch
import gpytorch.models
import gpytorch.constraints
from torch import Tensor

from supervised_models.SupervisedModel import SupervisedModel


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.backends.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

torch.set_default_dtype(torch.float64)


class ExactGPModel(gpytorch.models.ExactGP):

    """
    Implementation of a formally exact Gaussian Process model with a given mean and covariance model.
    """
    def __init__(
            self,
            train_features: Tensor,
            train_targets: Tensor,
            likelihood: Callable,
            kernel_type: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
            dist_constraint: Optional[float] = None,
            **kwargs
    ):
        """
        Instantiates the super class (gpytorch.models.ExactGP) and sets a constant mean and an RBFKernel as the
        mean and covariance modules, respectively.

        Args:
            train_features: Tensor of all training data points and features (n_datapoints x n_features)
            train_targets: Tensor of all training data points and targets (n_datapoints x 1)
            likelihood: Likelihood function
        """
        super(ExactGPModel, self).__init__(train_features, train_targets, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        n_features = train_features.size(dim=1)

        if not dist_constraint:
            constraints = gpytorch.constraints.Positive()
        else:
            constraints = gpytorch.constraints.Interval(1/dist_constraint, dist_constraint)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel_type(
                ard_num_dims=n_features,
                lengthscale_constraint=constraints,
                **kwargs
            )
        )

    def forward(self, x: Tensor):
        """
        Implementation of the forward method required for model training and predictions.
        Computes the mean and covariance from the given input tensor, and returns a multivariate Normal distribution
        with the given mean and covariance.

        Args:
            x: Tensor of features passed.

        Returns:
            gpytorch.distributions.MultivariateNormal: A multivariate normal distribution with the given means and
                                                       covariances.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess(SupervisedModel):

    """
    Instance of the SupervisedModel metaclass using a GaussianProcess Regressor (from gpytorch) as the Supervised
    Learning Model.
    """

    name = "GaussianProcess"

    def __init__(
            self,
            output_dir: Path,
            hyperparameters_fixed: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            likelihood=gpytorch.likelihoods.GaussianLikelihood,
            **kwargs
    ):
        super().__init__(
            output_dir,
            hyperparameters_fixed=hyperparameters_fixed,
            hyperparameters=hyperparameters,
            likelihood=likelihood,
            **kwargs
        )
        self.likelihood = likelihood()

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the Gaussian Process model on the passed input data.

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of all input features.
            targets: Numpy ndarray (n_datapoints x 1) of all input targets.
        """
        training_data_x = torch.tensor(features)  # torch.from_numpy(features)
        training_data_y = torch.tensor(targets.flatten())  # torch.from_numpy(targets)

        # Instantiation of the Likelihood Function and the GP Model
        self._model = ExactGPModel(
            training_data_x,
            training_data_y,
            self.likelihood,
            **self.hyperparameters
        )

        # Setting the Model and the Likelihood to Training Mode
        self._model.train()
        self.likelihood.train()

        # Optimization
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hyperparameters.get("learning_rate", 1E-3))
        loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._model)

        for i in range(self.hyperparameters.get("training_iterations", 1E4)):
            optimizer.zero_grad()
            output = self._model(training_data_x)
            loss_value = -loss(output, training_data_y)
            loss_value.mean().backward()
            optimizer.step()

    def _predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs the prediction of targets based on a set of passed features. Predicts mean and variance for each
        data point.

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of test data.

        Returns:
            np.ndarray: Predicted means for all data points
            np.ndarray: Predicted variances for all data points.
        """
        self._model.eval()
        self.likelihood.eval()

        test_data_x = torch.from_numpy(features)
        prediction = self.likelihood(self._model(test_data_x))
        return prediction.mean.detach().numpy(), prediction.variance.detach().numpy()
