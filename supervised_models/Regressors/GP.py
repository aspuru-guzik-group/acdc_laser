import numpy as np
from pathlib import Path
from typing import Tuple, Callable, Optional
import torch
import gpytorch.models
import gpytorch.constraints
from torch import Tensor

from supervised_models.SupervisedModel import SupervisedModel
from supervised_models.Utils import CategoricalTransformer


if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.set_default_dtype(torch.float64)

elif torch.backends.mps.is_available():
    DEVICE = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    # ATTN: This is just a temporary fix until PyTorch includes the update for https://github.com/pytorch/pytorch/pull/88542
    # DEVICE = torch.device("mps")
    # torch.set_default_dtype(torch.float32)

else:
    DEVICE = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

print(f"PyTorch / GPyTorch Using {DEVICE}.")


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Implementation of a formally exact Gaussian Process regressor with a given mean and covariance model.
    """
    def __init__(
            self,
            prediction_type: str,
            train_features: Tensor,
            train_targets: Tensor,
            likelihood: Callable,
            kernel_type: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
            dist_constraint: Optional[float] = None,
            n_classes: Optional[int] = None,
            **kwargs
    ):
        """
        Instantiates the super class (gpytorch.models.ExactGP) and sets a constant mean and a given kernel as the
        mean and covariance modules, respectively.

        Args:
            train_features: Tensor of all training data points and features (n_datapoints x n_features)
            train_targets: Tensor of all training data points and targets (n_datapoints x 1)
            likelihood: Likelihood function
            kernel_type: Kernel from gpytorch.kernels
            dist_constraint (optional): Distance constraint for the kernel implementation
            n_classes (optional): Number of classes (for classification, ignored if prediction_type == "regression")
        """
        super(ExactGPModel, self).__init__(train_features, train_targets, likelihood)

        batch_shape: dict = {"batch_shape": torch.Size((n_classes, ))} if prediction_type == "classification" else {}

        self.mean_module = gpytorch.means.ConstantMean(**batch_shape)

        n_features = train_features.size(dim=1)

        if not dist_constraint:
            constraints = gpytorch.constraints.Positive()
        else:
            constraints = gpytorch.constraints.Interval(1/dist_constraint, dist_constraint)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel_type(
                ard_num_dims=n_features,
                lengthscale_constraint=constraints,
                **batch_shape,
                **kwargs
            ),
            **batch_shape
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
            prediction_type: str,
            output_dir: Path,
            hyperparameters_fixed: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            likelihood=gpytorch.likelihoods.GaussianLikelihood,
            **kwargs
    ):
        super().__init__(
            prediction_type=prediction_type,
            output_dir=output_dir,
            hyperparameters_fixed=hyperparameters_fixed,
            hyperparameters=hyperparameters,
            likelihood=likelihood,
            **kwargs
        )
        self.likelihood = None

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the Gaussian Process model on the passed input data.

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of all input features.
            targets: Numpy ndarray (n_datapoints x 1) of all input targets.
        """
        training_data_x = torch.tensor(features).to(DEVICE)
        # Currently, this is a little hacky workaround for the fact that the NGBClassifier can only work with integer-
        # type category labels. Should be cleaned up at some point (factory pattern or two separate subclasses for
        # regression and classification)
        if self._prediction_type == "classification":
            self._category_scaler = CategoricalTransformer(categories=range(len(np.unique(targets.flatten()))))
            targets = self._category_scaler.fit_transform(targets.flatten())
            training_data_y = torch.tensor(targets.flatten(), dtype=torch.long).to(DEVICE)

        else:
            training_data_y = torch.tensor(targets.flatten()).to(DEVICE)

        # Instantiation of the Likelihood Function and the GP Model
        self.likelihood = self._get_likelihood(training_data_y)
        classification_details: dict = {"n_classes": self.likelihood.num_classes} if self._prediction_type == "classification" else {}

        self._model = ExactGPModel(
            prediction_type=self._prediction_type,
            train_features=training_data_x,
            train_targets=training_data_y,
            likelihood=self.likelihood,
            **classification_details,
            **self.hyperparameters
        ).to(DEVICE)

        # Setting the Model and the Likelihood to Training Mode
        self._model.train()
        self.likelihood.train()

        # Optimization
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hyperparameters.get("learning_rate", 1E-3))
        loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._model)

        for i in range(self.hyperparameters.get("training_iterations", 10000)):
            optimizer.zero_grad()
            output = self._model(training_data_x)

            # TODO: Properly rewrite this as a factory pattern in the class
            if self._prediction_type == "regression":
                loss_value = -loss(output, training_data_y)
                loss_value = loss_value.mean()

            elif self._prediction_type == "classification":
                loss_value = -loss(output, self.likelihood.transformed_targets).sum()

            loss_value.backward()
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

        test_data_x = torch.from_numpy(features).to(DEVICE)
        prediction = self.likelihood(self._model(test_data_x))

        if self._prediction_type == "regression":
            return prediction.mean.detach().cpu().numpy().reshape(-1, 1), prediction.variance.detach().cpu().numpy().reshape(-1, 1)

        elif self._prediction_type == "classification":
            predicted_labels = prediction.mean.detach().cpu().numpy().argmax(axis=0)
            predicted_variance = prediction.variance.detach().cpu().numpy()[predicted_labels, np.arange(len(predicted_labels))]

            # TODO: Check if the distribution variance at the maximum mean value is actually a good uncertainty metric
            #       for classification problems.

            return predicted_labels.reshape(-1, 1), predicted_variance.reshape(-1, 1)

    def _get_likelihood(self, targets: torch.Tensor) -> gpytorch.likelihoods.Likelihood:
        """
        Instantiates the likelihood depending on the prediction problem.
        Classification -> DirichletClassificationLikelihood
        Regression -> GaussianLikelihood (default), or passed as kwarg

        Args:
            targets: Tensor of target values (only for classification likelihood)

        Returns:
            gpytorch.likelihoods.Likelihood: Instance of the specific likelihood.
        """
        # TODO: Rewrite this as a factory pattern in a class variable?
        if self._prediction_type == "classification":
            return gpytorch.likelihoods.DirichletClassificationLikelihood(targets)

        elif "likelihood" in self._kwargs:
            return self._kwargs["likelihood"]()

        else:
            return gpytorch.likelihoods.GaussianLikelihood()
