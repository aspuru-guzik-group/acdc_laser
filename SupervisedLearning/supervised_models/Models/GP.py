import numpy as np
from pathlib import Path
from typing import Tuple, Callable, Optional, Union

import torch.distributions
from ..Utils.TorchSettings import *
import gpytorch.models
import gpytorch.constraints
from gpytorch.kernels import Kernel
from gpytorch.kernels.kernel import default_postprocess_script

from supervised_models.SupervisedModel import SupervisedModel
from supervised_models.Utils import CategoricalTransformer


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Implementation of a formally exact Gaussian Process regressor with a given mean and covariance model.
    Requires a Gaussian likelihood for exact MLL inference
    """
    def __init__(
            self,
            prediction_type: str,
            train_features: Tensor,
            train_targets: Tensor,
            likelihood: Callable,
            kernel_type: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
            dist_constraint: Optional[float] = None,
            feature_lengthscale: str = "all",
            n_classes: Optional[int] = None,
            **kwargs
    ):
        """
        Instantiates the super class (gpytorch.models.ExactGP) with the generalized forward method from GPModel.

        Args:
            prediction_type: "classification" or "regression"
            train_features: Tensor of all training data points and features (n_datapoints x n_features)
            train_targets: Tensor of all training data points and targets (n_datapoints x 1)
            likelihood: Likelihood function
            kernel_type: Kernel from gpytorch.kernels
            dist_constraint (optional): Distance constraint for the kernel implementation
            feature_lengthscale: "none" (no lengthscale), "single" (one lengthscale for all features), "all" (one
                                lengthscale for each feature)
            n_classes (optional): Number of classes (for classification, ignored if prediction_type == "regression")
        """
        gpytorch.models.ExactGP.__init__(
            self,
            train_inputs=train_features,
            train_targets=train_targets,
            likelihood=likelihood
        )

        batch_shape: dict = {"batch_shape": torch.Size((n_classes, ))} if prediction_type == "classification" else {}

        self.mean_module = gpytorch.means.ConstantMean(**batch_shape)

        if not dist_constraint:
            constraints = gpytorch.constraints.Positive()
        else:
            constraints = gpytorch.constraints.Interval(1/dist_constraint, dist_constraint)

        n_lengthscales = None if feature_lengthscale == "none" else 1 if feature_lengthscale == "single" else train_features.size(dim=1)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel_type(
                ard_num_dims=n_lengthscales,
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


class MultitaskGPModel(gpytorch.models.ExactGP):
    """
    Implementation of a formally exact Multitask Gaussian Process regressor with a given mean and covariance model.
    Implements two types of multitask approaches:
        - Regular Multitask GP (with a single covariance matrix) -> requires all data points to have values for all
                                                                    tasks
        - Hadamard Multitask GP (one covariance matrix for features, one for tasks) -> requires min. one value for any
                                                                                       data point
    """
    def __init__(
            self,
            prediction_type: str,
            train_features: Tensor,
            train_targets: Tensor,
            num_tasks: int,
            likelihood: Callable,
            kernel_type: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
            target_indices: Optional[Tensor] = None,
            dist_constraint: Optional[float] = None,
            feature_lengthscale: str = "all",
            hadamard_method: bool = False,
            n_classes: Optional[int] = None,
            rank: int = 1,
            **kwargs
    ):
        """
        Instantiates the super class (gpytorch.models.ExactGP) with the generalized forward method from GPModel.
        Uses the MultitaskMean and MultitaskKernel from gpytorch to implement a multitask GP.

        Args:
            prediction_type: "classification" or "regression"
            train_features: Tensor of all training data points and features (n_datapoints x n_features)
            train_targets: Tensor of all training data points and targets (n_datapoints x n_tasks)
            num_tasks: Number of tasks
            likelihood: Likelihood function
            kernel_type: Kernel from gpytorch.kernels
            target_indices: Tensor of indices for the tasks (n_datapoints x 1)
            dist_constraint (optional): Distance constraint for the kernel implementation
            feature_lengthscale: "none" (no lengthscale), "single" (one lengthscale for all features), "all" (one
                                lengthscale for each feature)
            hadamard_method (optional): Whether to use the Hadamard method for multitask GP
            n_classes (optional): Number of classes (for classification, ignored if prediction_type == "regression")
            rank (optional): Rank for the MultiTaskKernel / MultiIndexKernel (Hadamard method)
        """
        if prediction_type == "classification":
            raise NotImplementedError("Multitask GP classification not implemented yet.")

        self._hadamard_method = hadamard_method

        # Call the super class constructor
        gpytorch.models.ExactGP.__init__(
            self,
            train_inputs=train_features if hadamard_method is False else (train_features, target_indices),
            train_targets=train_targets,
            likelihood=likelihood
        )

        # Compute the distance constraints for the kernel
        if not dist_constraint:
            constraints = gpytorch.constraints.Positive()
        else:
            constraints = gpytorch.constraints.Interval(1 / dist_constraint, dist_constraint)

        n_lengthscales = None if feature_lengthscale == "none" else 1 if feature_lengthscale == "single" else train_features.size(dim=1)

        # Instantiate the mean and covariance modules for regular Multitask GPs
        if hadamard_method is False:

            n_targets = train_targets.size(dim=1)

            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(),
                num_tasks=n_targets
            )

            self.covar_module = gpytorch.kernels.MultitaskKernel(
                kernel_type(
                    ard_num_dims=n_lengthscales,
                    lengthscale_constraint=constraints,
                    **kwargs
                ),
                num_tasks=n_targets,
                rank=rank
        )

        # Define mean and covariance functions for the Hadamard method (separate covariances for features and different
        # targets)
        else:
            self.mean_module = gpytorch.means.ConstantMean()

            self.feature_covar = gpytorch.kernels.ScaleKernel(
                kernel_type(
                    ard_num_dims=n_lengthscales,
                    lengthscale_constraint=constraints,
                    **kwargs
                ),
            )

            self.target_covar = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=rank)

    def forward(self, x: Tensor, idx: Optional[Tensor] = None):
        """
        Implementation of the forward method required for model training and predictions.
        Computes the mean and covariance from the given input tensor, and returns the multitask version of the
        MultivariateNormal distribution with the given means and covariances.

        Args:
            x: Tensor of features passed.
            idx: Tensor of indices of the targets to be used for prediction (only for the Hadamard method).

        Returns:
            gpytorch.distributions.MultitaskMultivariateNormal: A (multitask) multivariate normal distribution with the
                                                                given means and covariances.
        """
        if self._hadamard_method is False:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        else:
            mean_x = self.mean_module(x)
            feature_covar = self.feature_covar(x)
            target_covar = self.target_covar(idx)
            return gpytorch.distributions.MultivariateNormal(mean_x, feature_covar.mul(target_covar))


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
            device_idx: Optional[int] = 0,
            **kwargs
    ):
        super().__init__(
            prediction_type=prediction_type,
            output_dir=output_dir,
            hyperparameters_fixed=hyperparameters_fixed,
            hyperparameters=hyperparameters,
            **kwargs
        )

        self.likelihood = None
        self.device = DEVICES[device_idx]
        self._logger.info(f"GP model initialized on device {self.device}")

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the Gaussian Process model on the passed input data.

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of all input features.
            targets: Numpy ndarray (n_datapoints x 1) of all input targets.
        """
        training_data_x = torch.tensor(features, dtype=torch.get_default_dtype()).to(self.device)
        # Currently, this is a little hacky workaround for the fact that the NGBClassifier can only work with integer-
        # type category labels. Should be cleaned up at some point (factory pattern or two separate subclasses for
        # regression and classification)
        if self._prediction_type == "classification":
            self._category_scaler = CategoricalTransformer(categories=range(len(np.unique(targets.flatten()))))
            targets = self._category_scaler.fit_transform(targets.flatten())
            training_data_y = torch.tensor(targets.flatten(), dtype=torch.get_default_dtype()).to(self.device)

        else:
            training_data_y = torch.tensor(targets.flatten(), dtype=torch.get_default_dtype()).to(self.device)

        # Instantiation of the Likelihood Function and the GP Model
        self.likelihood = self._get_likelihood(training_data_y).to(self.device)
        classification_details: dict = {"n_classes": self.likelihood.num_classes} if self._prediction_type == "classification" else {}

        self._model = ExactGPModel(
            prediction_type=self._prediction_type,
            train_features=training_data_x,
            train_targets=training_data_y,
            likelihood=self.likelihood,
            **classification_details,
            **self.hyperparameters
        ).to(self.device)

        # Setting the Model and the Likelihood to Training Mode
        self._model.train()
        self.likelihood.train()

        # Optimization
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hyperparameters.get("learning_rate", 1E-3))
        loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._model)

        for i in range(self.hyperparameters.get("training_iterations", 1000)):
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

        test_data_x = torch.tensor(features, dtype=torch.get_default_dtype()).to(self.device)
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


class MultitaskGaussianProcess(SupervisedModel):

    """
    Instance of the SupervisedModel metaclass using a Multi-Task GaussianProcess Regressor (from gpytorch) as the
    Supervised Learning Model. Allows for two implementations of the multi-task prediction problem:
    - Regular Multitask GP (with a single covariance matrix) -> requires all data points to have values for all
                                                                tasks
    - Hadamard Multitask GP (one covariance matrix for features, one for tasks) -> requires min. one value for any
                                                                                   data point
    """

    name = "MultitaskGaussianProcess"

    def __init__(
            self,
            prediction_type: str,
            output_dir: Path,
            hyperparameters_fixed: Optional[dict] = None,
            hyperparameters: Optional[dict] = None,
            device_idx: Optional[int] = 0,
            **kwargs
    ):
        """
        Instantiates the MultitaskGaussianProcess class.
        """
        self.likelihood = None
        self._hadamard_method = hyperparameters_fixed.pop("hadamard_method", False) if hyperparameters_fixed is not None else False
        self.device = DEVICES[device_idx]

        super().__init__(
            prediction_type=prediction_type,
            output_dir=output_dir,
            hyperparameters_fixed=hyperparameters_fixed,
            hyperparameters=hyperparameters,
            **kwargs
        )

        self._logger.info(f"GP model initialized on device {self.device}")

    def _train(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Trains the Gaussian Process model on the passed input data.

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of all input features.
            targets: Numpy ndarray (n_datapoints x n_tasks) of all input targets.
        """
        if self._hadamard_method is True:
            training_data_x, train_idx, training_data_y = self._preprocess_data(features, targets)
        else:
            training_data_x, training_data_y = self._preprocess_data(features, targets)

        # Instantiation of the Likelihood Function and the GP Model
        if self._hadamard_method is True:
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        else:
            self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self._n_tasks).to(self.device)

        self._model = MultitaskGPModel(
            prediction_type=self._prediction_type,
            train_features=training_data_x,
            train_targets=training_data_y,
            likelihood=self.likelihood,
            num_tasks=self._n_tasks,
            hadamard_method=self._hadamard_method,
            target_indices=train_idx if self._hadamard_method is True else None,
            **self.hyperparameters
        ).to(self.device)

        # Setting the Model and the Likelihood to Training Mode
        self._model.train()
        self.likelihood.train()

        # Optimization
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hyperparameters.get("learning_rate", 1E-2))
        loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._model)

        for i in range(self.hyperparameters.get("training_iterations", 1000)):
            optimizer.zero_grad()
            if self._hadamard_method is False:
                output = self._model(training_data_x)
            else:
                output = self._model(training_data_x, train_idx)

            loss_value = -loss(output, training_data_y)
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

        if self._hadamard_method is False:
            test_data_x = self._preprocess_data(features)
            prediction = self.likelihood(self._model(test_data_x))
        else:
            test_data_x, test_idx = self._preprocess_data(features)
            prediction = self.likelihood(self._model(test_data_x, test_idx))

        return self._postprocess_predictions(prediction)

    def _preprocess_data(
            self,
            features: np.ndarray,
            targets: Optional[np.ndarray] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Preprocesses the features and targets to be torch.Tensors that are compatible with the respective required
        GP Models (especially for the Hadamard-type multi-task models).

        Args:
            features: Numpy ndarray (n_datapoints x n_features) of all input features.
            targets (optional): Numpy ndarray (n_datapoints x n_tasks) of all input targets.

        Returns:
            torch.Tensor (always): Tensor (n_datapoints x n_features)
            torch.Tensor (if hadamard_method is False and targets is not None): Tensor (n_datapoints x n_tasks)
            torch.Tensor (if hadamard_method is True): Tensor ((n_datapoints * n_tasks) x 1)
            torch.Tensor (if hadamard_method is True and targets is not None): Tensor ((n_datapoints * n_tasks) x 1)
        """
        # Regular processing (conversion into a torch.Tensor) for standard multi-task GP models
        if self._hadamard_method is False:
            train_data_x = torch.tensor(features, dtype=torch.get_default_dtype()).to(self.device)

            if targets is not None:
                train_data_y = torch.tensor(targets, dtype=torch.get_default_dtype()).to(self.device)
                return train_data_x, train_data_y
            else:
                return train_data_x

        # Preprocessing into multiple observations for Hadamard-type multi-task GP models
        else:
            all_features, all_task_indices, all_targets = [], [], []
            for task_idx in range(self._n_tasks):
                if targets is not None:
                    indices = ~np.isnan(targets[:, task_idx])
                    all_features.append(torch.tensor(features[indices, :], dtype=torch.get_default_dtype()).to(self.device))
                    all_task_indices.append(torch.full((sum(indices), 1), dtype=torch.get_default_dtype(), fill_value=task_idx).to(self.device))
                    all_targets.append(torch.tensor(targets[indices, task_idx], dtype=torch.get_default_dtype()).to(self.device))
                else:
                    all_features.append(torch.tensor(features, dtype=torch.get_default_dtype()).to(self.device))
                    all_task_indices.append(torch.full((features.shape[0], 1), dtype=torch.get_default_dtype(), fill_value=task_idx).to(self.device))

            train_data_x = torch.cat(all_features, dim=0)
            train_data_idx = torch.cat(all_task_indices, dim=0)

            if targets is not None:
                train_data_y = torch.cat(all_targets, dim=0)
                return train_data_x, train_data_idx, train_data_y
            else:
                return train_data_x, train_data_idx

    def _postprocess_predictions(
            self,
            prediction: torch.distributions.Distribution,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocesses the predictions of the GP model.

        Args:
            prediction: Distribution of the predictions of the GP model.

        Returns:
            np.ndarray: Predicted means for all data points (n_datapoints x n_tasks)
            np.ndarray: Predicted variances for all data points (n_datapoints x n_tasks)
        """
        means, variances = prediction.mean.detach().cpu().numpy(), prediction.variance.detach().cpu().numpy()

        if self._hadamard_method is False:
            return means, variances

        else:
            n_samples = means.shape[0] // self._n_tasks
            return means.reshape(self._n_tasks, n_samples).T, variances.reshape(self._n_tasks, n_samples).T


class TanimotoKernel(Kernel):
    """
    Implementation of a Tanimoto Kernel for Gaussian Processes in GPyTorch. Computes the Jaccard distance between bit
    vectors as a measure of covariance.

    k(x1, x2) = (x1, x2) / (||x1||^2 + ||x2||^2 - (x1, x2))

    Follows the general implementation in GAUCHE (Rhys-Griffiths et al., 2023).
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        """
        General implementation of the forward pass of the kernel.

        Args:
            x1: Tensor (n_datapoints x n_features, optional: batch) of input features.
            x2: Tensor (n_datapoints x n_features, optional: batch) of input features.
            **kwargs: Additional keyword arguments to be passed to the covariance module.
        """
        return self.covar(x1, x2, **kwargs)

    def covar(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            last_dim_is_batch: bool = False,
            eps: float = 1e-6,
            **kwargs
    ):
        """
        Performs the required tensor reshaping and computes the Tanimoto similarity covariance matrix.

        Args:
            x1: Tensor (n_datapoints x n_features, optional: batch) of input features.
            x2: Tensor (n_datapoints x n_features, optional: batch) of input features.
            last_dim_is_batch: Boolean indicating whether the last dimension of the input tensors is a batch dimension.
            eps: Small value to ensure numerical stability.
            **kwargs: Additional keyword arguments (ignored here).
        """
        # adjust shapes if last dimension is a batch dimension
        if last_dim_is_batch is True:
            x1 = x1.transpose(-1, 2).unsqueeze(-1)
            x2 = x2.transpose(-1, 2).unsqueeze(-1)

        # calculate the Tanimoto similarity matrix, adjusting for numerical stability
        x1_sum = torch.sum(x1**2, dim=-1, keepdim=True)
        x2_sum = torch.sum(x2**2, dim=-1, keepdim=True)
        x1x2 = torch.matmul(x1, torch.transpose(x2, -1, -2))
        tanimoto = (x1x2 + eps) / (x1_sum + torch.transpose(x2_sum, -1, -2) - x1x2 + eps)

        # remove negative values
        tanimoto.clamp_min_(0)

        return self._postprocess(tanimoto)
