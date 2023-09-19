from typing import Type, Tuple, Union, Callable, Optional
from abc import ABCMeta, abstractmethod
from joblib import Parallel, delayed
import logging
from tqdm import tqdm
import torch
from botorch.posteriors import Posterior, GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import Kernel, ScaleKernel, IndexKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from . import DEVICE
from .Utils import clear_cache


class SurrogateModel(metaclass=ABCMeta):
    """
    Metaclass for surrogate models for Bayesian Optimization on a fixed grid of discrete options.
    Designed for compatibility with botorch's MC acquisition functions.
    """
    name = ""

    def __init__(
            self,
            train_features: torch.Tensor,
            train_targets: torch.Tensor,
            logger: logging.Logger,
            **kwargs
    ):
        """
        Instantiates the surrogate model and sets the training features and targets as attributes.

        Args:
            train_features: torch Tensor (n_observations x n_features) of all features of observations.
            train_targets: torch Tensor (n_observations x 1) of all target values of the observations.
            logger: Logger object.
            kwargs: Additional keyword arguments.
        """
        self._logger = logger
        self._train_features = train_features.to(DEVICE)
        self._train_targets = train_targets.to(DEVICE)
        self.num_obs = self._train_features.size(0)
        self.num_outputs = self._train_targets.size(1)
        self._kwargs = kwargs

    def __str__(self):
        return self.name

    @abstractmethod
    def forward(self, x: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input features.
            idx: Index of the output to be predicted.

        Returns:
            torch.Tensor: Predicted target values.
        """
        raise NotImplementedError()

    def train_model(self) -> None:
        """
        Train the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def posterior(
            self,
            X: torch.Tensor,
            observation_noise: bool = False,
            objective_index: int = 0
    ) -> Posterior:
        """
        Posterior distribution of the model.

        Args:
            X: Input features.
            observation_noise: True if observation noise should be included.
            objective_index: Index of the objective to be optimized.

        Returns:
            Posterior: A botorch object for the posterior distribution.
        """
        raise NotImplementedError()

    def get_posterior_samples(
            self,
            X: torch.Tensor,
            objective_index: Optional[int] = None,
            n_samples: int = 1024,
            batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Draw samples from the posterior distribution.

        Args:
            X: Tensor (n_test_points x n_features) of test features.
            objective_index: Index of the objective to be optimized.
            n_samples: Number of samples to draw.
            batch_size: Batch size for sampling.

        Returns:
            torch.Tensor (n_samples x n_test_points): Samples from the posterior distribution.
        """

    @abstractmethod
    def fantasy_model(
            self,
            X: torch.Tensor,
            objective_index: Optional[int] = 0,
            n_posterior_samples: int = 1,
    ):
        """
        Create a fantasy model that fantasizes the objective values for the given features, and returns a new
        SurrogateModel object.

        Args:
            X: Input features.
            objective_index: Index of the objective to be optimized.
            n_posterior_samples: Number of posterior samples to draw.

        Returns:
            SurrogateModel: A surrogate model jointly trained on real and fantasized observations.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(
            self,
            X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict target means and variances for a given set of features.

        Args:
            X: Input features (n_test_points x n_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted target values and variances.
        """
        raise NotImplementedError()


class GPSurrogate(SurrogateModel, ExactGP):
    """
    Implementation of a Gaussian process Surrogate model for Bayesian Optimization on a fixed grid of discrete options.
    Uses the ExactGP class from gpytorch for the GP model.
    Posterior sampling is implemented using gpytorch's native sampling functions (very slow...).
    """
    name = "Gaussian Process"

    def __init__(
            self,
            train_features: torch.Tensor,
            train_targets: torch.Tensor,
            likelihood: Type[Likelihood] = GaussianLikelihood,
            kernel: Union[Type[Kernel], str] = RBFKernel,
            **kwargs
    ):
        """
        Instantiates the GP Surrogate model by initializing both parent classes.
        Instantiates the Kernel and Likelihood of the GP with some GammaPriors for noise and lengthscale constraints
        (parametrization adapted from botorch's SingleTaskGP model).
        Defines the mean_module and covar_module of the GP (required for the forward pass).

        Args:
            train_features: torch Tensor (n_observations x n_features) of all features of observations.
            train_targets: torch Tensor (n_observations x 1) of all target values of the observations.
            likelihood: Type of likelihood to be used for the Gaussian process.
        """
        SurrogateModel.__init__(self, train_features, train_targets)
        likelihood = likelihood().to(DEVICE)

        train_features, train_indices, train_targets = self._preprocess_data(train_features, train_targets)

        ExactGP.__init__(
            self,
            train_inputs=train_features if self.num_outputs == 1 else (train_features, train_indices),
            train_targets=train_targets,
            likelihood=likelihood
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.kernels: dict = {}
        self.covar_module = self._setup_covar_module(kernel)
        self.likelihood = likelihood

        # Override the attributes of the parent class to include the specifically preprocessed data
        self._train_features = train_features
        self._train_indices = train_indices
        self._train_targets = train_targets

        self._kwargs = kwargs

        self.to(DEVICE)

    def _setup_covar_module(
            self,
            kernel: Union[Type[Kernel], str],
    ) -> Callable:
        """
        Sets up the covariance module of the GP.

        Single Output: Scaled version of the defined feature kernel.
        Multi Output: Scaled version of the defined feature kernel, multiplied with an IndexKernel.

        Args:
            kernel: Type of kernel to be used.

        Returns:
            Callable: Covariance module of the GP as a function of features and optional indices.
        """
        # Define the feature kernel and scale it
        if isinstance(kernel, str):
            kernel = eval(kernel)

        self.kernels["feature_kernel"] = ScaleKernel(
            kernel(
                self._train_features.size(1) if self._kwargs.get("single_lengthscale", False) is False else None
            )
        ).to(DEVICE)

        # Single output case: Covariance corresponds to the feature kernel
        if self.num_outputs == 1:
            def covar_module(x: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
                return self.kernels["feature_kernel"](x)

        # Multi output case: Covariance corresponds to the feature kernel multiplied with an IndexKernel
        else:
            self.kernels["target_kernel"] = IndexKernel(
                num_tasks=self.num_outputs,
                rank=1
            ).to(DEVICE)

            def covar_module(x: torch.Tensor, idx: Optional[torch.Tensor]) -> torch.Tensor:
                return self.kernels["feature_kernel"](x).mul(self.kernels["target_kernel"](idx))

        return covar_module

    def forward(
            self,
            x: torch.Tensor,
            idx: Optional[torch.Tensor] = None,
    ) -> MultivariateNormal:
        """
        Implementation of the GP forward pass.

        Args:
            x: The features to evaluate the GP at.
            idx: The indices of the targets to compute the covariance matrix.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x, idx)
        return MultivariateNormal(mean_x, covar_x)

    def train_model(self) -> None:
        """
        Implementation of a standard GP training via backpropagation of the marginal log-likelihood loss.
        Uses Adam as the default optimizer without batched training, early stopping or regularization.
        """
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), self._kwargs.get("learning_rate", 0.01))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for _ in tqdm(range(self._kwargs.get("training_iterations", 1000)), disable=not self._kwargs.get("verbose", False)):
            optimizer.zero_grad()
            output = self(self._train_features, self._train_indices)
            loss = -mll(output, self._train_targets)
            loss.backward()
            optimizer.step()

    def posterior(
            self,
            X: torch.Tensor,
            observation_noise: bool = False,
            objective_index: Optional[int] = None,
            **kwargs
    ) -> GPyTorchPosterior:
        """
        Generates the posterior distribution of the GP model, evaluated at the data points X.
        Returns a botorch object for the posterior distribution (implemented to match botorch's acquisition function
        logic).

        Args:
            X: Torch Tensor (n_datapoints x n_features) for evaluation of the posterior.
            observation_noise: True if observation noise should be included.
            objective_index: Index of the target to be evaluated (only relevant for generating a single-target posterior
                          from a multi-output GP).

        Returns:
            GPyTorchPosterior: A botorch object for the posterior distribution.
        """
        features, indices, _ = self._preprocess_data(X)
        self.eval()
        self.likelihood.eval()

        if objective_index is not None:
            features = features[indices.flatten() == objective_index]
            indices = indices[indices.flatten() == objective_index]

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches(True):
            posterior_distribution: MultivariateNormal = self.likelihood(self(features, indices))

        return GPyTorchPosterior(distribution=posterior_distribution)

    def get_posterior_samples(
            self,
            X: torch.Tensor,
            objective_index: Optional[int] = None,
            n_samples: int = 1024,
            batch_size: Optional[int] = None,
    ):
        """
        Samples from the posterior of the trained surrogate model. Uses botorch's SoboQMCNormalSampler for sampling.

        Performs batch-wise evaluation. Shuffles the data points in x randomly before sampling. This is required to
        sample from joint posterior distributions over different data points. The shuffling is reversed after sampling.

        Args:
            X: torch.Tensor (n_points x n_features) of all features of the discrete data points.
            objective_index: Index of the target to be evaluated (only relevant for generating a single-target posterior
                             from a multi-output GP).
            n_samples: Number of posterior samples to draw.
            batch_size: Number of data points to evaluate in parallel.

        Returns:
            torch.Tensor: Tensor (n_samples x n_points) of posterior samples.
        """
        clear_cache()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples])).to(DEVICE)

        shuffling_indices = torch.randperm(X.shape[0])
        X = X[shuffling_indices]

        posterior_samples = []
        if batch_size is None:
            batch_size = X.shape[0]

        for batch in tqdm(X.split(batch_size), disable=not self._logger.isEnabledFor(logging.DEBUG)):
            posterior = self.posterior(batch, objective_index=objective_index)
            posterior_samples.append(sampler(posterior=posterior).squeeze(-1).cpu())
            clear_cache()

        posterior_samples = torch.cat(posterior_samples, dim=1)

        return posterior_samples[:, torch.argsort(shuffling_indices)]

    def fantasy_model(
            self,
            X: torch.Tensor,
            objective_index: Optional[int] = None,
            n_posterior_samples: int = 1,
    ) -> SurrogateModel:
        """
        Returns a copy of the current model with the specified fantasy observations added to the training data.
        The objective values for the fantasy observations are sampled from the posterior distribution of the GP.

        Args:
            X: Tensor (n_fantasy_points, n_features) of the fantasy observation features.
            objective_index: The index (int) of the objective to be fantasized.
            n_posterior_samples: The number of posterior samples to draw for the fantasy observations.

        Returns:
            SurrogateModel: A copy of the current model with the fantasy observations added to the training data.
        """
        features, indices, _ = self._preprocess_data(X)

        if objective_index is not None:
            features = features[indices.flatten() == objective_index]
            indices = indices[indices.flatten() == objective_index]
            input_data = [features, indices]
        else:
            input_data = features

        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches(True):
            posterior_distribution: MultivariateNormal = self(features, indices)
            predictive_distribution = self.likelihood(posterior_distribution)
            samples = predictive_distribution.rsample(torch.Size([n_posterior_samples]))

        predictive_means = samples.mean(dim=0)

        return self.get_fantasy_model(input_data, predictive_means)

    def predict(
            self,
            X: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            X: The features (n_datapoints x n_features) to evaluate the GP at.

        Returns:
            torch.Tensor: The predictive mean of the GP at all datapoints in X.
            torch.Tensor: The predictive variance of the GP at all datapoints in X.
        """

        features, indices, _ = self._preprocess_data(X)
        self.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.detach_test_caches(True):
            posterior_distribution: MultivariateNormal = self(features, indices)

        predictive_distribution = self.likelihood(posterior_distribution)

        return self._postprocess_prediction(predictive_distribution, self.num_outputs == 1)

    def _preprocess_data(
            self,
            features: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Preprocesses the data for the GP Model by removing all entries with nan values and ...

            - single target:    ...removes the last dimension of the tensor (i.e. "flattens" the tensor).
            - multiple targets: ...converts data points with multiple targets into multiple observations with a single
                                   target and a corresponding target index each.

                                   n_observations = n_datapoints * n_targets - n_nan_values

        Args:
            features: Tensor (n_datapoints, [...], n_features) of  data.
            targets (optional): Tensor (n_datapoints, n_targets) of the target values.

        Returns:
            torch.Tensor: Preprocessed features (n_observations, [...], n_features).
            torch.Tensor: Preprocessed target indices (n_observations, [...], 1). None for single-target models.
            torch.Tensor: Preprocessed targets (n_observations, 1). None if targets is None.
        """
        if self.num_outputs == 1:

            if targets is not None:
                valid_indices = ~torch.isnan(targets)[:, 0]
                targets = targets[valid_indices, :].squeeze(-1).to(DEVICE)
                features = features[valid_indices].to(DEVICE)

            else:
                features = features.to(DEVICE)

            indices = torch.empty(0, 1)

        else:
            all_features, all_indices, all_targets = [], [], []
            for target_idx in range(self.num_outputs):

                if targets is not None:
                    valid_indices = ~torch.isnan(targets[:, target_idx])
                    all_features.append(features[valid_indices].to(DEVICE))
                    all_indices.append(torch.full((valid_indices.sum(), *features.size()[1:-1], 1), fill_value=target_idx, dtype=torch.get_default_dtype()).to(DEVICE))
                    all_targets.append(targets[valid_indices, target_idx].to(DEVICE))

                else:
                    all_features.append(features.to(DEVICE))
                    all_indices.append(torch.full((*features.size()[:-1], 1), fill_value=target_idx, dtype=torch.get_default_dtype()).to(DEVICE))

            features = torch.cat(all_features, dim=0).to(DEVICE)
            indices = torch.cat(all_indices, dim=0).to(DEVICE)

            if targets is not None:
                targets = torch.cat(all_targets, dim=0)

        return features, indices, targets

    def _postprocess_prediction(
            self,
            predictive_distribution: MultivariateNormal,
            single_target: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Postprocesses the predictive distribution of the GP model by returning the mean and variance of the predictive
        distribution. Reshapes the mean and variance for multi-output GPs.

        Args:
            predictive_distribution: The predictive distribution of the GP model.
            single_target: True if the GP model is a single-output GP.

        Returns:
            torch.Tensor: The predictive mean of the GP at all datapoints (n_datapoints x n_targets).
            torch.Tensor: The predictive variance of the GP at all datapoints (n_datapoints x n_targets).
        """
        means, variances = predictive_distribution.mean.detach().cpu(), predictive_distribution.variance.detach().cpu()

        if self.num_outputs == 1 or single_target is True:
            pass

        else:
            n_samples = means.size(0) // self.num_outputs
            means = means.reshape(self.num_outputs, n_samples).transpose(0, 1)
            variances = variances.reshape(self.num_outputs, n_samples).transpose(0, 1)

        return means, variances





