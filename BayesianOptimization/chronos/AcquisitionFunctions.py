from typing import Optional, List
import math
from abc import ABCMeta, abstractmethod
import torch
from botorch.sampling import SobolQMCNormalSampler

from . import DEVICE
from .SurrogateModels import SurrogateModel
from .Utils import clear_cache


class DiscreteQAcquisitionFunction(metaclass=ABCMeta):
    """
    Abstract base class for acquisition functions that are evaluated on a "grid" of discrete entries (required for
    optimization of discrete entries, e.g. molecules). Acquisition function optimization is then trivial.

    Follows the logic of botorch's MC acquisition functions (sampling from a joint posterior over q points, calculating
    the acquisition function value for each sample, and averaging / maximizing over the samples).
    """

    _default_sample_shape = torch.Size([1024])
    _name = ""
    _min_value = 1E-6

    def __init__(
            self,
            model: SurrogateModel,
            max_batch_size: int = 16384,
            penalty_lengthscale: float = 1.0,
            objective_index: Optional[int] = None,
            **kwargs
    ):
        """
        Instantiates the AcquisitionFunction object.

        Args:
            model: SurrogateModel object of the trained surrogate model. Must implement a posterior() method.
            max_batch_size: Maximum batch size for evaluating the acquisition function.
            penalty_lengthscale: Lengthscale of the penalty function for pending data points.
            objective_index: Index of the objective to be optimized (None for single-objective optimization).
        """
        super().__init__()
        self.model = model
        self._objective_index = objective_index
        self._max_batch_size = max_batch_size
        self._penalty_lengthscale = penalty_lengthscale

    def __str__(self):
        return self._name

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of calculating the acquisition function value for a batch of posterior samples.

        Args:
            x: torch.Tensor (n_samples, n_points, n_features) of all features of the discrete data points.

        Returns:
            torch.Tensor: torch.Tensor (n_points,) of acquisition function values for each sample.
        """
        raise NotImplementedError

    def evaluate(
            self,
            x: torch.Tensor,
            posterior_samples: torch.Tensor,
            idx_pending: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates the acquisition function for a grid of discrete data points. Performs batchwise evaluation if the
        number of data points is larger than the maximum batch size.

        Returns the min-max scaled acquisition function values, followed by penalization.

        Args:
            x: torch.Tensor (n_points x n_features) of all features of the discrete data points.
            idx_pending: torch.Tensor (n_pending_points) of indices of pending data points (should be penalized).
            posterior_samples: Optional torch.Tensor (n_points x default_sample_shape) of posterior samples.

        Returns:
            torch.Tensor (n_points) of acquisition function values.

        Raises:
            ValueError: If all acquisition function values are the same.
        """
        if posterior_samples is None:
            posterior_samples = self.model.get_posterior_samples(
                x,
                objective_index=self._objective_index,
                n_samples=self._default_sample_shape[0],
                batch_size=self._max_batch_size
            )

        acqf_values = self.forward(posterior_samples)

        max_value = acqf_values.max().item()
        min_value = acqf_values.min().item()

        if (max_value - min_value) < self._min_value:
            raise ValueError("Acquisition function values are all the same. No further optimization possible!")

        acqf_values = (acqf_values - acqf_values.min().item()) / (acqf_values.max().item() - acqf_values.min().item())

        return self.apply_acqf_penalty(acqf_values, x, idx_pending)

    def apply_acqf_penalty(
            self,
            acqf_values: torch.Tensor,
            x: torch.Tensor,
            idx_pending: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Applies a penalty to the acquisition function values, accounting for pending data points.

        Calculates the penalty for each data point as
                penalty = - sum_{i=1}^{n_pending} (penalty_lengthscale / distance(x_i, x))^2
        This ensures that the penalty is infinite for all pending data points.

        Args:
            acqf_values: torch.Tensor (n_points) of un-penalized acquisition function values.
            x: torch.Tensor (n_points x n_features) of all features of the discrete data points.
            idx_pending: torch.Tensor (n_pending_points) of indices of pending data points (should be penalized).

        Returns:
            torch.Tensor (n_points) of acquisition function values with penalty applied.
        """
        if idx_pending is None:
            return acqf_values

        distances_pending = torch.cdist(x, x[idx_pending, :])
        penalties = torch.sum((self._penalty_lengthscale/distances_pending)**2, dim=1)
        penalties[idx_pending] = torch.inf

        return acqf_values - penalties


class DQUpperConfidenceBound(DiscreteQAcquisitionFunction):
    """
    Implementation of a discrete-only version of the q-UpperConfidenceBound acquisition function.
    """
    _name = "Discrete q-Upper Confidence Bound"

    def __init__(
            self,
            model: SurrogateModel,
            beta: float,
            **kwargs
    ):
        super().__init__(model, **kwargs)
        self.beta = math.sqrt(beta * math.pi / 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the upper confidence bound (UCB) acquisition function for a batch of discrete data points.

        Follows the implementation of qUpperConfidenceBound in botorch (sampling a joint posterior over q points,
        maximizing the upper confidence bound over q, averaging over all drawn samples).

        Args:
            x: torch.Tensor (n_samples x n_points x q) of all samples from the posterior distribution.

        Returns:
            torch.Tensor (n_points) of UCB acquisition function values.
        """
        if len(x.size()) == 2:  # Handle the case of q = 1
            x = x.unsqueeze(-1)

        mean = x.mean(dim=0)
        ucb_samples = mean + self.beta * (x - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)


class DQExpectedImprovement(DiscreteQAcquisitionFunction):
    """
    Implementation of a discrete-only version of the q-ExpectedImprovement acquisition function.
    """
    _name = "Discrete q-Expected Improvement"
    _min_value = 1e-9

    def __init__(
            self,
            model: SurrogateModel,
            best_f: float,
            **kwargs
    ):
        super().__init__(model, **kwargs)
        self._best_f = torch.as_tensor(best_f, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected improvement (EI) acquisition function for a batch of discrete data points.

        Follows the implementation of qExpectedImprovement in botorch (sampling a joint posterior over q points,
        maximizing the expected improvement over q, averaging over all drawn samples).

        Args:
            x: torch.Tensor (n_samples x n_points x q) of all samples from the posterior distribution.

        Returns:
            torch.Tensor (n_points) of EI acquisition function values.
        """
        if len(x.size()) == 2:  # Handle the case of q = 1
            x = x.unsqueeze(-1)

        objective = (x - self._best_f.unsqueeze(-1).to(x)).clamp_min(0)
        return objective.max(dim=-1)[0].mean(dim=0)


class DQProbabilityOfImprovement(DiscreteQAcquisitionFunction):
    """
    Implementation of a discrete-only version of the q-ProbabilityOfImprovement acquisition function.
    """
    _name = "Discrete q-Probability of Improvement"

    def __init__(
            self,
            model: SurrogateModel,
            best_f: float,
            tau: float = 1E-3,
            **kwargs
    ):
        super().__init__(model, **kwargs)
        self._best_f = torch.as_tensor(best_f, dtype=torch.float64)
        self._tau = torch.as_tensor(tau, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability of improvement (PI) acquisition function for a batch of discrete data points.

        Follows the implementation of qProbabilityOfImprovement in botorch (sampling a joint posterior over q points,
        maximizing the probability of improvement over q, averaging over all drawn samples).

        Args:
            x: torch.Tensor (n_samples x n_points x q) of all samples from the posterior distribution.

        Returns:
            torch.Tensor (n_points) of PI acquisition function values.
        """
        if len(x.size()) == 2:  # Handle the case of q = 1
            x = x.unsqueeze(-1)

        max_objective = x.max(dim=-1)[0]
        improvement = max_objective - self._best_f
        return torch.sigmoid(improvement / self._tau).mean(dim=0)


class DQRandomSearch(DiscreteQAcquisitionFunction):
    """
    Implementation of a discrete-only version of a random search acquisition function.
    """
    _name = "Discrete Random Search"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the random search acquisition function for a batch of discrete data points.

        Args:
            x: torch.Tensor (n_samples x n_points x q) of all samples from the posterior distribution.

        Returns:
            torch.Tensor (n_points) of random search acquisition function values.
        """
        return torch.rand(x.shape[1])
