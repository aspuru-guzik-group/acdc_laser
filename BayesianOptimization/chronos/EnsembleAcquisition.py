from typing import Tuple, List, Type, Optional
from pathlib import Path
import gc
import logging
import time
from tqdm import tqdm
import torch
from botorch.sampling import SobolQMCNormalSampler
from . import DEVICE
from .SurrogateModels import SurrogateModel
from .AcquisitionFunctions import DiscreteQAcquisitionFunction
from .Utils import clear_cache


class DQEnsembleAcquisition(object):
    """
    Acquisition strategy for batched, discrete-only Bayesian Optimization using a weighted ensemble of individual
    acquisition functions. Generates batches of recommendations by sequentially conditioning and optimizing the ensemble
    of acquisition functions.

    Usage of an ensemble of acquisition functions is based on:
    E. Brochu et al., "Portfolio Allocation for Bayesian Optimization", https://arxiv.org/abs/1009.5419.

    Strategy for batched Bayesian optimization by sequential conditioning and optimization is adapted from botorch:
     - Monte-Carlo optimization of acquisition functions: botorch.acquisition.monte_carlo
     - Sequential optimization of discrete acquisition functions: botorch.optim.optimize_acqf_discrete
    """

    def __init__(
            self,
            iteration: int,
            directory: Path,
            model: SurrogateModel,
            acquisition_functions: List[Type[DiscreteQAcquisitionFunction]],
            parameters: List[dict],
            penalty_lengthscale: float,
            objective_index: int = 0,
            eta: float = 1.0,
            memory_decay: float = 0.05,
            acquisition_batch_size: int = 8192,
            acquisition_MC_sample_size: int = 1024,
            fantasy_strategy: bool = True,
            logger: Optional[logging.Logger] = None,
            **kwargs
    ):
        """
        Instantiates a new DQEnsembleAcquisition object from scratch (iteration = 0) or from weights and parameters
        stored from a previous iteration (iteration > 0).

        Args:
            iteration: Iteration number. If iteration = 0, the acquisition function is initialized from scratch.
            directory: Directory where the weights and parameters are / will be stored.
            model: Trained surrogate model.
            acquisition_functions: List of acquisition function types.
            parameters: List of dictionaries of parameters for the acquisition functions.
            objective_index: Index of the objective function to be optimized.
            eta: Temperature parameter for weighting the ensemble of acquisition functions.
            memory_decay: Decay parameter for the memory of previous recommendations.
            acquisition_batch_size: Size of batches for evaluating the acquisition functions (posterior sampling).
            acquisition_MC_sample_size: Number of samples to be drawn for Monte-Carlo optimization of the acquisition
                                        functions.
            fantasy_strategy: If True, the posterior is drawn from a fantasy model based on the trained surrogate,
                              fantasizing over all pending data points. If False, the posterior is drawn from the
                              trained surrogate only.
            logger: Logger object.
            **kwargs: Additional keyword arguments to be passed to the acquisition functions.
        """
        self._logger = logger if logger is not None else logging.getLogger()
        self._logger.info(f"Initialized ensemble acquisition with {len(acquisition_functions)} acquisition functions.")
        self._directory = directory

        # Surrogate model attributes
        self._surrogate_model = model
        self._objective_index = objective_index
        self._fantasy_strategy = fantasy_strategy

        # Acquisition function attributes (including posterior sampling behavior)
        self._acquisition_functions = [
            function(model=model, **params, penalty_lengthscale=penalty_lengthscale, objective_index=objective_index, **kwargs)
            for function, params in zip(acquisition_functions, parameters)
        ]
        self._batch_size = acquisition_batch_size
        self._posterior_sample_size = acquisition_MC_sample_size

        # Acquisition function ensemble attributes
        self._iteration = iteration
        self._eta = eta
        self._memory_decay = memory_decay
        self._previous_recommendations = None  # Shape: (q, n_acqf, n_features)
        self._gains = torch.zeros(iteration, len(self._acquisition_functions))  # Shape: (iteration, n_acqf)
        self._probabilities = None  # Shape: (n_acqf,)
        self._train_data_sizes = torch.zeros(iteration)  # Shape: (iteration,)

        # Load previous state if iteration > 0, else initialize from scratch
        if iteration != 0:
            self._gains, self._previous_recommendations, self._train_data_sizes = self._load_state()
            self._update_gains()
        self._update_probabilities()
        self._train_data_sizes = torch.cat(
            (
                self._train_data_sizes,
                torch.tensor([self._surrogate_model.num_obs]).type(torch.get_default_dtype())
            )
        )

    def optimize_discrete(
            self,
            test_x: torch.Tensor,
            pending_x: Optional[torch.Tensor] = None,
            q: int = 1,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates q recommendations from a set of recommendations by sequentially conditioned optimization of the
        hedge acquisition function. Sequentially conditioned optimization is implemented following the logic of
        botorch.optim.optimize_acqf_discrete.

        Args:
            test_x: torch.Tensor (n_test_points x n_features) of all features of the discrete options in the search
                    space.
            pending_x: torch.Tensor (n_pending_points x n_features) of all features of the pending discrete options.
            q: Number of recommendations to generate.

        Returns:
            torch.Tensor: Indices of the recommended features (shape: (q,)).
            torch.Tensor: Weighted sum of acquisition function values for the recommended features (shape: (q,)).
        """
        self._logger.info(f"Starting to optimize the generation of {q} recommendations "
                          f"from {test_x.size(dim=0)} options.")

        # Initialize tensors of recommendations and acquisition function values
        recommendation_indices = torch.zeros(q, dtype=torch.int64)  # Shape: (q,)
        recommendation_candidates_all = torch.zeros(q, len(self._acquisition_functions), test_x.shape[1])  # Shape: (q, n_acqf, n_features)
        pending_x = pending_x if pending_x is not None else torch.empty(0, test_x.shape[1])  # Shape: (n_pending_points, n_features)

        # Iterate over all samples and draw them by sequential conditioning
        for sample_idx in range(q):
            start_time = time.time()

            if self._fantasy_strategy is True:
                fantasy_model = self._surrogate_model.fantasy_model(
                    X=pending_x,
                    objective_index=self._objective_index,
                    n_posterior_samples=self._posterior_sample_size
                )
            else:
                fantasy_model = None

            indices_per_acqf = self._evaluate_acquisition_functions(
                x=test_x,
                pending_indices=recommendation_indices[:sample_idx],
                model=fantasy_model
            ).argmax(dim=0)  # generates all acqf values and takes the maximum index per acqf

            recommended_index = indices_per_acqf[torch.multinomial(self._probabilities, 1, replacement=True).item()]
            recommendation_indices[sample_idx] = recommended_index
            recommendation_candidates_all[sample_idx, :, :] = test_x[indices_per_acqf, :]
            pending_x = torch.cat((pending_x, test_x[recommended_index, :].unsqueeze(dim=0)), dim=0)

            self._logger.info(f"Generated recommendation {sample_idx + 1}/{q} in {round(time.time() - start_time)} sec.")

        # Calculate the acquisition function values for the recommended data points
        recommendation_x = test_x[recommendation_indices, :]

        if self._fantasy_strategy is True:
            initial_fantasy_model = self._surrogate_model.fantasy_model(
                X=pending_x[:-q, :],
                objective_index=self._objective_index,
                n_posterior_samples=self._posterior_sample_size
            )
        else:
            initial_fantasy_model = self._surrogate_model

        recommendation_acqf_values = self._evaluate_acquisition_functions(recommendation_x, model=initial_fantasy_model)
        recommendation_acqf_values = torch.matmul(recommendation_acqf_values, self._probabilities)

        # Update the previous recommendations and save the state
        self._previous_recommendations = recommendation_candidates_all
        self._save_state()
        self._logger.info(f"Finished generating {q} recommendations.")

        return recommendation_indices, recommendation_acqf_values

    def _evaluate_acquisition_functions(
            self,
            x: torch.Tensor,
            model: SurrogateModel,
            pending_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evaluates all acquisition functions at the given points x, considering the pending observations at the given
        pending indices. Samples from the model posterior and computes the values of all acquisition functions based on
        these posterior samples.

        Args:
            x: torch.Tensor (n_test_points x n_features) of all features of the discrete points to evaluate.
            pending_indices: torch.Tensor (n_pending_points) of indices of pending observations.
            model: SurrogateModel to use for sampling. If None, the default model is used.

        Returns:
            torch.Tensor: (n_test_points x n_acqf) of acquisition function values.
        """
        all_acqf_values = torch.zeros(x.shape[0], len(self._acquisition_functions))  # Shape: (n_test_points, n_acqf)

        posterior_samples: torch.Tensor = model.get_posterior_samples(
            X=x,
            objective_index=self._objective_index,
            n_samples=self._posterior_sample_size,
            batch_size=self._batch_size
        )  # Shape: (n_posterior_samples, n_test_points)

        # Iterate over all acquisition functions and compute their values based on the posterior samples
        for function_idx, acqf in enumerate(self._acquisition_functions):
            try:
                acqf_values = acqf.evaluate(
                    x=x,
                    idx_pending=pending_indices,
                    posterior_samples=posterior_samples
                )  # Shape: (n_test_points,)

            except ValueError as e:
                self._logger.error(e)
                self._logger.warning(f"No acquisition function optimization possible ({acqf}). "
                                     f"Fallback to random sampling.")

                acqf_values = acqf.apply_acqf_penalty(
                    torch.rand(x.shape[0]),
                    x,
                    pending_indices
                )

            all_acqf_values[:, function_idx] = acqf_values

        return all_acqf_values

    def _update_gains(self) -> None:
        """
        Updates the gains for each acquisition function by evaluating the surrogate model at the recommendations of that
        acquisition function from the previous iteration. Takes the mean over all q predictions as the gain of a
        function in a given iteration (required for varying batch sizes per iteration).
        """
        average_iteration_gains = torch.zeros(len(self._acquisition_functions))

        for i in range(self._previous_recommendations.shape[1]):  # Iterate over acquisition functions
            recommendations = self._previous_recommendations[:, i, :].squeeze(1)  # Shape: (q, n_features)
            average_iteration_gains[i] = torch.mean(self._surrogate_model.predict(recommendations)[0])

        self._gains = torch.cat((self._gains, average_iteration_gains.unsqueeze(0)), dim=0)

    def _update_probabilities(self) -> None:
        """
        Updates the sampling probabilities for each acquisition f_i function by computing a weighted average w_i over
        all gains obtained at previous time points t_j by that acquisition function. Weights are computed by an
        exponential decay over time with the decay parameter m.

        w_i = sum_j^t exp(-m * (t - j)) * g_ij

        The sampling probabilities are then computed through a temperature-scaled softmax:

        p_i = exp(eta * w_i) / sum_j exp(eta * w_j)
        """
        if self._gains.shape[0] == 0:
            self._probabilities = torch.ones(len(self._acquisition_functions)) / len(self._acquisition_functions)
            return

        diff_sample_sizes = - self._train_data_sizes + self._train_data_sizes[-1]  # Shape: (iteration,)
        sample_weights = torch.exp(-self._memory_decay * diff_sample_sizes).reshape(1, -1)  # Shape: (1, iteration)
        weighted_gains = torch.matmul(sample_weights, self._gains).flatten()  # Shape: (n_acqf,)

        self._probabilities = torch.exp(self._eta * weighted_gains) / torch.exp(self._eta * weighted_gains).sum()

    def _save_state(self):
        """
        Saves the current state of the acquistion function (i.e. the gains and the recommendations from the current
        iteration) to the specified directory. Enables the acquisition function to be resumed from the current state.
        """
        torch.save(self._gains, self._directory / f"gains_{self._iteration}.pt")
        torch.save(self._previous_recommendations, self._directory / f"all_recommendations_{self._iteration}.pt")
        torch.save(self._train_data_sizes, self._directory / f"train_data_sizes_{self._iteration}.pt")

    def _load_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads the gains and recommendations from the previous iteration.

        Returns:
            torch.Tensor: Gains from the previous iteration.
            torch.Tensor: Recommendations from the previous iteration.
            torch.Tensor: The number of training points used for each acquisition function in all previous iterations.
        """
        gains = torch.load(self._directory / f"gains_{self._iteration-1}.pt")
        if gains.shape[1] != len(self._acquisition_functions):
            raise ValueError("The number of acquisition functions does not match the gains loaded.")

        recommendations = torch.load(self._directory / f"all_recommendations_{self._iteration-1}.pt")
        if recommendations.shape[1] != len(self._acquisition_functions):
            raise ValueError("The number of acquisition functions does not match the number of recommendations loaded.")

        train_data_sizes = torch.load(self._directory / f"train_data_sizes_{self._iteration-1}.pt")

        self._logger.info(f"Loaded gains and recommendations from iteration {self._iteration-1}.")

        return gains, recommendations, train_data_sizes
