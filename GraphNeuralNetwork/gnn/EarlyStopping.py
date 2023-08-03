from typing import Callable
from pathlib import Path
import shutil

import numpy as np
import tensorflow as tf
import sonnet as snt


class EarlyStopping(object):

    def __init__(
            self,
            model: snt.Module,
            patience: int = 100,
            threshold: float = 1e-3,
            minimize: bool = True,
            tmp_dir: Path = Path.cwd() / "tmp"
    ):
        """
        Instantiates

        Args:
            model: Sonnet module.
            patience: Number of iterations without improvement before early stopping is triggered.
            threshold: Absolute threshold to be considered an improvement.
            minimize: True if the metric should be minimized upon training.
            tmp_dir: Path to the directory where training checkpoints should be saved.
        """
        self._patience = patience
        self._num_no_improvement: int = 0

        self._tmp_dir: Path = tmp_dir
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        if minimize is True:
            self._check_criterion: Callable = lambda value, reference: np.less(value + threshold, reference)
            self.best_value: float = np.inf
        else:
            self._check_criterion: Callable = lambda value, reference: np.greater(value - threshold, reference)
            self.best_value: float = -np.inf

        self._checkpoint = tf.train.Checkpoint(model=model)
        self._checkpoint_manager = tf.train.CheckpointManager(
            self._checkpoint,
            self._tmp_dir,
            max_to_keep=5
        )

    def check_convergence(self, loss_value: float) -> bool:
        """
        Checks for validantion performance improvement and model convergence after each training iteration.
        Saves a checkpoint of the optimized model.

        Args:
            loss_value: Value of the validation loss.

        Returns:
            bool: True if the model has not improved by the defined threshold over the defined number of iterations.
        """
        if self._check_criterion(loss_value, self.best_value):
            self.best_value = loss_value
            self._num_no_improvement = 0
            self._checkpoint_manager.save()

        else:
            self._num_no_improvement += 1

        return self._num_no_improvement >= self._patience

    def restore_best_model(self) -> None:
        """
        Restores the best trained model from the saved checkpoint.
        Clears the saved checkpoints folder.
        """
        self._checkpoint.restore(self._checkpoint_manager.latest_checkpoint)
        shutil.rmtree(self._tmp_dir)




