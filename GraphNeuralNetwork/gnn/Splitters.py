from typing import Tuple, Union
import numpy as np
import logging
from deepchem.splits import FingerprintSplitter
from deepchem.data import DiskDataset


logging.getLogger("deepchem").setLevel(logging.CRITICAL)


class ECFPSplitter(object):
    """
    Wrapper around the Deepchem FingerprintSplitter class, matching the syntax / use of Scikit-Learn's Splitters
    (i.e., implements the split method that creats an iterator).

    Creates n differently initialized (bootstrapped) Fingerprint Splits.

    Basic Usage:

    splitter = ECFPSplitter(molecule_smiles=smiles, n_splits=5, test_size=0.3, shuffle=True, random_state=42)

    for train, test in splitter.split(feature_array):
        ...
    """
    def __init__(
            self,
            molecule_smiles: Union[list, np.ndarray],
            n_splits: int = 1,
            test_size: float = 0.3,
            random_state: int = 42,
            **kwargs
    ):
        """
        Instantiates the ECFPSplitter.

        Args:
            molecule_smiles: List or array of SMILES strings for all molecules.
            n_splits: Number of splits to be generated.
            test_size: Relative size of the test set (0 < test_size < 1).
            random_state: Random state variable.
        """
        np.random.seed(random_state)
        self._random_state: int = random_state
        self._structures: np.array = np.asarray(molecule_smiles).flatten()
        self._n_splits: int = n_splits
        self._test_size: float = test_size

    def split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator that splits the dataset using the FingerprintSplitter from Deepchem.
        Generates n independent, bootstrapped splits with different initial seeds.

        Args:
            data: Numpy ndarray (n_datapoints x n_features) of data to be split.

        Yields:
            np.ndarray: Train indices
            np.ndarray: Test indices
        """
        assert data.shape[0] == len(self._structures)

        deepchem_dataset = DiskDataset.from_numpy(
            X=np.zeros(len(self._structures)),
            y=np.zeros(len(self._structures)),
            w=np.zeros(len(self._structures)),
            ids=self._structures
        )

        splitter = FingerprintSplitter()

        for _ in range(self._n_splits):
            train_idx, _, test_idx = splitter.split(
                dataset=deepchem_dataset,
                frac_train=1-self._test_size,
                frac_valid=0,
                frac_test=self._test_size,
                seed=np.random.randint(low=0, high=100)
            )

            yield np.asarray(train_idx), np.asarray(test_idx)


class RandomSplitter(object):

    def __init__(
            self,
            n_splits: int = 1,
            test_size: float = 0.3,
            random_state: int = 42,
            **kwargs
    ):
        """
        Instantiates the RandomSplitter.

        Args:
            n_splits: Number of splits to be generated.
            test_size: Relative size of the test set (0 < test_size < 1).
            random_state: Random state variable.
        """
        np.random.seed(random_state)
        self._random_state: int = random_state
        self._n_splits: int = n_splits
        self._test_size: float = test_size

    def split(
            self,
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator that splits the dataset randomly.
        Generates n independent, bootstrapped splits with different initial seeds.

        Args:
            data: Numpy ndarray (n_datapoints x n_features) of data to be split.

        Yields:
            np.ndarray: Train indices
            np.ndarray: Test indices
        """
        n_datapoints = data.shape[0]
        n_test = int(self._test_size * n_datapoints)
        n_train = n_datapoints - n_test

        for _ in range(self._n_splits):
            idx = np.arange(n_datapoints)
            np.random.shuffle(idx)

            yield idx[:n_train], idx[n_train:]
