from typing import Union, Tuple
import numpy as np
from deepchem.splits import FingerprintSplitter, MaxMinSplitter
from deepchem.data import DiskDataset
from sklearn.model_selection import KFold


class NoOuterSplitter(object):
    """
    Dummy splitter that does not split the data at all. Implements the split method so it can be used on a "normal"
    dataset using the logics of the SupervisedModel metaclass.
    """
    def __init__(self, *args, **kwargs):
        pass

    def split(self, data: np.ndarray):
        data = np.asarray(data)
        return [(np.arange(data.shape[0]), None)]


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

        ATTN: Can currently not generate multiple, differently bootstraped splits, since the random state cannot be
              set. This is a problem of Deepchem's FingerprintSplitter.

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

        for i in range(self._n_splits):
            train_idx, _, test_idx = splitter.split(
                dataset=deepchem_dataset,
                frac_train=1-self._test_size,
                frac_valid=0,
                frac_test=self._test_size
            )

            yield np.asarray(train_idx), np.asarray(test_idx)


class MultipleKFoldSplitter(object):
    """
    Generates multiple K-fold splits to get better / more reliable statistics of predictive performance.
    Re-sets the random seed each time to ensure a different splitting behaviour.

    Args:
        n_iter: Number of k-fold train-test split sets to generate.
        k: Number of folds.
    """
    def __init__(self, n_iter: int, k: int, **kwargs):
        self._n = n_iter
        self._k = k

    def split(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generator that splits the dataset using multiple k-fold train-test splits.
        Generates n x k train-test splits.

        Args:
            data: Numpy ndarray (n_datapoints x n_features) of data to be split.

        Yields:
            np.ndarray: Train indices
            np.ndarray: Test indices
        """
        for n in range(self._n):
            splitter = KFold(n_splits=self._k, shuffle=True, random_state=n)
            for train, test in splitter.split(data):
                yield train, test
