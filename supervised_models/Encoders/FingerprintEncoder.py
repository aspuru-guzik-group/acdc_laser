from typing import Callable, List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


class FingerprintEncoder(object):

    fingerprints: dict = {
        "Morgan": AllChem.GetMorganFingerprintAsBitVect,
        "AtomPairs": AllChem.GetHashedAtomPairFingerprintAsBitVect,
        "TopologicalTorsion": AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect,
    }

    def __init__(self, fingerprint: str, **kwargs):

        self.fingerprint: Callable = self.fingerprints[fingerprint]
        self.kwargs = kwargs

    def encode_entry(self, *mols) -> np.array:
        """
        Encodes a single entry (molecule or multiple molecules) as a concatenated fingerprint.

        Args:

            mols: SMILES strings of all molecules to be encoded

        Returns:
            np.array: Concatenated bit vector of the fingerprint for all molecules.
        """
        fp_objects: list = [
            self.fingerprint(Chem.MolFromSmiles(smiles), **self.kwargs)
            for smiles in mols
        ]

        return np.concatenate(fp_objects).astype("float64")

    def encode_data(self, entries: np.ndarray) -> np.ndarray:
        """
        Encodes multiple entries (each one as a single or multiple molecules) into the corresponding fingerprints.

        Args:
             entries: Array of arrays of SMILES

        Returns:
            np.ndarray: Numpy ndarray of all entries (rows) converted to fingerprints.
        """
        fingerprints = [self.encode_entry(*entry) for entry in entries]
        return np.asarray(fingerprints)