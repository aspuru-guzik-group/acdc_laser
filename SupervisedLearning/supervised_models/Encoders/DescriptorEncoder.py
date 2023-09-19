from typing import List, Optional
import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DescriptorEncoder(object):

    def __init__(self, calc_3d: bool = False, n_jobs: int = -1):
        """
        Instantiates a DescriptorEncoder object that can calculate mordred descriptors or post-process external
        descriptors.

        Args:
            calc_3d: True if mordred descriptors should be calculated from the 3D geometry of the molecule.
                      Uses RDKit to do a force field optimization of all molecules.
            n_jobs: Number of parallel processes for calculating descriptors.
        """
        self.calc_3d = calc_3d
        self.calculator = Calculator(descriptors, ignore_3D=not calc_3d)
        self.descriptors = np.asarray([str(desc) for desc in self.calculator.descriptors])
        self._n_jobs: int = n_jobs

    def encode_entry(self, *mols, concatenate: bool = True) -> np.array:
        """
        Encodes one or multiple molecules (given as SMILES strings) into a single concatenated
        array of MORDRED descriptors.

        Args:
            mols: SMILES representation of the molecules.
            concatenate: True if the descriptors for all molecules should be concatenated.

        Returns:
            np.array: Array (n_descriptors, ) of mordred descriptors. Failed values are represented as np.nan.
        """
        mol_list: list = [self._generate_rdkit_mol(mol) for mol in mols]
        descriptor_list: list = [np.asarray(self.calculator(mol)).astype("float64") for mol in mol_list]

        if concatenate:
            return np.concatenate(descriptor_list)
        else:
            return np.asarray(descriptor_list)

    def encode_data(
            self,
            entries: np.ndarray,
            concatenate: bool = True,
            additional_features: Optional[np.ndarray] = None,
            pca: bool = False,
            pca_components: int = 30
    ) -> np.ndarray:
        """
        Encodes a list of lists of molecules (given as SMILES strings) into an ndarray of concatenated MORDRED
        descriptors. Removes all descriptors that could not be calculated for any molecule (represented as np.nan),
        and any descriptors that give zero variance across the data. Optionally performs PCA on the descriptors.

        Args:
            entries: Array of arrays of molecules (given as SMILES strings).
            concatenate: True if the descriptors for all components per entry should be concatenated.
            additional_features: Features to be added to the computed mordred descriptors. Ndarray of dimensionality
                                 (n_entries, n_features). Only possible if concatenate==True.
            pca: True if the descriptors should be down-selected by PCA.
            pca_components: Number of PCA components.

        Returns:
            np.ndarray: Ndarray (n_samples, n_descriptors or pca_components) of calculated descriptors.
        """
        descriptor_array: np.ndarray = np.asarray(Parallel(n_jobs=self._n_jobs)(
            delayed(self.encode_entry)(*entry, concatenate=concatenate) for entry in entries)
        )

        if concatenate:
            descriptor_array = self._remove_redundant_descriptors(descriptor_array)

            if additional_features is not None:
                descriptor_array = np.concatenate((descriptor_array, additional_features), axis=1)

            if not pca:
                return descriptor_array
            else:
                return self.pca_analysis(descriptor_array, pca_components)

        else:
            filtered_descriptors: list = []
            for idx in descriptor_array.shape[1]:
                all_desc = self._remove_redundant_descriptors(descriptor_array[:, idx, :])

                if not pca:
                    filtered_descriptors.append(all_desc)
                else:
                    filtered_descriptors.append(self.pca_analysis(all_desc, pca_components))

            return np.stack(filtered_descriptors, axis=1)

    def _generate_rdkit_mol(self, smiles: str) -> Chem.Mol:
        """
        Generates an RDKit Mol object from a SMILES string.
        If the calc_3d attribute is True, embeds a 3D geometry and optimizes it using the MMFF force field.

        Args:
             smiles: SMILES string of the molecule.

        Returns:
            Chem.Mol: RDKit Mol object (with optionally embedded conformer).
        """
        if not self.calc_3d:
            return Chem.MolFromSmiles(smiles)
        else:
            mol: Chem.Mol = Chem.MolFromSmiles(smiles)
            try:
                AllChem.EmbedMolecule(mol, maxAttempts=50000)  # Large number of maxAttempts to embed large molecules
            except ValueError:
                try:
                    AllChem.EmbedMolecule(mol, useRandomCoords=True, maxAttompts=5000)  # try different embedding mode
                except ValueError:
                    return mol

            AllChem.MMFFOptimizeMolecule(mol)

            return mol

    def _remove_redundant_descriptors(
            self,
            all_desc: np.ndarray,
            r_threshold: float = 0.999
    ) -> np.ndarray:
        """
        Removes redundant descriptors from a dataset of n_samples x n_descriptors.

        Descriptors are considered redundant if:
            - they are nan for any sample
            - they have zero variance across the dataset
            - they are highly correlated with another descriptor (r > r_threshold)

        Args:
            all_desc: ndarray (n_samples, n_descriptors) of descriptors.
            r_threshold: Threshold for correlation coefficient between descriptors.

        Returns:
            ndarray: ndarray (n_samples, n_descriptors - x) of descriptors with redundant descriptors removed.
        """
        # Filter out nan columns
        idx_to_keep = np.invert(np.any(np.isnan(all_desc), axis=0))
        all_desc = all_desc[:, idx_to_keep]
        self.descriptors = self.descriptors[idx_to_keep]

        # Filter out columns with constant values
        idx_to_keep = np.invert(np.all(all_desc == all_desc[0, :], axis=0))
        all_desc = all_desc[:, idx_to_keep]
        self.descriptors = self.descriptors[idx_to_keep]

        # Filter out highly correlated columns
        feature_correl = np.corrcoef(all_desc, rowvar=False)
        features_to_remove: set = set()
        for feature_idx in range(feature_correl.shape[0]):
            if feature_idx in features_to_remove:
                continue

            feature_correl_subset = feature_correl[feature_idx, feature_idx+1:]
            features_to_remove.update(np.where(feature_correl_subset > r_threshold)[0] + feature_idx + 1)

        features_to_remove: np.ndarray = np.array(sorted(list(features_to_remove)))
        all_desc = np.delete(all_desc, features_to_remove, axis=1)
        self.descriptors = np.delete(self.descriptors, features_to_remove, axis=0)

        return all_desc

    @staticmethod
    def pca_analysis(descriptors: np.ndarray, n_components: int = 30, scaler=StandardScaler) -> np.ndarray:
        """
        Performs PCA analysis on the given ndarray of descriptors (n_samples, n_descriptors).
        Scales the features using a defined scaler before PCA.

        Args:
             descriptors: ndarray (n_samples, n_descriptors) of descriptors.
             n_components: Number of PCA components to calculate.
             scaler: Class type of a scaler that implements the fit_transform method.

        Returns:
            np.ndarray: ndarray (n_samples, n_components) of principal components.
        """
        feature_scaler = scaler()
        descriptors_normalized: np.ndarray = feature_scaler.fit_transform(descriptors)

        pca = PCA(n_components=n_components)
        pca_components: np.ndarray = pca.fit_transform(descriptors_normalized)

        return pca_components
