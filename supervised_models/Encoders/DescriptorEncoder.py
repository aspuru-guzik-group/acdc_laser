from typing import List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class DescriptorEncoder(object):

    def __init__(self, calc_3d: bool = False):
        self.calc_3d = calc_3d
        self.calculator = Calculator(descriptors, ignore_3D=not calc_3d)

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
        descriptor_array: np.ndarray = np.asarray([self.encode_entry(*entry, concatenate=concatenate) for entry in entries])

        if concatenate:
            descriptor_array = descriptor_array[:, np.invert(np.any(np.isnan(descriptor_array), axis=0))]  # filters nan columns out
            descriptor_array = descriptor_array[:, np.invert(np.all(descriptor_array == descriptor_array[0, :], axis=0))]  # filters non-differentiating columns out

            if additional_features is not None:
                descriptor_array = np.concatenate((descriptor_array, additional_features), axis=1)

            if not pca:
                return descriptor_array
            else:
                return self.pca_analysis(descriptor_array, pca_components)

        else:
            filtered_descriptors: list = []
            for idx in descriptor_array.shape[1]:
                all_desc = descriptor_array[:, idx, :]
                all_desc = all_desc[:, np.invert(np.any(np.isnan(all_desc), axis=0))]
                all_desc = all_desc[:, np.all(all_desc == all_desc[0, :], axis=0)]

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

    @staticmethod
    def pca_analysis(descriptors: np.ndarray, n_components: int = 30) -> np.ndarray:
        """
        Performs PCA analysis on the given ndarray of descriptors (n_samples, n_descriptors).
        Scales the features by a MinMax Scaler before PCA.

        Args:
             descriptors: ndarray (n_samples, n_descriptors) of descriptors.
             n_components: Number of PCA components to calculate.

        Returns:
            np.ndarray: ndarray (n_samples, n_components) of principal components.
        """
        scaler = MinMaxScaler()
        descriptors_normalized: np.ndarray = scaler.fit_transform(descriptors)

        pca = PCA(n_components=n_components)
        pca_components: np.ndarray = pca.fit_transform(descriptors_normalized)

        return pca_components
