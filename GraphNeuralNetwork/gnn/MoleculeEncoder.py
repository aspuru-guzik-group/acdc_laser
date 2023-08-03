import copy
from typing import List, Union, Dict, Tuple, Optional
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from rdkit import Chem
import graph_nets


class MolFeaturizer(object):
    """
    Featurizer for molecules to generate atom / node features as inputs for Graph Neural Networks.
    Generates featurization for DeepMind's graph_nets architecture.
    """
    _atom_features: dict = {
        "AtomSymbol": lambda atom: atom.GetSymbol(),
        "ChiralTag": lambda atom: atom.GetChiralTag(),
        "TotalDegree": lambda atom: atom.GetTotalDegree(),
        "FormalCharge": lambda atom: atom.GetFormalCharge(),
        "TotalNumHs": lambda atom: atom.GetTotalNumHs(),
        "RadicalElectrons": lambda atom: atom.GetNumRadicalElectrons(),
        "Hybridization": lambda atom: str(atom.GetHybridization()),
        "IsAromatic": lambda atom: atom.GetIsAromatic(),
        "IsInRing": lambda atom: atom.IsInRing()
    }

    _bond_features: dict = {
        "BondType": lambda bond: str(bond.GetBondType()),
        "Stereo": lambda bond: str(bond.GetStereo()),
        "IsConjugated": lambda bond: bond.GetIsConjugated()
    }

    def __init__(self, feature_config: Dict[str, Dict], n_jobs: int = 1, verbose: bool = False):
        """
        Instantiates a MolFeaturizer by parsing the configuration dictionary (which atom and bond features to encode).

        Args:
            feature_config: Dictionary of {"AtomFeatures": {...}, "BondFeatures": {...}}, specifying which atom and bond
                            features should be encoded, and which value ranges are encoded in the model.
        """
        self._node_features, self._num_node_features = self._parse_config(feature_config.get("AtomFeatures", {}))
        self._edge_features, self._num_edge_features = self._parse_config(feature_config.get("BondFeatures", {}))

        self._n_jobs: int = n_jobs
        self._verbose: bool = verbose

    def _parse_config(self, feature_config: Dict[str, List]) -> Tuple[Dict, int]:
        """
        Parses a configuration Dictionary for Atom or Bond Features.

        Args:
            feature_config: Dictionary of AtomFeatures and allowed values (for one-hot encoding).
                            e.g.  "AtomSymbol": ["H", "C", "N", "O"]

        Returns:
            dict: Dictionary of features and possible values for that atom feature.
            int: Total number of encoded features.
        """
        value_count: int = 0

        for feature, values in feature_config.items():
            to_add = (len(values) + 1) if len(values) > 1 else 1
            value_count += to_add

        return feature_config, value_count

    def _encode_feature(
            self,
            object: Union[Chem.Atom, Chem.Bond],
            feature_type: str,
            feature_name: str,
            options: list
    ) -> np.array:
        """
        Encodes each atom/bond feature as a one-hot-encoded vector. All options that are not contained in the set of options
        are encoded as a single category.

        Args:
            object: RDKit Atom or Bond object
            feature_type: "Atom" (for atom features) / "Bond" (for bond features)
            feature_name: Name of the feature (from the class attribute dictionary).
            options: List of n possible values for that feature.

        Returns:
             np.array: 1D-Array (length n+1), one-hot-encoded feature value.
        """
        if feature_type == "Atom":
            value = self._atom_features[feature_name](object)
        elif feature_type == "Bond":
            value = self._bond_features[feature_name](object)

        if isinstance(value, bool):
            return np.asarray([value])

        else:
            return np.append(value == np.asarray(options), value not in options)

    def _encode_molecule(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        Encodes a single molecule as input for a graph neural network.

        Args:
            mol: RDKit Mol object.

        Returns:
            Dict[str, np.ndarray]: Dictionary of features for {"nodes", "edges", "globals", "senders", "receivers"} (as
                                   required for the graph_nets package).
        """
        nodes: np.ndarray = np.zeros((mol.GetNumAtoms(), self._num_node_features))
        for idx, atom in enumerate(mol.GetAtoms()):
            nodes[idx, :] = np.hstack([self._encode_feature(atom, "Atom", feature, values) for feature, values in self._node_features.items()])

        edges: np.ndarray = np.zeros((mol.GetNumBonds() * 2, self._num_edge_features))
        senders: np.ndarray = np.zeros(2 * mol.GetNumBonds())
        receivers: np.ndarray = np.zeros(2 * mol.GetNumBonds())
        for idx, bond in enumerate(mol.GetBonds()):
            start_idx, end_idx = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
            bond_features = np.hstack([self._encode_feature(bond, "Bond", feature, values) for feature, values in self._edge_features.items()])
            edges[2 * idx, :] = bond_features
            edges[2 * idx + 1, :] = bond_features
            senders[2 * idx : 2 * idx + 2] = np.asarray([start_idx, end_idx])
            receivers[2 * idx : 2 * idx + 2] = np.asarray([end_idx, start_idx])

        data = {
            "nodes": nodes.astype(np.float32),
            "edges": edges.astype(np.float32),
            "globals": np.array([0.], dtype=np.float32),
            "senders": senders.astype(np.float32),
            "receivers": receivers.astype(np.float32)
        }

        return data

    def encode_molecules(
            self,
            molecules: List[str],
            batch_size: Optional[int] = None
    ) -> Union[graph_nets.graphs.GraphsTuple, List[graph_nets.graphs.GraphsTuple]]:
        """
        Encodes a list of molecules (given as SMILES strings) into the GraphsTuple object required for the
        graph_nets library. Supports batch encodings, returns a list of GraphsTuple objects if batch_size is not None.

        Args:
            molecules: List of SMILES strings of molecules.
            batch_size: Batch size for batched encoding.

        Returns:
            GraphsTuple: graph_nets object (or list of graph_nets objects) that can be passed to the GNN encoder.
        """
        if batch_size is None:
            batch_size = len(molecules)

        n_batches: int = int(np.ceil(len(molecules) / batch_size))
        molecules: np.array = np.asarray(molecules)

        encdoded_graphs: list = []
        for i in tqdm(range(n_batches), desc="Encoding Batches of Molecules", disable=not self._verbose):
            batch: np.array = molecules[i * batch_size : min((i + 1) * batch_size, len(molecules))]
            mol_objects: list = Parallel(n_jobs=self._n_jobs)(delayed(lambda x: Chem.MolFromSmiles(x))(smiles) for smiles in batch)
            mol_encodings: list = Parallel(n_jobs=self._n_jobs)(delayed(self._encode_molecule)(mol) for mol in mol_objects)
            encdoded_graphs.append(graph_nets.utils_tf.data_dicts_to_graphs_tuple(mol_encodings))

        if n_batches == 1:
            return encdoded_graphs[0]
        else:
            return encdoded_graphs

