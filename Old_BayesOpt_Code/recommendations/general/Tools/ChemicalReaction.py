import itertools
import random
import string
from typing import List, Tuple, Optional, Union
import networkx
import numpy as np
from networkx.exception import NodeNotFound, NetworkXNoPath
from networkx import DiGraph
from rdkit import Chem
from rdkit.Chem import rdChemReactions


class ChemicalReaction:

    """
    Class for Describing and Enumerating Simple and more Complex Chemical Reactions.
    Based on a Reaction-SMARTS, possible reaction products and intermediates are enumerated using a reaction network
    approach (implemented using RdKit and NetworkX).
    """

    def __init__(self, *reaction_smarts):
        """
        Instantiates a ChemicalReaction object by setting up a list of RDKit reaction objects (reactions), and a
        list of matching SMARTS patterns per reaction..

        Args:
            reaction_smarts: SMARTS string(s) describing the chemical transformation(s).

        ATTN: Probably, the implementation with nested iterations is not the most efficient for larger networks
              Think about a numpy-ish implementation of the whole thing (np.vectorize(lambda x: ...))
        """
        to_reaction = np.vectorize(lambda x: rdChemReactions.ReactionFromSmarts(x))
        mol_from_smarts = np.vectorize(lambda x: Chem.MolFromSmarts(x))

        self.reactions: np.ndarray = to_reaction(reaction_smarts)
        self.reactants: np.ndarray = np.asarray(
            mol_from_smarts([smarts.split(">>")[0].split(".") for smarts in reaction_smarts])
        )

        self.network = None

    def __call__(self, *reactants, max_iterations=5) -> Union[str, List[str]]:
        return self.generate_reaction_network(*reactants, max_iterations=max_iterations)

    def generate_reaction_network(
            self,
            *reactants,
            max_iterations: Optional[int] = 5,
            return_type: str = "product"
    ) -> Union[str, List[str], DiGraph]:
        """
        Generates a reaction network starting from the reactant nodes through iterative network expansion.
        Network expansion is terminated when
            - the network is not expanded in an iteration
            - the maximum number of iterations has been reached

        Args:
             reactants: List of SMILES strings of reactants to construct the reaction network from.
             max_iterations (optional): Maximum iterations to generate the network.
             return_type (optional): What to return ("product" -> SMILES of product(s) as str or list;
                                                     "compounds" -> SMILES of all intermediates as list;
                                                     "network" -> NetworkX.DiGraph object)
        """
        self.network = DiGraph()

        for smiles in reactants:
            self._add_node(smiles)

        graph_size = 0
        for _ in range(max_iterations):
            self._expand_network()

            if self.network.number_of_nodes() == graph_size:
                break
            graph_size = self.network.number_of_nodes()

        if return_type == "product":
            products = [node for node in self.network.nodes() if self.network.out_degree(node) == 0]
            return products[0] if len(products) == 1 else products
        elif return_type == "compounds":
            return [node for node in self.network if self.network.nodes[node]["type"]]
        elif return_type == "network":
            return self.network

    def _expand_network(self) -> None:
        """
        Expands the reaction network by evaluating all possible permutations between reaction nodes.
        """
        # Per reactant, get a list of nodes which match the SMARTS pattern and can serve as reactant
        # ATTN: This Loop should be converted to numpy
        reaction_matches = []
        for i, reaction_smarts in enumerate(self.reactants):
            reactant_matches = []
            for j in range(len(reaction_smarts)):
                matches = [
                    node for node in self.network
                    if self.network.nodes[node]["type"] and self.network.nodes[node]["smarts"][i][j]
                ]
                reactant_matches.append(matches)
            reaction_matches.append(reactant_matches)

        # Enumerate all possible reactant combinations, generate the products and, if applicable, expand the network
        # ATTN: Is there a way to convert this loop to numpy (given that itertools is more efficient than np?)
        for i, reaction in enumerate(reaction_matches):
            for reactants in itertools.product(*reaction):
                possible_products = self.reactions[i].RunReactants([self.network.nodes[mol]["mol"] for mol in reactants])
                for products in possible_products:
                    products = [Chem.MolToSmiles(mol) for mol in products]
                    self._check_add_reaction(reactants, products)

    def _check_add_reaction(self, reactants: tuple, products: list) -> None:
        """
        Checks if a reaction from reactants to products is in the graph (i.e. if there is a path from each reactant
        to each product with the Path weight of 1). If not, calls the method for adding a new reaction.

        Args:
              reactants: Tuple of SMILES strings of the reactants
              products: Tuple of SMILES strings of the products
        """

        for reactant_prod_permutation in itertools.product(reactants, products):
            try:
                if not networkx.shortest_path_length(self.network, *reactant_prod_permutation, weight="weight") == 1:
                    return self._add_reaction(reactants, products)
            except (NodeNotFound, NetworkXNoPath):
                return self._add_reaction(reactants, products)

    def _add_reaction(self, reactants: Tuple, products: List) -> None:
        """
        Adds a new reaction to the graph by
            - creating a dummy node
            - adding the reactant / product nodes, if they don't exist
            - adding the directed paths between reactants and dummy node (weight = 0), and the dummy node and products
              (weight = 1)

        Args:
            reactants: Tuple of SMILES strings of the reactants
            products: Tuple of SMILES strings of the products
        """
        rxn_node = self._add_dummy_node()

        for molecule in reactants:
            self._add_node(molecule)
            self.network.add_edge(molecule, rxn_node, weight=0)

        for molecule in products:
            self._add_node(molecule)
            self.network.add_edge(rxn_node, molecule, weight=1)

    def _add_node(self, smiles: str) -> None:
        """
        Adds a node to the reaction network, if it does not already exist.
        Sets the node features "mol_object" (Chem.Mol) and i (bool for i in range(self.reactants)).

        Args:
            smiles: Canonical SMILES of the molecule to be added.
        """
        if smiles not in self.network:

            mol_object = Chem.MolFromSmiles(smiles)
            has_mol_match = np.vectorize(lambda x: mol_object.HasSubstructMatch(x))
            smarts_matches: np.ndarray = has_mol_match(self.reactants)

            self.network.add_node(smiles, type="molecule", mol=mol_object, smarts=smarts_matches)

    def _add_dummy_node(self) -> str:
        """
        Adds a dummy node to the graph in self.network. Generates a random 10-character identifier for the node,
        and returns the identifier.

        Returns:
             str: Generated node identifier
        """
        identifier = "__" + "".join(random.choices(string.ascii_lowercase, k=8))
        if not identifier in self.network:
            self.network.add_node(identifier, type=None)
            return identifier
        else:
            return self._add_dummy_node()