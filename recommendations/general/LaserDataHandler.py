import numpy as np
import pandas as pd
from typing import Tuple, Callable, List, Dict

from ChemicalReaction import ChemicalReaction
from MolarInterface import MolarInterface


class LaserDataHandler(MolarInterface):

    def __init__(self, db_name: str, fragments: tuple, active_labs: list):
        super().__init__(db_name=db_name, fragments=fragments)

        self.active_labs = active_labs
        self.available_fragments: dict = {lab: {frag: set() for frag in fragments} for lab in active_labs}

        self.all_previous_results = None

    def load_previous_results(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Queries the database for all previously generated syntheses.
        Returns the syntheses in progress (for generating constraints) and the completed syntheses (for generating
        training data).

        Returns:
            in_progress: Dataframe of all syntheses in progress
            completed: Dataframe of all completed syntheses.
        """
        self.all_previous_results = self.get_all_syntheses()

        in_progress = self.all_previous_results[self.all_previous_results["synthesis.status"].isin(["AVAILABLE", "ACQUIRED", "PROCESSING", "SYNTHESIZED", "SHIPPED", "RECEIVED"])]
        completed = self.all_previous_results[self.all_previous_results["synthesis.status"].isin(["DONE", "FAILED"])]

        return in_progress, completed

    def process_previous_results(self, previous_results: pd.DataFrame, get_target_property: Callable) -> Tuple[List[dict], Dict[str, set]]:
        """
        Processes the previous results by extracting the used fragments and the measured objective values.

        Args:
            previous_results: Dataframe of previous results
            get_target_property: Function to access the objective value from a single row

        Returns:
            observations: List of observations, as required for the Gryffin call (each observation as a dictionary)
            used_fragments: Dictionary of used fragments.
        """
        observations: list = []
        used_fragments: dict = {frag: set() for frag in self._fragments}

        for _, row in previous_results.iterrows():
            row = row.to_dict()
            data: dict = dict()

            # Add fragment hids to the data and the overview of used fragments
            for frag in self._fragments:
                data[frag] = row[f"{frag}.hid"]
                used_fragments[frag].add(row[f"{frag}.hid"])

            # Extract target information
            try:
                data["obj"] = get_target_property(row) if get_target_property(row) else np.nan
            except KeyError:
                data["obj"] = np.nan

            observations.append(data)

        return observations, used_fragments

    def get_all_available_fragments(self):

        all_available_fragments = {frag: set() for frag in self._fragments}

        for lab in self.active_labs:
            for frag in self._fragments:
                self.available_fragments[lab][frag] = set(self.get_available_fragments(frag, lab)["molecule.hid"])
                all_available_fragments[frag].update(self.available_fragments[lab][frag])

        return all_available_fragments

    def target_is_makable(self, parameters: dict, *labs) -> bool:
        """
        Checks if a target can be made in a single location (i.e. all fragments are available at one spot).

        Args:
            parameters: Dictionary of parameters (needs the fragments as keys)

        Returns:
            bool: True if the target can be made.
        """
        for lab in labs:
            if all([parameters[frag] in self.available_fragments[lab][frag] for frag in self._fragments]):
                return True
        return False

    def target_is_novel(self, parameters: dict) -> bool:
        """
        Checks if a target is novel (i.e. has never been made before
        Args:
            parameters:

        Returns:

        """
        hid = "".join([parameters[frag] for frag in self._fragments])
        return hid not in self.all_previous_results["product.hid"].values


def run_two_step_suzuki(smiles_a: str, smiles_b: str, smiles_c: str) -> str:
    first_step = ChemicalReaction(
        "[C,c:1]B([OH])[OH].[C,c:2][Br,I]>>[C,c:1][C,c:2]",
        "[C,c:1]B1OC(C)(C)C(C)(C)O1.[C,c:2][Br,I]>>[C,c:1][C,c:2]"
    )
    second_step = ChemicalReaction(
        "[C,c:1]B1OC(=O)CN(C)CC(=O)O1.[C,c:2][Br,1]>>[C,c:1][C,c:2]"
    )
    intermediate: str = first_step(smiles_a, smiles_b)
    product: str = second_step(intermediate, smiles_c)

    return product
