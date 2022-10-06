import numpy as np
import pandas as pd
from typing import Tuple, Callable, List, Dict
from pathlib import Path
from Tools.FileHandling import save_pkl
from Tools.MolarInterface import MolarInterface


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

        print(f"Currently in Progress: {in_progress.shape[0]}")
        print(f"Completed Experiments: {completed.shape[0]}")

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

            data["procedure"] = row["synthesis.procedure"]

            if get_target_property(row):
                data["obj"] = get_target_property(row)
                observations.append(data)
            elif row["validation_status"] == "failed characterization":
                data["obj"] = 0.0
                observations.append(data)

        print(f"{len(observations)} Observations were created for Gryffin.")
        print(f"Used Fragments:", ", ".join([f"{frag} ({len(used_fragments[frag])})" for frag in used_fragments]))

        return observations, used_fragments

    def get_all_available_fragments(self):

        all_available_fragments = {frag: set() for frag in self._fragments}

        for lab in self.active_labs:
            for frag in self._fragments:
                self.available_fragments[lab][frag] = set(self.get_available_fragments(frag, lab)["molecule.hid"])
                all_available_fragments[frag].update(self.available_fragments[lab][frag])

        return all_available_fragments

    def target_is_makable(self, parameters: dict) -> bool:
        """
        Checks if a target can be made in a single location (i.e. all fragments are available at one spot).

        Args:
            parameters: Dictionary of parameters (needs the fragments as keys)

        Returns:
            bool: True if the target can be made.
        """
        for lab in self.active_labs:
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

    def save_status(self) -> None:
        """
        Saves the available_fragments and all_previous_results attributes as pkl object.
        """
        save_pkl(self.all_previous_results, Path.cwd() / "TMP_previous_results.pkl")
        save_pkl(self.available_fragments, Path.cwd() / "TMP_available_fragments.pkl")



def get_gain_cross_section(data_entry: dict) -> float:
    """
    Method to extract the target gain cross section from the data.

    Args:
        data_entry: Database entry (as dict)

    Returns:
        float: Gain cross section
    """
    if data_entry["synthesis.status"] == "DONE":
        if data_entry["product.optical_properties"].get("validation_status") is True:
            return data_entry["product.optical_properties"].get("gain_cross_section")
        else:
            return np.nan
    else:
        return np.nan

