from typing import Dict, Tuple, Union, List, Any
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import time
from molarinterface import MolarInterface


FRAGMENTS = ("fragment_a", "fragment_b", "fragment_c")


def process_successful_syntheses(
        database_entries: pd.DataFrame,
        targets: Dict[str, Union[float, None]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Processes the database entries of completed syntheses and returns the HIDs, SMILES and Objectives of all successful
    entries as np.ndarrays. If the validation status of a synthesis is "poor emission", the objective value is set to a
    random value between 0.1 * min_objective_value and min_objective_value.

    Args:
        database_entries: DataFrame containing all database entries of completed syntheses.
        targets: Dictionary of type {objective_name: min_objective_value} containing the minimum value to set the
                    objective of poorly emissive samples.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the HIDs, SMILES and objectives of all successful syntheses.
        Dict[str, np.ndarray]: Dictionary containing the HIDs and SMILES of all syntheses.
    """
    hids, smiles, all_targets, failed_hids, failed_smiles = [], [], [], [], []

    for _, row in database_entries.iterrows():
        row = row.to_dict()
        validation_status = row["product.optical_properties"].get("validation_status", None)

        if validation_status is True:
            try:
                hid, smiles_str = row["product.hid"], row["product.smiles"]
                target_values = [row["product.optical_properties"].get(obj, np.nan) for obj in targets]
                hids.append(hid), smiles.append(smiles_str), all_targets.append(target_values)
            except KeyError:
                logging.warning(f"Could not find HID / SMILES in for product {row['product.hid']}")
                failed_hids.append(row["product.hid"])
                failed_smiles.append(row["product.smiles"])

        # All "poor emission" samples are assigned a random value between 0.1 * min_objective_value and
        # min_objective_value
        elif validation_status == ["poor emission"]:
            try:
                hid, smiles_str = row["product.hid"], row["product.smiles"]
                target_values = []
                for min_value in targets.values():
                    if min_value is None:
                        target_values.append(np.nan)
                    else:
                        target_values.append(np.random.uniform(0.1 * min_value, min_value))
                hids.append(hid), smiles.append(smiles_str), all_targets.append(target_values)
            except KeyError:
                logging.warning(f"Could not process {row['product.hid']}")
                failed_hids.append(row["product.hid"])
                failed_smiles.append(row["product.smiles"])

        # All other validation statuses are considered "failed"
        else:
            failed_hids.append(row["product.hid"])
            failed_smiles.append(row["product.smiles"])

    successful_syntheses: dict = {
        "hid": np.asarray(hids),
        "smiles": np.asarray(smiles),
        "targets": np.asarray(all_targets)
    }

    all_syntheses: dict = {
        "hid": np.concatenate([np.asarray(failed_hids), np.asarray(hids)]),
        "smiles": np.concatenate([np.asarray(failed_smiles), np.asarray(smiles)])
    }

    return successful_syntheses, all_syntheses


def process_incomplete_syntheses(*database_entries) -> Dict[str, np.ndarray]:
    """
    Extracts the HIDs of all failed syntheses.

    Args:
        database_entries: DataFrames containing all database entries of failed / incomplete syntheses.

    Returns:
        np.ndarray: Array containing the HIDs and SMILES of all incomplete syntheses.
    """
    incomplete_syntheses: dict = {
        "hid": np.concatenate([np.asarray(df["product.hid"]) for df in database_entries]),
        "smiles": np.concatenate([np.asarray(df["product.smiles"]) for df in database_entries])
    }

    return incomplete_syntheses


def get_available_fragments(*labs) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Retrieves the available fragments from the database for the given labs.

    Args:
        *labs: List of labs for which to retrieve the available fragments.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: Dictionary containing the available fragments for each lab. The dictionary
            has the following structure:
            {
                "lab_1": {
                    "fragment_a": {"A001": SMILES, "A002": SMILES, "A003": SMILES, ...},
                    "fragment_b": {"B001": SMILES, "B002": SMILES, "B003": SMILES, ...},
                    "fragment_c": {"C001": SMILES, "C002": SMILES, "C003": SMILES, ...}
                },
                "lab_2": {...},
                ...
            }
    """
    db_interface = MolarInterface("madness_laser", fragments=FRAGMENTS)

    available_fragments = {}
    for lab in labs:
        local_fragments = {}
        for frag in FRAGMENTS:
            local_fragments_per_type = db_interface.get_available_fragments(frag, lab)
            local_fragments[frag] = dict(zip(local_fragments_per_type["molecule.hid"], local_fragments_per_type["molecule.smiles"]))
        available_fragments[lab] = local_fragments

    return available_fragments


def get_all_previous_observations(
        objectives: Dict[str, Union[float, None]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """

    Args:
        objectives: Dictionary of type {objective_name: min_objective_value} containing the minimum value to set the
                    objective of poorly emissive samples.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing the HIDs, SMILES and targets of all successful observations.
        Dict[str, np.ndarray]: Dictionary containing the HIDs and SMILES of all pending observations.
        Dict[str, np.ndarray]: Dictionary containing the HIDs and SMILES of all non-assigned observations.
        Dict[str, np.ndarray]: Dictionary containing the HIDs and SMILES of all acquisition constraints (i.e.
                               successful, unsuccessful and pending observations).
    """
    db_interface = MolarInterface("madness_laser", fragments=FRAGMENTS)
    all_observations: pd.DataFrame = db_interface.get_all_syntheses()

    not_assigned = process_incomplete_syntheses(
        all_observations[all_observations["synthesis.status"] == "AVAILABLE"]
    )

    pending = process_incomplete_syntheses(
        all_observations[all_observations["synthesis.status"] == "ACQUIRED"]
    )

    failed = process_incomplete_syntheses(
        all_observations[all_observations["synthesis.status"] == "FAILED"]
    )

    completed_successfully, completed_all = process_successful_syntheses(
        all_observations[all_observations["synthesis.status"] == "DONE"],
        objectives
    )

    acquisition_constraints = {
        "hid": np.concatenate([pending["hid"], failed["hid"], completed_all["hid"]]),
        "smiles": np.concatenate([pending["smiles"], failed["smiles"], completed_all["smiles"]])
    }

    return completed_successfully, pending, not_assigned, acquisition_constraints


def upload_new_recommendations(
        hids: np.ndarray,
        smiles: np.ndarray
) -> None:
    """
    Uploads the new recommendations to the database.

    Args:
        hids: Array containing the HIDs of the new recommendations.
        smiles: Array containing the SMILES of the new recommendations.
    """
    db_interface = MolarInterface("madness_laser", fragments=FRAGMENTS)

    for hid, smiles in zip(hids, smiles):
        uuids = db_interface.create_target_compound(
            fragments=[hid[0:4], hid[4:8], hid[8:12]],
            smiles=smiles,
            identifier_type="hid",
            create_synthesis=True,
            procedure="gen_2"
        )
        logging.debug(f"Created new database entry for {hid} with UUIDs: {uuids}.")
        time.sleep(1)


def remove_recommendations(
        hids: np.ndarray
) -> None:
    """
    Removes the unclaimed recommendations from the database.

    Args:
        hids: Array containing the HIDs of the unclaimed recommendations.
    """
    db_interface = MolarInterface("madness_laser", fragments=FRAGMENTS)

    for hid in hids:
        synthesis_id = db_interface._get_synthesis_uuid_from_molecule(hid)
        db_interface.delete_entry(table="synthesis", identifier=synthesis_id, identifier_type="synthesis_id")


def upload_to_database(
        hid: str,
        smiles: str,
        hplc_data: Path,
        spectroscopy_data: Dict[str, Any],
) -> None:
    """
    Uploads new synthesis and characterization data to the database.

    Args:
        hid: HID of the compound to be updated.
        smiles: SMILES string of the compound to be updated.
        hplc_data: Path to the processed HPLC-MS data file.
        spectroscopy_data: Dictionary containing the processed spectroscopy data.
    """
    db_interface = MolarInterface("madness_laser", fragments=FRAGMENTS)

    db_interface.create_target_compound(
        (hid[0:4], hid[4:8], hid[8:12]),
        smiles,
        identifier_type="hid",
        procedure="gen_2"
    )

    db_interface.upload_hplc_data(hplc_data, hid)
    db_interface.update_optics_data(hid, spectroscopy_data)

