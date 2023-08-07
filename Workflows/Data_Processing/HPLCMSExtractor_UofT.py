import json
import pickle
from pathlib import Path
from typing import Dict, List, Union, Tuple
import numpy as np
from .HPLCMSExtractor import HPLCMSExtractor


class ThermoExtractor(HPLCMSExtractor):
    """
    Implementation of the HPLCMSExtractor Abstract Base Class for the ThermoFisher HPLC-MS in the MatterLab at UofT.
    The data extractor is specific to the output data format of the current Python implementation by Kazu & Tony.

    Column information is specified as a json file that is passed to the constructor.
    """
    def __init__(self, column_info: Path):
        super().__init__()
        self._column_info: dict = json.load(open(column_info, "r"))
        self.pkl: Dict = {}

    @staticmethod
    def _find_output_files(data_dir: Path, job_name: str) -> Tuple[Path, Path]:
        """
        Finds the .pkl file and the .raw file by searching through the input directory and returning the most recent
        files with the correct names.

        Args:
            data_dir: Path to the input directory.
            job_name: Name of the HPLC-MS run

        Returns:
            Path: Path to the .pkl file
            Path: Path to the .raw file
        """
        matching_pkl_files: dict = {}
        matching_raw_files: dict = {}

        for file in data_dir.rglob(f"{job_name}*.pkl"):
            matching_pkl_files[file.stat().st_ctime] = file
        for file in data_dir.rglob(f"{job_name}*.raw"):
            matching_raw_files[file.stat().st_ctime] = file

        if all([matching_pkl_files, matching_raw_files]):
            return matching_pkl_files[max(matching_pkl_files.keys())], matching_raw_files[max(matching_raw_files.keys())]
        else:
            raise FileNotFoundError(f"The files for {job_name} could not be found")

    def _extract_dad_data(self, data_dir: Path, job_name: str) -> Dict[str, np.ndarray]:
        """
        Extracts the DAD data from the .pkl file.

        Args:
            data_dir: Path to the input directory.
            job_name: Name of the HPLC-MS run

        Returns:
            dict: Dictionary containing the DAD data.
        """
        pkl_path, _ = self._find_output_files(data_dir, job_name)
        with open(pkl_path, "rb") as pkl_file:
            pkl = pickle.load(pkl_file)
        return pkl["DAD_data"]["3D_fields"]

    def _extract_ms_data(self, data_dir: Path, job_name: str) -> Path:
        """
        Returns the .raw file corresponding to the HPLC-MS run.

        Args:
            data_dir: Path to the input directory.
            job_name: Name of the HPLC-MS run

        Returns:
            Path: Path to the .raw file
        """
        _, raw_path = self._find_output_files(data_dir, job_name)
        return raw_path

    def _extract_gradient_info(self, data_dir: Path, job_name: str) -> Dict[str, Union[List[Dict], Dict]]:
        """
        Extracts the gradient information from the .pkl file.

        Args:
            data_dir: Path to the input directory.
            job_name: Name of the HPLC-MS run

        Returns:
            dict: Dictionary containing the gradient information.
        """
        pkl_path, _ = self._find_output_files(data_dir, job_name)
        with open(pkl_path, "rb") as pkl_file:
            pkl = pickle.load(pkl_file)

        gradient_info = pkl["DAD_data"]["Metadata"]["solvent_gradient"]
        eluents = {'A': pkl["DAD_data"]["Metadata"]["aqueous_phase"], 'B': pkl["DAD_data"]["Metadata"]["organic_phase"]}

        for time_pt in gradient_info:
            time_pt["Time"] = time_pt.pop("time")
            time_pt["PercentB"] = time_pt.pop("solvent_B_proportion")
            time_pt["PercentA"] = 100 - time_pt["PercentB"]
            time_pt["Flow"] = time_pt.pop("flow_rate")

        return {"Timetable": gradient_info, "Eluents": eluents}

    def _extract_column_info(self, data_dir: Path, job_name: str) -> Dict[str, Union[str, int, float]]:
        """
        Returns the column information.

        Args:
            data_dir: Path to the input directory (just here for consistency with the parent class).
            job_name: Name of the HPLC-MS run (just here for consistency with the parent class).

        Returns:
            dict: Dictionary containing the column information.
        """
        return self._column_info

    def _extract_metadata(self, data_dir: Path, job_name: str) -> Dict[str, dict]:
        """
        Extracts the metadata from the .pkl file.

        Args:
            data_dir: Path to the input directory.
            job_name: Name of the HPLC-MS run.

        Returns:
            dict: Dictionary containing the metadata (sub-keys: Job_information, DAD_metadata, MS_metadata,
                  Experiment_settings).
        """
        pkl_path, _ = self._find_output_files(data_dir, job_name)
        with open(pkl_path, "rb") as pkl_file:
            pkl = pickle.load(pkl_file)

        metadata = {
            "Job_information": pkl["job"],
            "DAD_metadata": pkl["DAD_data"]["Metadata"],
            "MS_metadata": pkl["MS_data"]["Metadata"],
            "Experiment_settings": pkl["experiment_setting"]
        }

        return metadata


