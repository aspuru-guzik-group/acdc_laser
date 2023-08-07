from typing import Dict, Optional
from abc import ABCMeta, abstractmethod
import logging
from functools import wraps

import os
import shutil
from pathlib import Path

import json
import time
import numpy as np


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        logging.debug(f"{f.__name__} took: {round(te-ts, 3)} s")
        return result
    return wrap


class NumpyArrayEncoder(json.JSONEncoder):
    """
    Class to convert numpy objects to json-serializable data types & structures.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


class HPLCMSExtractor(metaclass=ABCMeta):
    """
    Parent class for data extractors for HPLC-MS code. Contains a set of convenient methods for extracting and
    standardizing the data from HPLC-MS runs.
    """
    def __init__(self):
        self.output: Optional[Path] = None

    def __call__(
            self,
            data_dir: Path,
            job_name: str,
            output_dir: Path
    ) -> Path:
        """
        Runs the extraction process.

        Args:
            dad_data: Dictionary containing the DAD data.
            ms_data: Path to the raw MS data.
            gradient_info: Dictionary containing the gradient information.
            column_info: Dictionary containing the column information.

        Returns:
            Path: Path to the output archive.
        """
        self._setup_output_dir(output_dir, job_name)

        # DAD Data
        dad_data = self._extract_dad_data(data_dir, job_name)
        self._save_dad_data(dad_data)

        # MS Data
        ms_data = self._extract_ms_data(data_dir, job_name)
        self._convert_to_mzml(ms_data)

        # Metadata
        metadata = self._extract_metadata(data_dir, job_name)
        metadata["Gradient"] = self._extract_gradient_info(data_dir, job_name)
        metadata["Column"] = self._extract_column_info(data_dir, job_name)
        self._save_metadata(metadata)

        # Create archive and clean up
        archive: Path = self._archive_output()
        time.sleep(1)
        self._del_output()

        return archive

    def _setup_output_dir(self, working_dir: Path, file_name: str) -> None:
        """
        Creates the output directory for the archive and all required subdirectories.

        Args:
            working_dir: Directory in which the archive should be created.
            file_name: Base name of the archive.
        """
        self.output: Path = working_dir / file_name
        for p in [self.output, self.output / "Metadata"]:
            p.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _extract_dad_data(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _extract_ms_data(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _extract_gradient_info(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _extract_column_info(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _extract_metadata(self, *args, **kwargs):
        raise NotImplementedError

    @timing
    def _convert_to_mzml(self, ms_file: Path) -> None:
        """
        Converts  mass spectrometry data from a vendor-specific format to the mzML format.
        Uses the msconvert command line tool from ProteoWizard (https://proteowizard.sourceforge.io/).

        Args:
            ms_file: Path to the input file.
        """
        mzml_file: str = "MS_data"
        msconv_cmd: str = (
            f'cmd /c "msconvert "{ms_file}" -z -o "{self.output}" --outfile {mzml_file} '
            f'--filter "peakPicking vendor" --filter metadataFixer"'
        )
        os.system(msconv_cmd)

    @timing
    def _save_dad_data(self, dad_dict: Dict[str, np.ndarray]) -> None:
        """
        Saves the DAD data to a JSON file.

        Args:
            dad_dict: Dictionary containing the DAD data.
        """
        with open(self.output / "DAD_data.json", "w") as f:
            json.dump(dad_dict, f, cls=NumpyArrayEncoder)

    def _save_metadata(self, experiment_metadata: Dict[str, dict]) -> None:
        """
        Saves the experiment metadata JSON files (one per key in the experiment_metadata dictionary).

        Args:
            experiment_metadata: Dictionary containing the experiment metadata.
        """
        for key, data in experiment_metadata.items():
            with open(self.output / "Metadata" / f"{key}.json", "w") as f:
                json.dump(data, f)

    @timing
    def _archive_output(self) -> Path:
        """
        Creates a compressed archive of the output directory.

        Returns:
            Path to the archive.
        """
        archive: str = shutil.make_archive(
            self.output,
            format="xztar",
            root_dir=self.output.parent,
            base_dir=self.output.name,
        )

        return Path(archive)

    @timing
    def _del_output(self) -> None:
        """
        Deletes the output directory.
        """
        shutil.rmtree(self.output)
        self.output = None

