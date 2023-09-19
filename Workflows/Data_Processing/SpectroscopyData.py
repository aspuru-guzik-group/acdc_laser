import json
import pickle
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import numpy as np
from scipy.constants import c, pi


class SpectroscopyData(object):
    """
    Wrapper for spectroscopic data for OSLD characterization, as obtained from the automated spectroscopy setup
    described in https://doi.org/10.1002/adma.202207070. Performs automated data analysis and sanity checking.

    Basic usage:

        data = SpectroscopyData(raw_data_file, settings_file)
        results: dict = data()
    """

    def __init__(self, raw_data_dir: Path, job_name: str, settings: Path):
        self._raw_data_dir = raw_data_dir
        raw_data = self._get_raw_data(job_name)

        self._settings = json.load(open(settings, "r"))

        self._properties: set = set()
        self._results: dict = {}

        for key, analysis_settings in self._settings.items():
            for prop in analysis_settings.get("properties", []):
                self._properties.add(prop)
                setattr(self, prop, self._results[key].get(prop, None))

        self._results["absorption"] = self._process_absorption_spectrum(raw_data["absorption"])
        self._results["emission"] = self._process_emission_spectrum(raw_data["PL"])
        self._results["time_trace"] = self._process_time_trace(raw_data["UV"], raw_data["PL"])
        self._results["transient_emission"] = self._process_transient_emission(raw_data["TE"])
        self._results["gain"] = self._calculate_gain_cross_section()
        self._validation_status = self._validate()

    def __call__(self) -> Dict[str, Any]:
        """
        Returns a dictionary with all spectroscopic analysis results. Loops through all properties and writes the
        corresponding values to the dictionary (if they are not None).

        Returns:
            Dict: Dictionary with all spectroscopic analysis results.
        """
        results: dict = {}

        for key, value in self._properties:
            if value is not None:
                results[key] = value

        results["validation_status"] = self._validation_status

        return results

    def _get_raw_data(self, job_name: str) -> Dict[str, Any]:
        """
        Finds the raw data files for the given job name and returns them as a dictionary.

        Args:
            job_name: Name of the job.

        Returns:
            Dict: Dictionary of raw data, as returned by the automated optics workflow. .
        """
        matching_files: dict = {}

        for file in self._raw_data_dir.rglob(f"{job_name}*.pkl"):
            matching_files[file.stat().st_ctime] = file

        if len(matching_files) == 0:
            raise FileNotFoundError(f"No raw data files found for job {job_name} in {self._raw_data_dir}.")

        with open(matching_files[max(matching_files.keys())], "rb") as f:
            raw_data = pickle.load(f)

        return raw_data

    def _process_absorption_spectrum(self, raw_data: Dict) -> Dict[str, float]:
        """
        Processes raw absorption spectra by
            1. subtracting the baseline (evaluated in a pre-defined wavelength range)
            2. Identifying the maximum absorption peak


        Args:
            raw_data: Dictionary of raw absorption data, as returned by the automated spectroscopy setup from
                      https://doi.org/10.1002/adma.202207070. Required keys: "absorbance" (contains a 2D numpy array
                      with wavelengths, index 0, and absorbances, index 1), "end_wavelength (contains the longest
                      wavelength for which significant absorption is observed).

        Returns:
            Dict: Dictionary of absorption properties with the following keys: "abs_lambda_max", "abs_max", "abs_300nm",
                  "abs_end".
        """
        wavelengths, absorbances = raw_data["absorbance"][0], raw_data["absorbance"][1]
        evaluation_range = self._settings["absorption"]["evaluation_range"]
        baseline_range = self._settings["absorption"]["baseline_range"]

        baseline_indices: np.ndarray = np.where(np.logical_and(wavelengths > baseline_range[0], wavelengths < baseline_range[1]))
        baseline_value = np.mean(absorbances[baseline_indices])

        range_full: np.ndarray = np.where(np.logical_and(wavelengths > evaluation_range[0], wavelengths < evaluation_range[1]))
        range_300nm: np.ndarray = np.where(np.logical_and(wavelengths > 300, wavelengths < 301))

        wavelengths_full, absorbance_full = wavelengths[range_full], absorbances[range_full] - baseline_value
        wavelengths_300nm, absorbance_300nm = wavelengths[range_300nm], absorbances[range_300nm] - baseline_value

        max_absorbance_idx_full = np.argmax(absorbance_full)
        max_absorbance_idx_300nm = np.argmax(absorbance_300nm)

        absorption_results = {
            "abs_lambda_max": wavelengths_full[max_absorbance_idx_full],
            "abs_max": absorbance_full[max_absorbance_idx_full],
            "abs_300nm": absorbance_300nm[max_absorbance_idx_300nm],
            "abs_end": raw_data.get("end_wavelength", None)  # TODO: Implement this here rather than relying on Kazu's results
        }

        return absorption_results

    def _process_emission_spectrum(self, raw_data: Dict) -> Dict[str, float]:
        """
        Processes raw photoluminescence spectra by
            1. subtracting the baseline (evaluated in a pre-defined wavelength range)
            2. Identifying the maximum emission peak
            3. Computing the gain spectrum as lambda ** 4 * emission spectrum / (8 * pi * c * emission integral)
            4. Identifying the maximum gain peak

        Args:
            raw_data: Dictionary of raw photoluminescence data, as returned by the automated spectroscopy setup from
                      https://doi.org/10.1002/adma.202207070. Required keys: "photons" (contains a 2D numpy array of the
                      wavelengths, index 0, and the emission intensities, index 1).

        Returns:
            Dict: Dictionary of photoluminescence properties with the following keys: "PL_lambda_max",
                  "PL_max_intensity", "PL_integral", "gain_at_PL_lambda_max", "max_gain_wl", "max_gain_factor"
        """
        if raw_data is None or "photons" not in raw_data:
            return {}

        wavelengths, intensities = raw_data["photons"][0], raw_data["photons"][1]
        baseline_range = self._settings["emission"]["baseline_range"]
        evaluation_range = self._settings["emission"]["evaluation_range"]

        baseline_indices: np.ndarray = np.where(np.logical_and(wavelengths > baseline_range[0], wavelengths < baseline_range[1]))
        baseline_value = np.mean(intensities[baseline_indices])

        range_full: np.ndarray = np.where(np.logical_and(wavelengths > evaluation_range[0], wavelengths < evaluation_range[1]))
        wavelengths_full, intensities_full = wavelengths[range_full], intensities[range_full] - baseline_value

        emission_integral = np.trapz(intensities_full, wavelengths_full)
        gain_spectrum = (wavelengths ** 4 * intensities / (8 * pi * c * emission_integral)) * 1e4  # convert to cm^2 s

        max_intensity_idx = np.argmax(intensities_full)
        max_gain_idx = np.argmax(gain_spectrum)

        emission_results = {
            "PL_lambda_max": wavelengths_full[max_intensity_idx],
            "PL_max_intensity": intensities_full[max_intensity_idx],
            "PL_integral": emission_integral,
            "gain_at_pl_lambda_max": gain_spectrum[max_intensity_idx],
            "max_gain_wl": wavelengths_full[max_gain_idx],
            "max_gain_factor": gain_spectrum[max_gain_idx],
        }

        return emission_results

    def _process_time_trace(self, raw_data_uv: Dict, raw_data_pl: Dict) -> Dict[str, float]:
        """
        Extracts the relevant time trace analysis results from the raw data provided by the automated spectroscopy setup
        from https://doi.org/10.1002/adma.202207070.

        Args:
            raw_data_uv: Dictionary of raw UV data, as returned by the automated spectroscopy setup.
                        Required keys: "relative_QY", "degradation_rate"
            raw_data_pl: Dictionary of raw photoluminescence data, as returned by the automated spectroscopy setup.

        Returns:
            Dict: Dictionary of time trace analysis results with the following keys: "relative_QY",
                  "abs_degradation_rate", "PL_degradation_rate"
        """
        return {
            "relative_QY": raw_data_uv.get("relative_QY", None),
            "abs_degradation_rate": raw_data_uv.get("degradation_rate", None),
            "PL_degradation_rate": raw_data_pl.get("degradation_rate", None),
        }

    def _process_transient_emission(self, raw_data: Dict) -> Dict[str, Optional[float]]:
        """
        Extracts the relevant transient emission analysis results from the raw data provided by the automated
        spectroscopy setup from https://doi.org/10.1002/adma.202207070.

        Args:
            raw_data: Dictionary of transient emission results, as returned by the automated spectroscopy setup.

        Returns:
            Dict: Dictionary of transient emission analysis results with the following keys: "tau1", "tau2"

        """
        tau1, tau2 = None, None

        if raw_data.get("fitting_results") is not None:
            if "tau" in raw_data["fitting_results"]:
                tau1 = raw_data["fitting_results"]["tau"]
            elif "tau1" in raw_data["fitting_results"]:
                tau1 = raw_data["fitting_results"]["tau1"]
                tau2 = raw_data["fitting_results"]["tau2"]

        return {"tau1": tau1, "tau2": tau2}

    def _calculate_gain_cross_section(self) -> Dict[str, Optional[float]]:
        """
        Calculates the gain cross section from the emission, time trace and transient emission data.
        Returns the values at the maximum emission wavelength an the maximum gain wavelength.

        Returns:
            Dict: Dictionary of gain cross section analysis results with the following keys:
                  "gain_cross_section_at_PL_lambda_max", "gain_cross_section_at_max_gain_wl"
        """
        refractive_index = self._settings["gain_cross_section"]["refractive_index"]

        if not all([hasattr(self, i) for i in ["relative_QY", "tau1", "gain_at_pl_lambda_max", "max_gain_factor"]]):
            return {
                "gain_cross_section_at_PL_lambda_max": None,
                "gain_cross_section": None
            }

        if any([i is None for i in [self.relative_QY, self.tau1, self.gain_at_pl_lambda_max, self.max_gain_factor]]):
            return {
                "gain_cross_section_at_PL_lambda_max": None,
                "gain_cross_section": None
            }

        wl_independent_factor = self.relative_QY / (refractive_index ** 2 * self.tau1)

        return {
            "gain_cross_section_at_PL_lambda_max": self.gain_at_pl_lambda_max * wl_independent_factor,
            "gain_cross_section": self.max_gain_factor * wl_independent_factor
        }

    def _validate(self) -> Union[bool, List[str]]:
        """
        Validates all data analysis results, and returns the validation status.

        Returns:
            True if all validation checks pass, otherwise a list of error codes.
        """
        validation_errors = []

        for key, analysis in self._settings.items():
            for prop, settings in analysis["properties"].items():
                if settings["required"] is True:

                    value = getattr(self, prop)
                    error_code = settings.get("error_code", f"error in {prop}")

                    if value is None or np.isnan(value):
                        validation_errors.append(error_code)
                    elif value < settings["range"][0] or value > settings["range"][1]:
                        validation_errors.append(error_code)

        return validation_errors if len(validation_errors) > 0 else True





