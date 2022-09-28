from typing import Dict, List
from pathlib import Path
import pandas as pd
from Tools.FileHandling import load_json, load_pkl, save_pkl
from Tools.LaserDataHandler import target_is_novel, target_is_makable
import numpy as np
from gryffin import Gryffin


def run_gryffin(
        config: dict,
        descriptors: Dict[str, Dict[str, list]],
        observations: [List[dict]],
        all_recommendations: pd.DataFrame,
        available_fragments: pd.DataFrame,
) -> List[dict]:
    """
    Instantiates Gryffin and runs the recommend method to generate a new batch of experimental recommendations
    according to the settings passed in the configuration.

    Args:
        config: Specified Gryffin configuration
        descriptors: Dictionary of all fragments (sorted by fragment type) and descriptors.
        observations: List of observations for Gryffin.
        all_recommendations: Dataframe of all previous recommendations.
        available_fragments: Dataframe of all available fragments.

    Returns:
        List[dict]: List of experimental recommendations to generate.
    """
    gryffin_config: dict = {
        "general": config["gryffin_settings"],
        "parameters": [],
        "objectives": [{"name": "obj", "goal": config["optimization_type"]}]
    }

    for parameter in descriptors:
        gryffin_config["parameters"].append(
            {
                "name": parameter,
                "type": "categorical",
                "options": list(descriptors[parameter].keys()),
                "category_details": descriptors[parameter]
            }
        )

    gryffin = Gryffin(
        config_dict=gryffin_config,
        known_constraints=lambda x: (target_is_makable(x, available_fragments, config["active_labs"]) and target_is_novel(x, all_recommendations) and x["procedure"] == "gen_2")
    )

    sampling_strategies = np.concatenate(
        (
            np.linspace(0.5, 1, int(config["no_recommendations"] * config["exploitation_fraction"])),
            np.linspace(-1, 1, config["no_recommendations"] - int(config["no_recommendations"] * config["exploitation_fraction"]))
        )
    )

    recommendations: list = gryffin.recommend(observations, sampling_strategies=sampling_strategies)
    print(f"{len(recommendations)} Recommendations Generated.")

    return recommendations


def load_input() -> tuple:
    """
    Loads the input written by prepare_gryffin.py.

    Returns:
        observations: List of all observations, prepared for Gryffin.
        descriptors: Dictionary of all categorical options and descriptors.
        all_recommendations: Dataframe of all previously made recommendations.
        available_fragments: Dataframe of all available fragments.
    """
    observations = load_pkl(Path.cwd() / "TMP_observations.pkl")
    descriptors = load_pkl(Path.cwd() / "TMP_descriptors.pkl")
    all_recommendations = load_pkl(Path.cwd() / "TMP_previous_results.pkl")
    available_fragments = load_pkl(Path.cwd() / "TMP_available_fragments.pkl")

    return observations, descriptors, all_recommendations, available_fragments


def save_output(recommendations: list) -> None:
    """
    Saves the generated recommendations to a .pkl file.
    Args:
        recommendations: List of all generated recommendations
    """
    save_pkl(recommendations, Path.cwd() / "TMP_new_recommendations.pkl")


if __name__ == "__main__":

    config: dict = load_json(Path(__file__).parent / "gryffin_settings.json")

    observations, fragments_descriptors, all_recommendations, available_fragments = load_input()

    recommendations = run_gryffin(config, fragments_descriptors, observations, all_recommendations, available_fragments)

    save_output(recommendations)
