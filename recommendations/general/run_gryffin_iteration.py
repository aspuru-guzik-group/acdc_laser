from typing import Tuple, List, Dict
import datetime
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from gryffin import Gryffin
from recommendations.general.Tools.LaserDataHandler import LaserDataHandler, get_gain_cross_section
from recommendations.general.Tools.TwoStepSuzuki import run_two_step_suzuki
from Tools.FileHandling import load_json


def process_previous_observations(handler: LaserDataHandler) -> Tuple[List[dict], Dict[str, set]]:
    """
    Loads all previously obtained observations from the data handler.
    Processes all completed experiments and converts them into a list of observations for Gryffin.

    Args:
        handler: LaserDataHandler object (database access and data processing)

    Returns:
        List[dict]: List of observations for Gryffin.
        Dict[str, set]: Dictionary of fragments used in the observations, sorted by fragment type (key).
    """
    in_progress, completed = handler.load_previous_results()
    observations, used_fragments = handler.process_previous_results(
        previous_results=completed,
        get_target_property=get_gain_cross_section
    )

    return observations, used_fragments


def generate_fragment_space(handler: LaserDataHandler, used_fragments: Dict[str, set]) -> Dict[str, Dict[str, list]]:
    """
    Processes the fragment space (available + previously used fragments) to obtain a dictionary of fragements and
    the corresponding descriptors for static Gryffin.

    Args:
        handler: LaserDataHandler object (database access and data processing)
        used_fragments: Dictionary fragments used in the observations, sorted by fragment type (key).

    Returns:
        Dict[str, Dict[str, list]]: Dictionary of all fragments (sorted by fragment type) and descriptors.
    """
    # Define Fragment Space of Available and Used Fragments
    available_fragments = handler.get_all_available_fragments()
    all_fragments = {frag: available_fragments[frag] | used_fragments[frag] for frag in available_fragments}

    print("Available Fragments:", ", ".join([f"{frag} ({len(available_fragments[frag])})" for frag in available_fragments]))
    print("Total Fragments:    ", ", ".join([f"{frag} ({len(all_fragments[frag])})" for frag in all_fragments]))

    # Load Descriptors for All Fragments
    all_fragments_with_descriptors = {frag_type: dict() for frag_type in all_fragments}

    for frag_type in all_fragments:
        for frag in all_fragments[frag_type]:
            descriptors = list(data_handler.get_molecule(frag).at[0, "descriptors"].values())

            if not descriptors:
                raise ValueError(f"No descriptors loaded for fragment {frag}!!!")

            all_fragments_with_descriptors[frag_type][frag] = descriptors

    print(f"Descriptors Successfully Loaded for all {sum([len(all_fragments_with_descriptors[frag_type]) for frag_type in all_fragments_with_descriptors])} Fragments")

    return all_fragments_with_descriptors


def run_gryffin(config: dict, descriptors: Dict[str, Dict[str, list]], observations: [List[dict]], handler: LaserDataHandler) -> List[dict]:
    """
    Instantiates Gryffin and runs the recommend method to generate a new batch of experimental recommendations
    according to the settings passed in the configuration.

    Args:
        config: Specified Gryffin configuration
        descriptors: Dictionary of all fragments (sorted by fragment type) and descriptors.
        observations: List of observations for Gryffin.

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
        known_constraints=lambda x: (handler.target_is_makable(x) and handler.target_is_novel(x) and x["procedure"] == "gen_2")
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


def process_recommendations(recommendations: List[dict], handler: LaserDataHandler, labs: list) -> None:
    """
    Processes the recommendations by drawing a grid image, writing a txt file with all recommendations, and saving the
    recommendations to the database.

    Args:
        recommendations: List of new recommendations by Gryffin.
        handler: LaserDataHandler object (database access and data processing)
        labs: List of labs for which the recommendations are generated.
    """
    iteration_name: str = datetime.date.today().strftime("%Y%m%d")

    with open(f"Recommendations/Iteration_{iteration_name}_{'_'.join(labs)}_all_samples.txt", "w") as file:
        for entry in recommendations:

            # Generate the Product SMILES and HID
            frag_smiles = [handler.get_molecule(entry[frag]).at[0, "smiles"] for frag in entry if "fragment" in frag]
            entry["smiles"] = run_two_step_suzuki(*frag_smiles)
            entry["hid"] = "".join([entry[frag] for frag in entry if "fragment" in frag])

            # Write to File and Upload to DB
            file.write(f"{entry['hid']},{entry['smiles']}\n")
            handler.create_target_compound(
                fragments=[entry[frag] for frag in entry if "fragment" in frag],
                smiles=entry["smiles"],
                procedure=entry["procedure"]
            )

    # Save Molecules as Grid Image
    img = Draw.MolsToGridImage(
        [Chem.MolFromSmiles(rec["smiles"]) for rec in recommendations],
        molsPerRow=5,
        subImgSize=(800, 800),
        legends=[rec["hid"] for rec in recommendations],
        returnPNG=False
    )
    img.save(f"Recommendations/Iteration_{iteration_name}_{'_'.join(labs)}_all_samples.png")


if __name__ == "__main__":
    config: dict = load_json(Path(__file__).parent / "gryffin_settings.json")

    data_handler = LaserDataHandler(
        db_name="madness_laser",
        fragments=("fragment_a", "fragment_b", "fragment_c"),
        active_labs=config["active_labs"]
    )

    observations, used_fragments = process_previous_observations(data_handler)
    fragments_descriptors = generate_fragment_space(data_handler, used_fragments)

    fragments_descriptors["procedure"] = {
        "gen_0": None,
        "gen_1": None,
        "gen_2": None
    }

    recommendations = run_gryffin(config, fragments_descriptors, observations, data_handler)
    process_recommendations(recommendations, data_handler, config["active_labs"])


