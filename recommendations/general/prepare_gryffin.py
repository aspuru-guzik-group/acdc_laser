from typing import Tuple, List, Dict
from pathlib import Path
from Tools.LaserDataHandler import LaserDataHandler, get_gain_cross_section
from Tools.FileHandling import load_json, save_pkl


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


def save_output(handler: LaserDataHandler, observations: list, descriptors: dict) -> None:
    """
    Saves all processed information (observations, descriptors, ...) as pkl files.
    Args:
        handler: LaserDataHandler object (database access and data processing)
        observations: List of all processed observations for Gryffin.
        descriptors: Dictionary of all categorical options and descriptors.
    """
    save_pkl(observations, Path.cwd() / "TMP_observations.pkl")
    save_pkl(descriptors, Path.cwd() / "TMP_descriptors.pkl")
    handler.save_status()


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

    save_output(data_handler, observations, fragments_descriptors)


