import pandas as pd


def target_is_makable(target: dict, available_fragments: pd.DataFrame, active_labs: list) -> bool:
    """
    Checks if a target can be made in a single location (i.e. all fragments are available at one spot).

    Args:
        target: Dictionary of parameters (needs the fragments as keys)
        available_fragments: Dataframe of all available fragments.
        active_labs: List of all active labs to check

    Returns:
        bool: True if the target can be made.
    """
    for lab in active_labs:
        if all([target[frag] in available_fragments[lab][frag] for frag in target if "fragment" in frag]):
            return True
    return False


def target_is_novel(target: dict, all_previous_recommendations: pd.DataFrame) -> bool:
    """
    Checks if a target is novel (i.e. has never been made or recommended before).

    Args:
        target: Dictionary of parameters (needs the fragments as keys)
        all_previous_recommendations: Dataframe of all recommendations ever made to the database.

    Returns:
        bool: True if the target is novel
    """
    # all_previous_recommendations = all_previous_recommendations[all_previous_recommendations["synthesis.status"] != "DONE"]
    hid = "".join([target[frag] for frag in target if "fragment" in frag])
    return hid not in all_previous_recommendations["product.hid"].values


