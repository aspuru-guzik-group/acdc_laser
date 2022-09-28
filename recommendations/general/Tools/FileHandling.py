from typing import Any
from pathlib import Path
import json
import pickle


def load_json(json_file: Path) -> Any:
    """
    Loads a json file and returns the saved Python object.
    Args:
        json_file: Path to the .json file.

    Returns:
        Any: Loaded object
    """
    with open(json_file, "r") as jsonfile:
        data: Any = json.load(jsonfile)

    return data


def load_pkl(pkl_file: Path) -> Any:
    """
    Loads a pickle file and returns the saved Python object.
    Args:
        pkl_file: Path to the .pkl file.

    Returns:
        Any: Loaded object
    """
    with open(pkl_file, "rb") as pklfile:
        data: Any = pickle.load(pklfile)

    return data


def save_pkl(data: Any, pkl_file: Path) -> None:
    """
    Saves a python object to a .pkl file.
    Args:
        data: Python object to be saved
        pkl_file: Path to the .pkl file.
    """
    with open(pkl_file, "wb") as pklfile:
        pickle.dump(data, pklfile)

    return data