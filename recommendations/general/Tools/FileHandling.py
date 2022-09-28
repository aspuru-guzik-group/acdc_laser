from typing import Any
from pathlib import Path
import json


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
