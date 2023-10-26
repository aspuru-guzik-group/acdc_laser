import copy
import json
from pathlib import Path
from typing import Union, Any, get_args
import pandas as pd
import numpy as np


SERIALIZABLE_SINGLE: type = Union[
    bool,
    int,
    float,
    str,
    None,
    np.float32,
    np.float64,
    np.int32,
    np.int64
]

SERIALIZABLE_COLLECTION: type = Union[
    list,
    dict
]


def load_json(file: Union[str, Path]) -> Any:
    with open(file, "r") as f:
        data = json.load(f)
    return data


def save_json(data: Union[list, dict], file: Union[str, Path]) -> None:
    data = copy.deepcopy(data)
    data = check_json_content(data)
    with open(file, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(data: np.ndarray, colnames: list, file_path: Path) -> None:
    df = pd.DataFrame(data, columns=colnames)
    df.to_csv(file_path, index=False)


def check_json_content(data: SERIALIZABLE_COLLECTION) -> SERIALIZABLE_COLLECTION:

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, get_args(SERIALIZABLE_COLLECTION)):
                data[idx] = check_json_content(entry)
            elif isinstance(entry, get_args(SERIALIZABLE_SINGLE)):
                continue
            else:
                data[idx] = str(data)

    elif isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, get_args(SERIALIZABLE_COLLECTION)):
                data[key] = check_json_content(value)
            elif isinstance(value, get_args(SERIALIZABLE_SINGLE)):
                continue
            else:
                data[key] = str(value)

    return data




