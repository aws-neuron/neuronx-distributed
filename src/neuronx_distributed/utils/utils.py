from enum import Enum
import os
import json
from typing import Union, Dict, Any


class hardware(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"


def get_dict_from_json(json_file: Union[str, os.PathLike]) -> Dict[Any, Any]:
    """Reads a JSON file and returns its contents as a Python dictionary.

    Args:
        json_file: Path to the JSON file

    Returns:
        Dict containing the parsed JSON content

    Raises:
        ValueError: If the file cannot be parsed as valid JSON
    """
    try:
        with open(json_file, "r", encoding="utf-8") as opened_file:
            return json.loads(opened_file.read())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to parse JSON file '{json_file}': {str(e)}")
    except FileNotFoundError as e:
        raise ValueError(f"JSON file not found at '{json_file}': {str(e)}")
