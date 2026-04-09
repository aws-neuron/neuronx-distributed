from enum import auto, Enum
import os
import json
from typing import Union, Dict, Any

class hardware(Enum):
    TRN1 = "trn1"
    TRN2 = "trn2"
    TRN3 = "trn3"
    CUSTOM = auto()

    @classmethod
    def _missing_(cls, value):
        _value_map = {
            "trn1": cls.TRN1,
            "trn1n": cls.TRN1,
            "inf2": cls.TRN1,
            "trn2": cls.TRN2,
            "trn3": cls.TRN3,
        }
        if value in _value_map:
            return _value_map[value]
        elif value == os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
            return cls.CUSTOM

        return None


class HloMetadataLevel(Enum):
    """Enumeration for HLO metadata generation modes.

    Defines three levels of metadata output: DETAILED for comprehensive information,
    MINIMAL for essential variables only, and DISABLED to skip metadata generation entirely.
    """
    DEBUG = "debug"
    INFO = "info"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        # This check is for backward compatibility with boolean debug flag
        # which will be deprecated in next version.
        if value is True:
            return cls.DEBUG
        elif value is False:
            return cls.INFO

        return None


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
