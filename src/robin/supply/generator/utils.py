"""Utils for the supply generator module."""

import yaml

from typing import Any, Dict


def read_yaml(path: str) -> Dict[str, Any]:
    """
    Read a YAML file and return its content.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Content of the YAML file.
    """
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.CSafeLoader)
    return data
