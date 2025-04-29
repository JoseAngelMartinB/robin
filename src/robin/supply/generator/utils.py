"""Utils for the supply generator module."""

from robin.supply.entities import Station, Line

from math import sin, cos, acos, radians
from typing import Any, Mapping

import yaml


def get_distance(line: Line, origin: Station, destination: Station) -> float:
    """
    Get distance between two stations in a line in km using Haversine formula.

    Args:
        line (Line): Line to get the distance between the stations.
        origin (Station): Origin station.
        destination (Station): Destination station.

    Returns:
        float: Distance between the two stations in km.
    """
    # NOTE: Considered geodesic distance for a more accurate result and cache it
    earth_radius = 6371.0
    assert origin in line.stations and destination in line.stations, 'Stations not in line'

    lat1, lon1, lat2, lon2 = map(radians, [*origin.coordinates, *destination.coordinates])
    lon_diff = lon2 - lon1
    return acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon_diff)) * earth_radius


def read_yaml(path: str) -> Mapping[str, Any]:
    """
    Read a YAML file and return its content.

    Args:
        path (str): Path to the YAML file.

    Returns:
        Mapping[str, Any]: Content of the YAML file.
    """
    with open(path, 'r') as file:
        data = yaml.load(file, Loader=yaml.CSafeLoader)
    return data
