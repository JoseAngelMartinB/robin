"""Utils for the data loader module."""

import datetime
import os
import yaml

from src.robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service

from copy import deepcopy
from typing import Dict, List, Mapping


def station_to_dict(obj: Station) -> Dict:
    """
    Convert a station object to a dictionary.

    Args:
        obj: The station object to convert.

    Returns:
        Dict: A dictionary representation of the station object.
    """
    return {'id': obj.id,
            'name': obj.name,
            'city': obj.city,
            'short_name': obj.shortname,
            'coordinates': {'latitude': float(obj.coords[0]), 'longitude': float(obj.coords[1])}}


def time_slot_to_dict(obj: TimeSlot) -> Dict:
    """
    Convert a time slot object to a dictionary.

    Args:
        obj: The time slot object to convert.

    Returns:
        Dict: A dictionary representation of the time slot object.
    """
    return {'id': obj.id,
            'start': str(obj.start),
            'end': str(obj.end)}


def corridor_to_dict(obj: Corridor) -> Dict:
    """
    Convert a corridor object to a dictionary.

    Args:
        obj: The corridor object to convert.

    Returns:
        Dict: A dictionary representation of the corridor object.
    """
    def yaml_tree(dictionary: Dict) -> List:
        """
        Convert a dictionary tree to a list of dictionaries (supply yaml format).

        Args:
            dictionary: The dictionary to convert.

        Returns:
            List: A list of dictionaries.
        """
        if not dictionary:  # Empty dictionary
            return []
        else:
            node = [{'org': k.id, 'des': yaml_tree(v)} for k, v in dictionary.items()]
            return node

    tree_ids = yaml_tree(deepcopy(obj.tree))

    return {'id': obj.id,
            'name': obj.name,
            'stations': tree_ids}


def line_to_dict(obj: Line) -> Dict:
    """
    Convert a line object to a dictionary.

    Args:
        obj: The line object to convert.

    Returns:
        Dict: A dictionary representation of the line object.
    """
    stops = []
    for s in obj.timetable:
        arr, dep = obj.timetable[s]
        stops.append({'station': s, 'arrival_time': arr, 'departure_time': dep})

    return {'id': obj.id,
            'name': obj.name,
            'corridor': obj.corridor.id,
            'stops': stops}


def seat_to_dict(obj: Seat) -> Dict:
    """
    Convert a seat object to a dictionary.

    Args:
        obj: The seat object to convert.

    Returns:
        Dict: A dictionary representation of the seat object.
    """
    return {'id': obj.id,
            'name': obj.name,
            'hard_type': obj.hard_type,
            'soft_type': obj.soft_type}


def rolling_stock_to_dict(obj: RollingStock) -> Dict:
    """
    Convert a rolling stock object to a dictionary.

    Args:
        obj: The rolling stock object to convert.

    Returns:
        Dict: A dictionary representation of the rolling stock object.
    """
    return {'id': obj.id,
            'name': obj.name,
            'seats': [{'hard_type': s, 'quantity': obj.seats[s]} for s in obj.seats]}


def tsp_to_dict(obj: TSP) -> Dict:
    """
    Convert a train service provider object to a dictionary.

    Args:
        obj: The train service provider object to convert.

    Returns:
        Dict: A dictionary representation of the train service provider object.
    """
    return {'id': obj.id,
            'name': obj.name,
            'rolling_stock': [rs.id for rs in obj.rolling_stock]}


def service_to_dict(obj: Service) -> Dict:
    """
    Convert a service object to a dictionary.

    Args:
        obj: The service object to convert.

    Returns:
        Dict: A dictionary representation of the service object.
    """
    prices = []
    for k, v in obj.prices.items():
        prices.append({'origin': k[0],
                       'destination': k[1],
                       'seats': [{'seat': str(ks.id), 'price': str(float(ps))} for ks, ps in v.items()]})

    return {'id': str(obj.id),
            'date': str(obj.date),
            'line': str(obj.line.id),
            'train_service_provider': str(obj.tsp.id),
            'time_slot': str(obj.time_slot.id),
            'rolling_stock': str(obj.rolling_stock.id),
            'origin_destination_tuples': prices,
            'capacity_constraints': obj.capacity_constraints}


def write_to_yaml(filename: str, objects: Mapping[str, List]) -> None:
    """
    Write the given objects to the given YAML file.

    Args:
        filename (str): The name of the YAML file.
        objects (list): The objects to write to the YAML file.
        key (str): The key to use for the objects.
    """
    if not os.path.isfile(filename):
        with open(filename, 'w') as yaml_file:
            yaml.safe_dump(objects, yaml_file, sort_keys=False, allow_unicode=True)
            return

    with open(filename, 'r') as yaml_file:
        yaml_file_mod = yaml.safe_load(yaml_file)
        try:
            yaml_file_mod.update(objects)
        except AttributeError:
            yaml_file_mod = objects

    if yaml_file_mod:
        with open(filename, 'w') as yaml_file:
            yaml.safe_dump(yaml_file_mod, yaml_file, sort_keys=False, allow_unicode=True)


def time_delta_to_time_string(time_delta: datetime.timedelta) -> str:
    """
    Convert time delta to string HH.MM

    Args:
        time_delta: datetime.timedelta object

    Returns:
        string with time delta in format HH.MM
    """
    hours = time_delta.total_seconds() // 3600
    minutes = (time_delta.total_seconds() % 3600) / 60
    return f"{int(hours):02d}.{int(minutes):02d}"