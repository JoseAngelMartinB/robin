"""Utils for the data loader module."""

from copy import deepcopy
from src.robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from typing import Mapping, List
import yaml
import os

def station_to_dict(obj: Station):
    return {'id': obj.id,
            'name': obj.name,
            'city': obj.city,
            'short_name': obj.shortname,
            'coordinates': {'latitude': float(obj.coords[0]), 'longitude': float(obj.coords[1])}}


def time_slot_to_dict(obj: TimeSlot):
    return {'id': obj.id,
            'start': str(obj.start),
            'end': str(obj.end)}


def corridor_to_dict(obj: Corridor):
    def yaml_tree(d):
        if not d:
            return []
        else:
            node = [{'org': k.id, 'des': yaml_tree(v)} for k, v in d.items()]
            return node

    tree_ids = yaml_tree(deepcopy(obj.tree))

    return {'id': obj.id,
            'name': obj.name,
            'stations': tree_ids}


def line_to_dict(obj: Line):
    stops = []
    for s in obj.timetable:
        arr, dep = obj.timetable[s]
        stops.append({'station': s, 'arrival_time': arr, 'departure_time': dep})

    return {'id': obj.id,
            'name': obj.name,
            'corridor': obj.corridor.id,
            'stops': stops}


def seat_to_dict(obj: Seat):
    return {'id': obj.id,
            'name': obj.name,
            'hard_type': obj.hard_type,
            'soft_type': obj.soft_type}


def rolling_stock_to_dict(obj: RollingStock):
    return {'id': obj.id,
            'name': obj.name,
            'seats': [{'hard_type': s, 'quantity': obj.seats[s]} for s in obj.seats]}


def tsp_to_dict(obj: TSP):
    return {'id': obj.id,
            'name': obj.name,
            'rolling_stock': [rs.id for rs in obj.rolling_stock]}


def service_to_dict(obj: Service):
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