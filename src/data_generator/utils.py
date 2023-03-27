"""Utils for data generator."""

import datetime
from math import sin, cos, acos, radians
from src.robin.supply.entities import Station, Line, TimeSlot, TSP, RollingStock, Service
from typing import List, Dict


def _get_start_time(s):
    """
    Get start time of a service by adding the date and time slot

    Args:
        s (Service): Service object

    Returns:
        datetime.datetime: Start time
    """
    return datetime.datetime.combine(s.date, datetime.datetime.min.time()) + s.time_slot.start


def _get_end_time(s):
    """
    Get end time of a service by adding the date, time slot and duration. Duration is retrieved from the last
    schedule entry.

    Args:
        s (Service): Service object

    Returns:
        datetime.datetime: End time
    """
    end_td = datetime.timedelta(seconds=int(list(s.line.timetable.values())[-1][-1] * 60))
    return datetime.datetime.combine(s.date, datetime.datetime.min.time()) + s.time_slot.start + end_td


def _to_station(tree: Dict, sta_dict: Dict[str, Station]) -> Dict:
    """
    Recursive function to build a tree of Station objects from a tree of station IDs
    Args:
        tree (Dict): Tree of station IDs
        sta_dict (Dict[str, Station]): Dict of Station objects {station_id: Station object}
    Returns:
        Dict: Tree of Station objects
    """
    if not tree:
        return {}

    return {sta_dict[node]: _to_station(tree[node], sta_dict) for node in tree}


def _tree_to_yaml(dict_tree: Dict[Station, Dict]) -> List[Dict]:
    """
    Recursive function to convert a Dict[Station , Dict] tree of Station objects to a tree for yaml file
    List[Dict[str: Station, str: Dict]

    Args:
        tree as nested dictionaries (Dict[Station, Dict]): Tree of Station objects
    Returns:
        tree parsed for yaml file (List[Dict])

    """
    if len(dict_tree) == 0:
        return []
    else:
        return [{'org': node, 'des': _tree_to_yaml(dict_tree[node])} for node in dict_tree]


def _build_service(date: datetime.date,
                   line: Line,
                   time_slot: TimeSlot,
                   tsp: TSP,
                   rs: RollingStock,
                   prices: Dict
                   ) -> Service:
    """
    Build service object from parameters

    Args:
        date (datetime.date): Date of the service
        line (Line): Line of the service
        time_slot (TimeSlot): Time slot of the service
        tsp (TSP): TSP of the service
        rs (RollingStock): Rolling stock of the service
        prices (Mapping[]): Prices of the service

    Returns:
        Service: Service object
    """
    return Service(id_=f'{line.id}_{time_slot.id}',
                   date=str(date),
                   line=line,
                   time_slot=time_slot,
                   tsp=tsp,
                   rolling_stock=rs,
                   prices=prices)


def _get_distance(line: Line, origin: Station, destination: Station) -> float:
    """
    Get distance between two stations.

    Args:
        line (Line): Line object to get the distance between stations
        origin (Station): Origin station object
        destination (Station): Destination station object

    Returns:
        float: Distance in km
    """
    earth_radius = 6371.0
    assert origin in line.stations and destination in line.stations, 'Stations not in line'

    lat1, lon1, lat2, lon2 = map(radians, [*origin.coords, *destination.coords])
    lon_diff = lon2 - lon1
    return acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon_diff)) * earth_radius
