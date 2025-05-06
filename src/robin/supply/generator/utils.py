"""Utils for the supply generator module."""

import yaml

from robin.supply.entities import Station, Service
from robin.supply.generator.exceptions import ServiceInMultiplePathsException

from datetime import timedelta
from functools import cache
from geopy.distance import geodesic
from typing import Any, Callable, List, Mapping, NamedTuple, Set, Tuple


class Segment(NamedTuple):
    service_idx: str
    start_pos: float
    end_pos: float
    time_at: 'Callable[[float], timedelta]'


def build_segments_for_service(
    service: Service,
    positions: Mapping[Station, float]
) -> List[Segment]:
    """
    Build a list of motion segments for a service, each with its spatial interval
    and a local linear time interpolator.
    """
    segments: List[Segment] = []
    stations = service.line.stations

    for prev_stn, next_stn in zip(stations, stations[1:]):
        if prev_stn not in positions or next_stn not in positions:
            raise ServiceInMultiplePathsException
        start_pos = positions[prev_stn]
        end_pos = positions[next_stn]

        # Scheduled departure and arrival
        depart_time = service.schedule[prev_stn.id][1] + timedelta(days=service.date.toordinal())
        arrive_time = service.schedule[next_stn.id][0] + timedelta(days=service.date.toordinal())

        # Local interpolator mapping any position in [start_pos, end_pos]
        time_interp = get_time_from_position(
            (depart_time, start_pos),
            (arrive_time, end_pos)
        )
        segments.append(Segment(service.id, start_pos, end_pos, time_interp))

    return segments


def get_edges_from_path(path: List[Station]) -> Set[Tuple[Station]]:
    """
    Returns the set of edges for a given path of stations.

    Args:
        path (List[Station]): A list of Station objects representing the path.

    Returns:
        Set[Tuple[Station]]: A set of edges, where each edge is represented as a tuple of two stations.
    """
    edges: Set[Tuple[Station]] = set()
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges.add((u, v))
    return edges


def get_stations_positions(stations: List[Station]) -> Mapping[Station, float]:
    """
    Compute the positions of stations along the line based on their coordinates.

    Args:
        stations: List of Station objects.

    Returns:
        A dictionary mapping each station to its position along the line.
    """
    positions = {}
    if not stations:
        return positions

    # First station is at position zero
    positions[stations[0]] = 0.0
    total_distance = 0.0

    # Iterate over consecutive station pairs
    for prev, curr in zip(stations, stations[1:]):
        segment = geodesic(prev.coordinates, curr.coordinates).km
        total_distance += segment
        positions[curr] = total_distance

    return positions


@cache
def get_time_from_position(
    point_a: Tuple[timedelta, float],
    point_b: Tuple[timedelta, float]
) -> Callable[[float], timedelta]:
    """
    Build a linear interpolator that maps a position (float) back to a time.

    Args:
        point_a: (time, position) for the first sample.
        point_b: (time, position) for the second sample.

    Returns:
        A function f(pos: float) -> timedelta giving the interpolated time.
    """
    # Convert times to minutes (float)
    t0 = point_a[0].total_seconds() / 60
    t1 = point_b[0].total_seconds() / 60

    # Extract positions
    x0 = point_a[1]
    x1 = point_b[1]

    # Ensure we have a valid line
    if t0 == t1:
        raise ValueError('point_a and point_b must have different times')
    if x0 == x1:
        raise ValueError('point_a and point_b must have different positions')

    # Slope: position change per minute
    slope = (x1 - x0) / (t1 - t0)

    def time_from_position(position: float) -> timedelta:
        """
        Given a position, compute the corresponding time via
        inverse of y = slope * t + intercept.
        """
        # Invert the line: t = (position - intercept) / slope
        minutes = (position - x0) / slope + t0
        return timedelta(minutes=minutes)

    return time_from_position


def infer_paths(service: Service) -> List[List[Station]]:
    """
    Infers the path of a service based on its line and corridor.

    Args:
        service (Service): The service to infer.

    Returns:
        List[List[Station]]: A list of paths, where each path is a list of Station objects.
    """
    paths = []
    for path in service.line.corridor.paths:
        # Skip paths that do not contain the service's stations
        if sum([station in path for station in service.line.stations]) < 2:
            continue
        # Only consider forward paths
        if not (service.line.stations[0] in path and service.line.stations[-1] in path):
            raise ServiceInMultiplePathsException
        if path.index(service.line.stations[0]) < path.index(service.line.stations[-1]):
            origin_index = path.index(service.line.stations[0])
            destination_index = path.index(service.line.stations[-1])
            paths.append(path[origin_index:destination_index + 1])
    return paths


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


def segments_conflict(
    seg1: Segment,
    seg2: Segment,
    safety_headway: int
) -> bool:
    """
    Determine if two motion segments conflict within a given safety headway in minutes.

    They conflict if their spatial intervals overlap and their time gaps
    at the overlap boundaries violate the headway constraint.
    """
    # Spatial overlap
    overlap_start = max(seg1.start_pos, seg2.start_pos)
    overlap_end = min(seg1.end_pos, seg2.end_pos)
    if overlap_start >= overlap_end:
        return False

    # Time at overlap boundaries
    t1_start = seg1.time_at(overlap_start)
    t1_end = seg1.time_at(overlap_end)
    t2_start = seg2.time_at(overlap_start)
    t2_end = seg2.time_at(overlap_end)

    # Time differences in whole minutes
    dt_start = int((t2_start - t1_start).total_seconds() / 60)
    dt_end = int((t2_end - t1_end).total_seconds() / 60)

    # No conflict if both differences have the same sign (ordering preserved)
    # and both exceed twice the safety headway
    same_order = dt_start * dt_end > 0
    if same_order and abs(dt_start) >= 2 * safety_headway and abs(dt_end) >= 2 * safety_headway:
        return False

    return True


def shared_edges_between_services(path1: List[Station], path2: List[Station]) -> Set[Tuple[Station]]:
    """
    Checks whether two services (given by their paths) share any track segments.

    Args:
        path1: A list of Station objects for the first service.
        path2: A list of Station objects for the second service.

    Returns:
        A set of shared edges (track segments), if any.
    """
    edges1 = get_edges_from_path(path1)
    edges2 = get_edges_from_path(path2)
    return edges1.intersection(edges2)
