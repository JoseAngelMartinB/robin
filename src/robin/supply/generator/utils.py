"""Utils for the supply generator module."""

import networkx as nx
import yaml

from robin.supply.entities import Station, Service

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from functools import cache
from geopy.distance import geodesic
from typing import Any, Callable, List, Mapping, Tuple


@dataclass
class Segment:
    service_idx: str
    path: List[Station]
    time_at: Callable[[Station], datetime]
    edges: List[Tuple[Station, Station]] = field(init=False)

    def __post_init__(self):
        # Precompute the edges based on the path
        self.edges = [
            (self.path[i], self.path[i+1])
            for i in range(len(self.path) - 1)
        ]


def build_graph(tree: Mapping[Station, ...], graph=None) -> nx.Graph:
    """
    Recursively builds a graph from the given tree structure.

    Args:
        tree (Mapping): A mapping of Station objects to their branches.
        graph (nx.Graph, optional): An existing graph to add edges to. If None, a new graph is created.

    Returns:
        nx.Graph: The graph with edges added based on the tree structure.
    """
    if graph is None:
        graph = nx.Graph()

    for origin_station, branches in tree.items():
        for destination_station in branches:
            # Get geodesic distance between the two stations
            coord_origen = origin_station.coordinates
            coord_destino = destination_station.coordinates
            distance_km = geodesic(coord_origen, coord_destino).kilometers

            # Add edge to the graph with the distance as weight
            graph.add_edge(origin_station, destination_station, weight=distance_km)

            # Recursively build the graph for the branches
            build_graph({destination_station: branches[destination_station]}, graph)

    return graph


def build_segments_for_service(
    graph: nx.Graph,
    service: Service,
) -> List[Segment]:
    """
    Build a list of motion segments for a service, each with its spatial interval
    and a local linear time interpolator.

    Args:
        graph (nx.Graph): The graph representing the corridor.
        service (Service): The service for which to build segments.

    Returns:
        List[Segment]: A list of segments, each representing a motion segment for the service.
    """
    segments: List[Segment] = []
    stops = service.line.stations  # EN ORDEN DE VIAJE

    for prev_stn, next_stn in zip(stops, stops[1:]):
        # Real path between prev_stn and next_stn
        subpath = nx.shortest_path(graph, prev_stn, next_stn, weight='weight')

        # Cumulate distances between stations
        dists = [0.0]
        for u, v in zip(subpath, subpath[1:]):
            dists.append(dists[-1] + geodesic(u.coordinates, v.coordinates).kilometers)

        total_km = dists[-1]
        midnight = time.min  # 00:00
        dep_delta = service.schedule[prev_stn.id][1]  # timedelta desde medianoche
        arr_delta = service.schedule[next_stn.id][0]

        dep_time = datetime.combine(service.date, midnight) + dep_delta
        arr_time = datetime.combine(service.date, midnight) + arr_delta

        # Interpolator function to compute time at each station
        def make_time_at(subpath, dists, dep_time, arr_time):
            def time_at(st: Station) -> datetime:
                # Get the index of the station in the subpath
                idx = subpath.index(st)
                frac = dists[idx] / (total_km or 1)
                delta = arr_time - dep_time
                return dep_time + frac * delta

            return time_at

        time_at_fn = make_time_at(subpath, dists, dep_time, arr_time)

        segments.append(
            Segment(
                service_idx=service.id,
                path=subpath,
                time_at=time_at_fn
            )
        )

    return segments


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
    segment_1: Segment,
    segment_2: Segment,
    safety_headway: int
) -> bool:
    """
    Determine if two motion segments conflict within a given safety headway in minutes.

    A conflict appears if their spatial intervals overlap and their time gaps at the overlap boundaries violate
        the headway constraint.

    Args:
        segment_1 (Segment): The first motion segment.
        segment_2 (Segment): The second motion segment.
        safety_headway (int): Safety headway in minutes.

    Returns:
        bool: True if the segments conflict, False otherwise.
    """
    common_edges = set(segment_1.edges) & set(segment_2.edges)
    if not common_edges:
        return False

    for origin_station, destination_station in common_edges:
        # Get time intervals for the common edges in both segments
        start_segment_1 = segment_1.time_at(origin_station)
        end_segment_1 = segment_1.time_at(destination_station)
        start_segment_2 = segment_2.time_at(origin_station)
        end_segment_2 = segment_2.time_at(destination_station)

        delta_start = (start_segment_2 - start_segment_1).total_seconds() / 60
        delta_end = (end_segment_2 - end_segment_1).total_seconds() / 60

        # Conflict if differences exceed twice the safety headway
        same_order = delta_start * delta_end > 0
        if not same_order or abs(delta_start) < 2 * safety_headway or abs(delta_end) < 2 * safety_headway:
            return True

    return False
