"""Utils for the plotter module."""

import functools
import networkx as nx

from robin.supply.generator.entities import ServiceScheduler
from robin.supply.entities import Station, Service

from loguru import logger
from typing import Any, Callable, List, Optional, Set, Tuple


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
        origin, destination = path[i], path[i + 1]
        edges.add((origin, destination))
    return edges


def infer_paths(service: Service) -> List[List[Station]]:
    """
    Infers the path of a service based on its line and corridor.

    Args:
        service (Service): The service to infer.

    Returns:
        List[List[Station]]: A list of paths, where each path is a list of Station objects.
    """
    graph = ServiceScheduler.build_graph(service.line.corridor.tree)
    stops = service.line.stations

    if len(stops) < 2:
        return []

    # Get the full path from the first to the last station
    full_path: List[Any] = []
    for origen, destino in zip(stops, stops[1:]):
        seg = nx.shortest_path(graph, origen, destino, weight='weight')
        if full_path:
            full_path.extend(seg[1:])
        else:
            full_path.extend(seg)

    # Detect split points in the path
    split_idxs = sorted({
        idx
        for idx, node in enumerate(full_path)
        if graph.degree[node] != 2
    })

    # Check if the first and last stations are split points
    if 0 not in split_idxs:
        split_idxs.insert(0, 0)
    last = len(full_path) - 1
    if last not in split_idxs:
        split_idxs.append(last)

    # Split the full path into subroutes
    subroutes: List[List[Any]] = []
    for a, b in zip(split_idxs[:-1], split_idxs[1:]):
        subroutes.append(full_path[a: b + 1])

    return subroutes


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


def requires_attribute(attr_name: str, error_msg: str) -> Callable:
    """
    Decorator factory to ensure that a method requires a specific attribute.

    Args:
        attr_name (str): The attribute name to check on self.
        error_msg (str): The error message to log if attribute is missing.
    
    Returns:
        Callable: A decorator that checks for the attribute before executing the method.
    """
    def decorator(method) -> Callable:
        """
        Decorator to check if the specified attribute exists on self.

        Args:
            method (Callable): The method to be decorated.
        
        Returns:
            Callable: The wrapped method that checks for the attribute.
        """
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs) -> Optional[Callable]:
            """
            Wrapper function to check if the specified attribute exists.

            Args:
                self: Instance of the class.
                *args: Positional arguments for the method.
                **kwargs: Keyword arguments for the method.
            
            Returns:
                Optional[Callable]: The result of the method if the attribute exists, otherwise logs an error.
            """
            if getattr(self, attr_name, None) is None:
                logger.error(f"Method '{method.__name__}' {error_msg}")
                return
            return method(self, *args, **kwargs)
        return wrapper
    return decorator


requires_config_supply = requires_attribute(
    'supply',
    "requires supply configuration. Please provide 'path_config_supply'."
)


requires_output_csv = requires_attribute(
    'output',
    "requires output CSV data. Please provide 'path_output_csv'."
)
