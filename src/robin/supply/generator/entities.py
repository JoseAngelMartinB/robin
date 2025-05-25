"""Entities for the supply generator module."""

import datetime
import hashlib
import json
import networkx as nx
import numpy as np
import os
import random

from robin.demand.utils import get_scipy_distribution

from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from robin.supply.saver.entities import SupplySaver

from robin.supply.generator.exceptions import ServiceWithConflicts, UnfeasibleServiceException
from robin.supply.generator.constants import MAX_RETRY, N_SERVICES, SAFETY_GAP, TIME_SLOT_SIZE
from robin.supply.generator.utils import read_yaml

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from geopy.distance import geodesic
from loguru import logger
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Mapping, Set, Union, Tuple


@dataclass
class Segment:
    """
    A segment represents a part of a service's route, including the stations path and timing information.

    Attributes:
        service_id (str): ID of the service this segment belongs to.
        path (List[Station]): List of stations in the segment.
        time_at (Callable[[Station], datetime.datetime]): Function to compute the time at a given station.
        edges (List[Tuple[Station, Station]]): List of edges in the segment.
    """
    service_id: str
    path: List[Station]
    time_at: Callable[[Station], datetime.datetime]
    edges: List[Tuple[Station, Station]] = field(init=False)

    def __post_init__(self):
        """
        Post-initialization to compute edges based on the path.
        """
        self.edges = [(self.path[i], self.path[i + 1]) for i in range(len(self.path) - 1)]


class ServiceScheduler:
    """
    Schedule services on shared corridors, ensuring no conflicts given a safety headway.

    Attributes:
        services (List[Service]): List of already scheduled services.
        corridor_graphs (Dict[str, nx.Graph]): Graphs representing corridors for each service.
        segments_cache (Dict[str, List[Segment]]): Cache for segments of each service.
        edge_index (Dict[Tuple[Station, Station], List[Segment]]): Index of segments by edges.
        without_conflicts (bool): Whether to check for conflicts when adding new services.
    """

    def __init__(self, services: List[Service], without_conflicts: bool = True) -> None:
        """
        Initialize with a list of pre-existing services, build internal indexes.

        Args:
            services (List[Service]): Already scheduled services.
            without_conflicts (bool, optional): Whether to check for conflicts when
                adding new services. Defaults to True.
        """
        self.services = services
        self.corridor_graphs = {}
        self.segments = {}
        self.edges_index = defaultdict(list)
        self.without_conflicts = without_conflicts

        for service in services:
            self._add_service_to_index(service)

    def _add_service_to_index(self, service: Service) -> None:
        """
        Index a service by its edges and build segments for it.

        Args:
            service (Service): Service to index and build segments for.
        """
        corridor_id = service.line.corridor.id
        if corridor_id not in self.corridor_graphs:
            self.corridor_graphs[corridor_id] = ServiceScheduler.build_graph(service.line.corridor.tree)
        graph = self.corridor_graphs[corridor_id]
        segments = self._build_segments_for_service(graph, service)
        self.segments[service.id] = segments
        for segment in segments:
            for edge in segment.edges:
                self.edges_index[edge].append(segment)

    def _build_segments_for_service(self, corridor_graph: nx.Graph, service: Service) -> List[Segment]:
        """
        Split a service's route into motion segments with per-edge timing.

        Args:
            corridor_graph (nx.Graph): Graph representing a corridor for shortest-path lookups.
            service (Service): Service to build segments for.

        Returns:
            List[Segment]: List of Segment representing the service's route.
        """
        segments: List[Segment] = []
        stops = service.line.stations

        for origin_station, destination_station in zip(stops, stops[1:]):
            # Get the detailed station‐by‐station path between stops
            subpath = nx.shortest_path(corridor_graph, origin_station, destination_station, weight='weight')

            # Build a cumulative distance list along subpath
            distances = [0.0]
            for origin_substation, destination_substation in zip(subpath, subpath[1:]):
                distance_between_stations = geodesic(origin_substation.coordinates, destination_substation.coordinates)
                distances.append(distances[-1] + distance_between_stations.kilometers)

            # Compute datetime at departure and arrival
            midnight = datetime.time.min
            departure_delta = service.schedule[origin_station.id][1]
            arrival_delta = service.schedule[destination_station.id][0]
            departure_time = datetime.datetime.combine(service.date, midnight) + departure_delta
            arrival_time = datetime.datetime.combine(service.date, midnight) + arrival_delta

            time_at = ServiceScheduler._make_time_interpolator(subpath, distances, departure_time, arrival_time)
            segments.append(Segment(service_id=service.id, path=subpath, time_at=time_at))
        return segments

    @staticmethod
    def _make_time_interpolator(
        path: List[Station],
        distances: List[float],
        departure: datetime,
        arrival: datetime
    ) -> Callable[[Station], datetime.datetime]:
        """
        Create a function mapping any station on path to its interpolated time.

        Args:
            path (List[Station]): Ordered stations along the subpath.
            distances (List[float]): Cumulative km at each index in path.
            departure (datetime.datetime): Exact departure datetime from first station.
            arrival (datetime.datetime): Exact arrival datetime at last station.

        Returns:
            Callable[[Station], datetime]: Function to compute time at a given station.
        """
        # Avoid division by zero if path is empty
        total = distances[-1] or 1.0
        duration = arrival - departure
        def time_at(station: Station) -> datetime.datetime:
            """
            Compute the time at a given station by linear interpolation.

            Args:
                station (Station): Station to interpolate.

            Returns:
                datetime.datetime: Interpolated time at the given station.
            """
            path_index = path.index(station)
            fraction = distances[path_index] / total
            return departure + fraction * duration
        return time_at

    @staticmethod
    def _segments_conflict(segment1: Segment, segment2: Segment, safety_headway: int) -> bool:
        """
        Check if two segments overlap in space and violate headway in time.

        Args:
            segment1 (Segment): First service segment.
            segment2 (Segment): Second service segment.
            safety_headway (int): Minimum minutes separation required.

        Returns:
            bool: True if there is a conflict, False otherwise.
        """
        common_edges = set(segment1.edges) & set(segment2.edges)
        if not common_edges:
            return False
        # For each overlapping edge, compare start and end times
        for (origin_station, destination_station) in common_edges:
            # Time when each enters and exits the edge
            t1_start, t1_end = segment1.time_at(origin_station), segment1.time_at(destination_station)
            t2_start, t2_end = segment2.time_at(origin_station), segment2.time_at(destination_station)
            diff_start = (t2_start - t1_start).total_seconds() / 60
            diff_end = (t2_end - t1_end).total_seconds() / 60
            # Conflict if different order or gaps below double headway
            double_headway = 2 * safety_headway
            if diff_start * diff_end <= 0 or abs(diff_start) <= double_headway or abs(diff_end) <= double_headway:
                return True
        return False

    def add_service(self, new_service: Service) -> None:
        """
        Add a new service to the scheduler, checking for conflicts.

        Args:
            new_service (Service): Service to schedule.

        Raises:
            ServiceWithConflicts: If the new service conflicts with existing ones.
        """
        if not self.is_feasible(new_service) and self.without_conflicts:
            raise ServiceWithConflicts
        self.services.append(new_service)
        self._add_service_to_index(new_service)

    @staticmethod
    def build_graph(tree: Mapping[Station, Mapping], corridor_graph: nx.Graph = None) -> nx.Graph:
        """
        Recursively convert a station-tree into a weighted NetworkX graph.

        Args:
            tree (Mapping[Station, Mapping): Corridor tree with stations as keys and branches as values.
            corridor_graph (nx.Graph, optional): Existing graph representing a corridor
                to add edges into. Defaults to None.

        Returns:
            nx.Graph: Undirected graph with weighted edges representing distances in km.
        """
        if corridor_graph is None:
            corridor_graph = nx.Graph()
        for origin, branches in tree.items():
            for destination in branches:
                km = geodesic(origin.coordinates, destination.coordinates).kilometers
                corridor_graph.add_edge(origin, destination, weight=km)
                ServiceScheduler.build_graph({destination: branches.get(destination, [])}, corridor_graph)
        return corridor_graph

    def find_conflicts(self, new_service: Service, safety_gap: int, early_stop: bool = False) -> Set[str]:
        """
        Find all services that conflict with a new service.

        Args:
            new_service (Service): Service to schedule.
            safety_gap (int): Required headway in minutes.
            early_stop (bool, optional): If True, stop at the first conflict found. Defaults to False.

        Returns:
            Set[str]: Set of service ids that conflict with the new service.
        """
        corridor_id = new_service.line.corridor.id
        if corridor_id not in self.corridor_graphs:
            self.corridor_graphs[corridor_id] = ServiceScheduler.build_graph(
                new_service.line.corridor.tree
            )
        graph = self.corridor_graphs[corridor_id]
        new_segments = self._build_segments_for_service(graph, new_service)

        conflicts: set[str] = set()
        for segment1 in new_segments:
            for edge in segment1.edges:
                for segment2 in self.edges_index.get(edge, []):
                    if self._segments_conflict(segment1, segment2, safety_gap):
                        if early_stop:
                            return {segment2.service_id}
                        conflicts.add(segment2.service_id)
        return conflicts

    def is_feasible(self, new_service: Service, safety_gap: int = SAFETY_GAP, early_stop: bool = True) -> bool:
        """
        Check if a new service can be scheduled without conflicts.

        Args:
            new_service (Service): Service to check for conflicts.
            safety_gap (int, optional): Required headway in minutes. Defaults to 10.
            early_stop (bool, optional): If True, stop at the first conflict found. Defaults to True.

        Returns:
            bool: True if the service can be scheduled without conflicts, False otherwise.
        """
        conflicts = self.find_conflicts(new_service, safety_gap, early_stop)
        return not conflicts


class SupplyGenerator(SupplySaver):
    """
    A SupplyGenerator is a class that generates supply entities based on configuration probabilities.

    Attributes:
        stations (Dict[str, Station]): Dictionary of station id to Station object.
        time_slots (Dict[str, TimeSlot]): Dictionary of time slot id to TimeSlot object.
        corridors (Dict[str, Corridor]): Dictionary of corridor id to Corridor object.
        lines (Dict[str, Line]): Dictionary of line id to Line object.
        seats (Dict[str, Seat]): Dictionary of seat id to Seat object.
        rolling_stocks (Dict[str, RollingStock]): Dictionary of rolling stock id to RollingStock object.
        tsps (Dict[str, TSP]): Dictionary of TSP id to TSP object.
        services (List[Service]): List of Service objects.
        config (Dict[str, Any]): Configuration dictionary for the generator.
        without_conflicts (bool): Whether to generate services without conflicts.
        service_scheduler (ServiceScheduler): Service scheduler for managing service conflicts.
    """

    def __init__(
        self,
        stations: Mapping[str, Station],
        time_slots: Mapping[str, TimeSlot],
        corridors: Mapping[str, Corridor],
        lines: Mapping[str, Line],
        seats: Mapping[str, Seat],
        rolling_stocks: Mapping[str, RollingStock],
        tsps: Mapping[str, TSP],
        services: List[Service],
        config: Mapping[str, Any],
        without_conflicts: bool = True
    ) -> None:
        """
        Initialize a SupplyGenerator with the given parameters.
        
        Unlike the other Supply classes, it is necessary to have the raw supply data from the YAML to generate the services.

        Args:
            stations (Mapping[str, Station]): Mapping of station id to Station object.
            time_slots (Mapping[str, TimeSlot]): Mapping of time slot id to TimeSlot object.
            corridors (Mapping[str, Corridor]): Mapping of corridor id to Corridor object.
            lines (Mapping[str, Line]): Mapping of line id to Line object.
            seats (Mapping[str, Seat]): Mapping of seat id to Seat object.
            rolling_stocks (Mapping[str, RollingStock]): Mapping of rolling stock id to RollingStock object.
            tsps (Mapping[str, TSP]): Mapping of TSP id to TSP object.
            services (List[Service]): List of Service objects.
            config (Mapping[str, Any]): Configuration mapping for the generator.
            without_conflicts (bool, optional): Whether to generate services without conflicts. Defaults to True.
        """
        SupplySaver.__init__(self, services)
        self.stations = stations
        self.time_slots = time_slots
        self.corridors = corridors
        self.lines = lines
        self.seats = seats
        self.rolling_stocks = rolling_stocks
        self.tsps = tsps
        self.services = services
        self.config = config
        self.without_conflicts = without_conflicts
        self.service_scheduler = ServiceScheduler(services=[], without_conflicts=without_conflicts)

    @classmethod
    def from_yaml(
        cls,
        path_config_supply: str,
        path_config_generator: str,
    ) -> 'SupplyGenerator':
        """
        Create a SupplyGenerator object from YAML configuration files.

        Args:
            path_config_supply (str): Path to the supply configuration YAML file.
            path_config_generator (str): Path to the generator configuration YAML file.

        Returns:
            SupplyGenerator: An instance of the SupplyGenerator class.
        """
        data = read_yaml(path_config_supply)
        stations = SupplySaver._get_stations(data, key='stations')
        time_slots = SupplySaver._get_time_slots(data, key='timeSlot')
        corridors = SupplySaver._get_corridors(data, stations, key='corridor')
        lines = SupplySaver._get_lines(data, corridors, key='line')
        seats = SupplySaver._get_seats(data, key='seat')
        rolling_stocks = SupplySaver._get_rolling_stock(data, seats, key='rollingStock')
        tsps = SupplySaver._get_tsps(data, rolling_stocks, key='trainServiceProvider')
        services = list(SupplySaver._get_services(data, lines, tsps, time_slots, seats, rolling_stocks, key='service').values())
        config = read_yaml(path_config_generator)
        return cls(stations, time_slots, corridors, lines, seats, rolling_stocks, tsps, services, config)

    def _filter_rolling_stocks(self) -> None:
        """
        Filter the rolling stocks of the train service providers to only include those that are used in the services.

        This is needed to avoid having rolling stocks used by the train
        service providers that are not defined in the generated YAML file.
        """
        used_rolling_stocks = set()
        for service in self.services:
            used_rolling_stocks.add(service.rolling_stock)
        for service in self.services:
            service.tsp.rolling_stock = list(filter(lambda x: x in used_rolling_stocks, service.tsp.rolling_stock))

    def _generate_date(self) -> datetime.date:
        """
        Generate a random date between the min and max dates specified in the config.

        The min and max dates are expected to be in the format 'YYYY-MM-DD', upper range is exclusive.

        Returns:
            datetime.date: Randomly generated date.
        """
        start, end = self.config['dates']['min'], self.config['dates']['max']
        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = np.random.randint(int_delta)
        return start + datetime.timedelta(seconds=random_second)

    def _generate_line(self) -> Tuple[Line, Line]:
        """
        Generate a line based on the configuration probabilities.

        Returns:
            Tuple[Line, Line]: Tuple of generated line and sampled line.
        """
        sampled_line: Line = self._sample_probabilities_from_config(key='lines')

        # Add normal noise to the arrival and departure times
        if self.config['lines'].get('noise_std', None):
            timetable = {}
            travel_time_noise, stop_time_noise = np.random.normal(0, self.config['lines']['noise_std'], size=2)
            for i, (station, (arrival, departure)) in enumerate(sampled_line.timetable.items()):
                noisy_arrival = float(round(arrival * (1 + travel_time_noise), ndigits=1))
                noisy_departure = float(round(noisy_arrival + (departure - arrival) * (1 + stop_time_noise), ndigits=1))
                if i == 0:
                    timetable[station] = (arrival, departure)
                elif i == len(sampled_line.timetable) - 1:
                    timetable[station] = (noisy_arrival, noisy_arrival)
                else:
                    timetable[station] = (noisy_arrival, noisy_departure)

            # Generate a unique line ID based on the timetable with 5 digits
            timetable_json = json.dumps(timetable, sort_keys=True)
            line_id = hashlib.md5(timetable_json.encode()).hexdigest()[:5]

            # Create a new line with the updated timetable
            line = Line(line_id, sampled_line.name, sampled_line.corridor, timetable)
        return line, sampled_line

    def _generate_tsp(self, tsp_id: Union[str, None] = None) -> TSP:
        """
        Generate a TSP based on the configuration probabilities.

        Args:
            tsp_id (Union[str, None], optional): TSP id to generate the TSP for. Defaults to None.

        Returns:
            TSP: Generated TSP based on the configuration probabilities.
        """
        if tsp_id:
            return self.tsps[tsp_id]
        return self._sample_probabilities_from_config(key='tsps')

    def _generate_time_slot(self, date: datetime.date) -> TimeSlot:
        """
        Generate a time slot based on the configuration probabilities.

        Returns:
            TimeSlot: Generated time slot based on the configuration probabilities.
        """
        start_hour = self._sample_scipy_distribution_from_config(key='time_slots', date=date, is_discrete=True)
        minutes = np.random.randint(0, 60)
        start = datetime.timedelta(hours=int(start_hour), minutes=minutes)
        # Clip the start time to be within the range of 00:00 to 23:59 - TIME_SLOT_SIZE
        start = max(min(start, datetime.timedelta(hours=23, minutes=59 - TIME_SLOT_SIZE)), datetime.timedelta(hours=0, minutes=0))
        end = start + datetime.timedelta(minutes=TIME_SLOT_SIZE)
        time_slot_id = f'{start.seconds}'
        return TimeSlot(time_slot_id, start, end)

    def _generate_rolling_stock(self, tsp: TSP) -> RollingStock:
        """
        Generate a rolling stock based on the configuration probabilities.

        Args:
            tsp (TSP): TSP for which to generate the rolling stock.

        Returns:
            RollingStock: Generated rolling stock based on the configuration probabilities.
        """
        return self._sample_probabilities_from_config(key='rolling_stocks', id=tsp.id)

    def _generate_prices(
        self,
        date: datetime.date,
        line: Line,
        tsp: TSP,
        time_slot: TimeSlot,
        rolling_stock: RollingStock
    ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Generate prices for the given services attributes based on the configuration probabilities and factors.

        Args:
            date (datetime.date): Date for which to generate prices.
            line (Line): Line for which to generate prices.
            tsp (TSP): TSP for which to generate prices.
            time_slot (TimeSlot): Time slot for which to generate prices.
            rolling_stock (RollingStock): Rolling stock for which to generate prices.
        
        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Dictionary of station pairs to prices.
        """
        prices = {}
        seats = [self.seats[str(seat_id)] for seat_id in rolling_stock.seats if str(seat_id) in self.seats]

        # Get the min and max prices from the configuration
        assert 'min' in self.config['prices'], "'min' not found in 'prices'"
        assert 'max' in self.config['prices'], "'max' not found in 'prices'"
        min_price = self.config['prices']['min']
        max_price = self.config['prices']['max']

        # Generate a base price for the service
        base_price = self._sample_scipy_distribution_from_config(key='prices', date=date, is_discrete=False)

        # Get the factors for the line, TSP, and time slot
        line_factor = self._sample_factor_from_config(key='prices', factor='line', id=line.id)
        tsp_factor = self._sample_factor_from_config(key='prices', factor='tsp', id=tsp.id)
        time_slot_hour = str(time_slot.start.seconds // 3600)
        time_slot_factor = self._sample_factor_from_config(key='prices', factor='time_slot', id=time_slot_hour)
        base_price *= line_factor * tsp_factor * time_slot_factor

        # Get the factors for each pair of stations and each seat
        for pair in line.pairs:
            origin_station, destination_station = line.pairs[pair]
            origin_station_factor = self._sample_factor_from_config(key='prices', factor='station', id=origin_station.id)
            destination_station_factor = self._sample_factor_from_config(key='prices', factor='station', id=destination_station.id)
            distance = geodesic(origin_station.coordinates, destination_station.coordinates).kilometers
            distance_factor = distance * self._sample_factor_from_config(key='prices', factor='distance')
            price = (base_price + distance_factor) * origin_station_factor * destination_station_factor
            prices[pair] = {}
            for seat in seats:
                seat_factor = self._sample_factor_from_config(key='prices', factor='seat', id=seat.id)
                price *= seat_factor
                prices[pair][seat] = float(round(np.clip(price, min_price, max_price), 2))
        return prices

    def _generate_service_id(self) -> str:
        """
        Generate a unique service ID based on the current number of services.

        Returns:
            str: Unique service ID.
        """
        return str(len(self.services) + 1).zfill(5)

    def _generate_service(
        self,
        tsp_id: Union[str, None],
        safety_gap: int = SAFETY_GAP,
        max_retry: int = MAX_RETRY
    ) -> Union[Service, None]:
        """
        Generate a service based on the configuration probabilities.

        It checks if the service is feasible and generates a new one if not until max_retry is reached.

        Args:
            tsp_id (Union[str, None], optional): TSP id to generate the service for.
            safety_gap (int, optional): Safety gap of the segments in minutes.
            max_retry (int, optional): Maximum number of retries to generate a feasible service.

        Returns:
            Union[Service, None]: Generated service if possible, None if max retries are reached.
        """
        retries = 0
        feasible = False
        while not feasible:
            date = self._generate_date()
            line, sampled_line = self._generate_line()
            tsp = self._generate_tsp(tsp_id)
            time_slot = self._generate_time_slot(date)
            rolling_stock = self._generate_rolling_stock(tsp)
            prices = self._generate_prices(date, sampled_line, tsp, time_slot, rolling_stock)
            service_id = self._generate_service_id()
            service = Service(service_id, date, line, tsp, time_slot, rolling_stock, prices)
            if not self.without_conflicts:
                break
            feasible = self.service_scheduler.is_feasible(service, safety_gap)
            retries += 1
            if retries > max_retry:
                logger.warning(f'Max retries reached. A feasible service could not be generated.')
                raise UnfeasibleServiceException
        return service

    @cache
    def _get_distributions_for_key(self, key: str, is_discrete: bool) -> Dict[str, Any]:
        """
        Get the SciPy distributions for the given key from the configuration.

        Args:
            key (str): The key in the configuration to get distributions for.
            is_discrete (bool): Whether the distributions are discrete or continuous.

        Returns:
            Dict[str, Any]: Dictionary mapping distribution IDs to their distribution objects and kwargs.
        """
        assert key in self.config, f"'{key}' not found in config"
        distributions = {}
        for distribution in self.config[key]['probabilities']:
            assert 'id' in distribution, f"'id' not found in '{key}'"
            _id = distribution['id']
            assert 'distribution' in distribution, f"'distribution' not found in distrbution '{_id}' of '{key}'"
            assert 'distribution_kwargs' in distribution, f"'distribution_kwargs' not found in distribution '{_id}' of '{key}'"
            distribution_name = distribution['distribution']
            distribution_kwargs = distribution['distribution_kwargs']
            distribution, distribution_kwargs = get_scipy_distribution(
                distribution_name, is_discrete, **distribution_kwargs
            )
            distributions[_id] = {'distribution': distribution, 'kwargs': distribution_kwargs}
        return distributions

    def _sample_distribution_by_date(self, key: str, distributions: Mapping[str, Any], date: datetime.date) -> Any:
        """
        Sample a value from a distribution based on the date.

        Args:
            key (str): The key in the configuration to sample from.
            distributions (Mapping[str, Any]): Mapping of distribution id to distribution object.
            date (datetime.date): The date to use for sampling.

        Returns:
            Any: The sampled value from the distribution.
        """
        assert 'days' in self.config[key], f"'days' not found in '{key}'"
        # Get the day of the week in the format 'Monday', 'Tuesday', etc
        day_of_week = date.strftime('%A')
        assert day_of_week in self.config[key]['days'], f"'{day_of_week}' not found in days of '{key}'"
        distribution_day_of_week = self.config[key]['days'][day_of_week]
        assert distribution_day_of_week in distributions, \
            f"distribution '{distribution_day_of_week}' not found in distributions of '{key}'"
        return distributions[distribution_day_of_week]['distribution'].rvs(
            **distributions[distribution_day_of_week]['kwargs']
        )

    def _sample_factor_from_config(self, key: str, factor: str, id: Union[str, None] = None) -> float:
        """
        Sample a factor from the configuration for the given key and id.

        Args:
            key (str): The key in the configuration to sample from.
            factor (str): The factor to sample.
            id (Union[str, None], optional): The id of the item to sample. Defaults to None.
        
        Returns:
            float: The sampled factor value.
        """
        assert key in self.config, f"'{key}' not found in config"
        factor = factor + '_factor'
        assert factor in self.config[key], f"'{factor}' not found in '{key}'"
        if not id:
            return self.config[key][factor]
        if id not in self.config[key][factor]:
            assert 'default' in self.config[key][factor], f"'{id}' and 'default' factors not found in '{factor}' of '{key}'"
            id = 'default'
        return self.config[key][factor][id]
        
    def _sample_probabilities_from_config(self, key: str, id: Union[str, None] = None) -> Any:
        """
        Sample an item from the configuration probabilities for the given key.

        If probabilities are not defined, all items will have the same probability.

        Args:
            key (str): The key in the configuration to sample from.
            id (Union[str, None], optional): The id of the item to sample. Defaults to None.

        Returns:
            Any: The sampled item from the configuration.
        """
        if id:
            assert id in self.config[key]['probabilities'], f"'{id}' not found in '{key}'"
            items = list(self.config[key]['probabilities'][id].keys())
            probabilities = list(self.config[key]['probabilities'][id].values())
        else:
            assert key in self.config, f"'{key}' not found in config"
            items = list(self.config[key]['probabilities'].keys())
            probabilities = list(self.config[key]['probabilities'].values())
        
        # If probabilities are not defined, assign equal probability to all items
        if not probabilities or sum(probabilities) == 0:
            items = list(getattr(self, key).keys())
            probabilities = [1 / len(items)] * len(items)
        
        sampled_item = np.random.choice(items, p=probabilities)
        assert sampled_item in getattr(self, key), f"'{sampled_item}' not found in '{key}'"
        return getattr(self, key)[sampled_item]

    def _sample_scipy_distribution_from_config(
        self,
        key: str,
        date: datetime.date,
        is_discrete: bool
    ) -> Union[np.int64, np.float64]:
        """
        Sample a value from a SciPy distribution based on the configuration probabilities.

        Args:
            key (str): The key in the configuration to sample from.
            date (datetime.date): The date to use for sampling.
            is_discrete (bool): Whether the distribution is discrete or continuous.

        Returns:
            Union[np.int64, np.float64]: The sampled value from the distribution.
        """
        distributions = self._get_distributions_for_key(key, is_discrete)
        sampled_value = self._sample_distribution_by_date(key, distributions, date)
        return sampled_value

    def generate(
        self,
        n_services: int = N_SERVICES,
        n_services_by_tsp: Mapping[str, int] = None,
        output_path: Union[str, None] = None,
        seed: Union[int, None] = None,
        progress_bar: bool = True,
        safety_gap: int = SAFETY_GAP,
        max_retry: int = MAX_RETRY,
        without_conflicts: bool = True
    ) -> List[Service]:
        """
        Generate a list of services.

        If the optional parameter n_services_by_tsp is provided, then for each TSP the specified number
        of services will be generated. Otherwise, n_services will be used as a global counter.

        Args:
            output_path (str, optional): Path to the output YAML file. Defaults to None.
            n_services (int, optional): Number of services to generate. Defaults to 1.
            n_services_by_tsp (Mapping[str, int], optional): Mapping of TSP id to number of services
                to generate for each TSP. Defaults to None.
            seed (int, optional): Seed for the random number generator.
            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.
            safety_gap (int, optional): Safety gap of the segments in minutes. Defaults to 10.
            max_retry (int, optional): Maximum number of retries to generate a feasible service. Defaults to 500.
            without_conflicts (bool, optional): Whether to generate services without conflicts. Defaults to True.

        Returns:
            List[Service]: List of generated Service objects.
        """
        # Set seed for reproducibility
        if seed is not None:
            self.set_seed(seed)
        
        # Initialize the mapping for the number of services to generate per TSP if not provided
        if n_services != N_SERVICES and n_services_by_tsp:
            logger.warning('Both n_services and n_services_by_tsp are provided. Using n_services_by_tsp.')
        if not n_services_by_tsp:
            n_services_by_tsp = {None: n_services}

        # Interoperability of the loguru logger with tqdm
        if progress_bar:
            logger.remove()
            logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)

        # Generate services
        self.services: List[Service] = []
        self.without_conflicts = without_conflicts
        self.service_scheduler = ServiceScheduler(services=[], without_conflicts=self.without_conflicts)
        for tsp_id, count in n_services_by_tsp.items():
            iterator = range(count)
            tsp_id_str = tsp_id if tsp_id else 'all'
            iterator = tqdm(iterator, desc=f'Generating services {tsp_id_str}', unit='service') if progress_bar else iterator
            for _ in iterator:
                try:
                    generated_service = self._generate_service(tsp_id, safety_gap, max_retry)
                    self.services.append(generated_service)
                    self.service_scheduler.add_service(generated_service)
                except UnfeasibleServiceException:
                    logger.warning(f'Unfeasible service generated. Stopping generation with {len(self.services)} generated services.')
                    break
        
        # Save the generated services to a YAML file
        self._filter_rolling_stocks()
        if output_path:
            SupplySaver(self.services).to_yaml(output_path)
        return self.services

    def set_seed(self, seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
