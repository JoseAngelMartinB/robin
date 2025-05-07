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

from robin.supply.generator.exceptions import UnfeasibleServiceException
from robin.supply.generator.constants import MAX_RETRY, N_SERVICES, SAFETY_GAP, TIME_SLOT_SIZE
from robin.supply.generator.utils import read_yaml

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cache
from geopy.distance import geodesic
from loguru import logger
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Mapping, Union, Tuple


@dataclass
class Segment:
    service_idx: str
    path: List[Station]
    time_at: Callable[[Station], datetime]
    edges: List[Tuple[Station, Station]] = field(init=False)

    def __post_init__(self):
        # Precompute the edges based on the path
        self.edges = [(self.path[i], self.path[i+1]) for i in range(len(self.path) - 1)]


class ServiceScheduler:
    def __init__(self, services: List[Service]):
        self.services = services
        # Cached data structures
        self.graphs: Dict[str, nx.Graph] = {}
        self.segments_cache: Dict[str, List[Segment]] = {}
        self.edge_index: Dict[Tuple[Station, Station], List[Segment]] = defaultdict(list)

        for service in services:
            self._add_service_to_index(service)

    def _add_service_to_index(self, svc: Service):
        corr_id = svc.line.corridor.id
        if corr_id not in self.graphs:
            self.graphs[corr_id] = self.build_graph(svc.line.corridor.tree)

        G = self.graphs[corr_id]
        segs = self._build_segments_for_service(G, svc)
        self.segments_cache[svc.id] = segs

        for seg in segs:
            for edge in seg.edges:
                self.edge_index[edge].append(seg)

    def _build_segments_for_service(
            self,
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
            midnight = datetime.time.min  # 00:00
            dep_delta = service.schedule[prev_stn.id][1]  # timedelta desde medianoche
            arr_delta = service.schedule[next_stn.id][0]

            dep_time = datetime.datetime.combine(service.date, midnight) + dep_delta
            arr_time = datetime.datetime.combine(service.date, midnight) + arr_delta

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

    def _segments_conflict(
            self,
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

    def add_service(self, new_service: Service):
        if not self.is_feasible(new_service):
            raise ValueError("Conflict detected!")
        self.services.append(new_service)
        self._add_service_to_index(new_service)


    def build_graph(self, tree: Mapping[Station, ...], graph=None) -> nx.Graph:
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
                self.build_graph({destination_station: branches[destination_station]}, graph)

        return graph

    def is_feasible(self, new_service: Service, safety_gap: int = SAFETY_GAP) -> bool:
        corr_id = new_service.line.corridor.id
        # 1) Obtén el grafo (o créalo si es un corridor nuevo)
        G = self.graphs.get(corr_id)
        if G is None:
            G = self.build_graph(new_service.line.corridor.tree)
            self.graphs[corr_id] = G

        # 2) Construye sólo los segmentos **del nuevo servicio**
        new_segs = self._build_segments_for_service(G, new_service)

        # 3) Para cada nuevo segmento y cada una de sus aristas,
        #    sólo compara con los viejos segmentos que pasan por esa misma arista
        for seg_new in new_segs:
            for edge in seg_new.edges:
                for seg_old in self.edge_index.get(edge, []):
                    if self._segments_conflict(seg_new, seg_old, safety_gap):
                        return False
        return True

    def add_service(self, new_service: Service):
        """Llamar tras aprobar su factibilidad."""
        if not self.is_feasible(new_service):
            raise ValueError("Conflict detected!")
        self.services.append(new_service)
        self._add_service_to_index(new_service)



class SupplyGenerator(SupplySaver):
    """
    A SupplyGenerator is a class that generates supply entities based on configuration probabilities.

    Attributes:
        stations (Mapping[str, Station]): Mapping of station id to Station object.
        time_slots (Mapping[str, TimeSlot]): Mapping of time slot id to TimeSlot object.
        corridors (Mapping[str, Corridor]): Mapping of corridor id to Corridor object.
        lines (Mapping[str, Line]): Mapping of line id to Line object.
        seats (Mapping[str, Seat]): Mapping of seat id to Seat object.
        rolling_stocks (Mapping[str, RollingStock]): Mapping of rolling stock id to RollingStock object.
        tsps (Mapping[str, TSP]): Mapping of TSP id to TSP object.
        services (List[Service]): List of Service objects.
        config (Mapping[str, Any]): Configuration mapping for the generator.
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

    def _generate_line(self) -> Line:
        """
        Generate a line based on the configuration probabilities.

        Returns:
            Line: Generated line based on the configuration probabilities.
        """
        line: Line = self._sample_from_config(key='lines')

        # Add normal noise to the arrival and departure times
        if self.config['lines'].get('noise_std', None):
            timetable = {}
            travel_time_noise, stop_time_noise = np.random.normal(0, self.config['lines']['noise_std'], size=2)
            for i, (station, (arrival, departure)) in enumerate(line.timetable.items()):
                noisy_arrival = float(round(arrival * (1 + travel_time_noise), ndigits=1))
                noisy_departure = float(round(noisy_arrival + (departure - arrival) * (1 + stop_time_noise), ndigits=1))
                if i == 0:
                    timetable[station] = (arrival, departure)
                elif i == len(line.timetable) - 1:
                    timetable[station] = (noisy_arrival, noisy_arrival)
                else:
                    timetable[station] = (noisy_arrival, noisy_departure)

            # Generate a unique line ID based on the timetable with 5 digits
            timetable_json = json.dumps(timetable, sort_keys=True)
            line_id = hashlib.md5(timetable_json.encode()).hexdigest()[:5]

            # Create a new line with the updated timetable
            line = Line(line_id, line.name, line.corridor, timetable)
        return line

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
        return self._sample_from_config(key='tsps')

    def _generate_time_slot(self, date: datetime.date) -> TimeSlot:
        """
        Generate a time slot based on the configuration probabilities.

        Returns:
            TimeSlot: Generated time slot based on the configuration probabilities.
        """
        start_hour = self._sample_scipy_distribution_from_config(key='time_slots', date=date)
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
        return self._sample_from_config(key='rolling_stocks', id=tsp.id)

    def _generate_prices(
        self,
        line: Line,
        rolling_stock: RollingStock,
        tsp: TSP
    ) -> Mapping[Tuple[str, str], Mapping[Seat, float]]:
        """
        """
        prices = {}
        hard_types = rolling_stock.seats.keys()
        seats = list(filter(lambda s: s.hard_type in hard_types, list(self.seats.values())))
        base_price = self.config['prices']['base']
        max_price = self.config['prices']['max']
        distance_factor = self.config['prices']['distance_factor']
        seat_factor = self.config['prices']['seat_type_factor']
        tsp_factor = self.config['prices']['tsp_factor']
        for pair in line.pairs:
            origin_sta, destination_sta = line.pairs[pair]
            distance = geodesic(origin_sta.coordinates, destination_sta.coordinates).kilometers
            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = str(round(price_calc, 2)) if price_calc < max_price else max_price
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
            line = self._generate_line()
            tsp = self._generate_tsp(tsp_id)
            time_slot = self._generate_time_slot(date)
            rolling_stock = self._generate_rolling_stock(tsp)
            prices = self._generate_prices(line, rolling_stock, tsp)
            service_id = self._generate_service_id()
            service = Service(service_id, date, line, tsp, time_slot, rolling_stock, prices)
            feasible = self.service_scheduler.is_feasible(service, safety_gap)
            retries += 1
            if retries > max_retry:
                logger.warning(f'Max retries reached. A feasible service could not be generated.')
                raise UnfeasibleServiceException
        return service

    @cache
    def _get_distributions_for_key(self, key: str) -> dict:
        """
        Get the SciPy distributions for the given key from the configuration.

        Args:
            key (str): The key in the configuration to get distributions for.

        Returns:
            dict: Dictionary mapping distribution IDs to their distribution objects and kwargs.
        """
        assert key in self.config, f'{key} not found in config'
        distributions = {}
        for distribution in self.config[key]['probabilities']:
            assert 'id' in distribution, f'id not found in {key}'
            _id = distribution['id']
            assert 'distribution' in distribution, f'distribution not found in distrbution {_id} of {key}'
            assert 'distribution_kwargs' in distribution, f'distribution_kwargs not found in distribution {_id} of {key}'
            distribution_name = distribution['distribution']
            distribution_kwargs = distribution['distribution_kwargs']
            distribution, distribution_kwargs = get_scipy_distribution(
                distribution_name, **distribution_kwargs, is_discrete=True
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
        assert 'days' in self.config[key], f'days not found in {key}'
        # Get the day of the week in the format 'Monday', 'Tuesday', etc
        day_of_week = date.strftime('%A')
        assert day_of_week in self.config[key]['days'], f'{day_of_week} not found in days of {key}'
        distribution_day_of_week = self.config[key]['days'][day_of_week]
        assert distribution_day_of_week in distributions, \
            f'distribution {distribution_day_of_week} not found in distributions of {key}'
        return distributions[distribution_day_of_week]['distribution'].rvs(
            **distributions[distribution_day_of_week]['kwargs']
        )

    def _sample_from_config(self, key: str, id: Union[str, None] = None) -> Any:
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
            assert id in self.config[key]['probabilities'], f'{id} not found in {key}'
            items = list(self.config[key]['probabilities'][id].keys())
            probabilities = list(self.config[key]['probabilities'][id].values())
        else:
            assert key in self.config, f'{key} not found in config'
            items = list(self.config[key]['probabilities'].keys())
            probabilities = list(self.config[key]['probabilities'].values())
        
        # If probabilities are not defined, assign equal probability to all items
        if not probabilities or sum(probabilities) == 0:
            items = list(getattr(self, key).keys())
            probabilities = [1 / len(items)] * len(items)
        
        sampled_item = np.random.choice(items, p=probabilities)
        assert sampled_item in getattr(self, key), f'{sampled_item} not found in {key}'
        return getattr(self, key)[sampled_item]

    def _sample_scipy_distribution_from_config(self, key: str, date: datetime.date) -> Union[np.int64, np.float64]:
        """
        Sample a value from a SciPy distribution based on the configuration probabilities.

        Args:
            key (str): The key in the configuration to sample from.
            date (datetime.date): The date to use for sampling.

        Returns:
            Union[np.int64, np.float64]: The sampled value from the distribution.
        """
        distributions = self._get_distributions_for_key(key)
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
        max_retry: int = MAX_RETRY
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
        self.service_scheduler = ServiceScheduler([])
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


if __name__ == '__main__':
    path_config_supply = 'configs/supply_generator/supply_data_new.yaml'
    path_config_generator = 'configs/supply_generator/config.yaml'
    generator = SupplyGenerator.from_yaml(path_config_supply, path_config_generator)
    print('Config:', generator.config)
    date = generator._generate_date()
    print('Random date:', date)
    print('Random line:', generator._generate_line())
    tsp = generator._generate_tsp()
    print('Random TSP:', tsp)
    print('Random time slot:', repr(generator._generate_time_slot(date)))
    print('Random rolling stock:', generator._generate_rolling_stock(tsp))
    print('TSP by ID:', generator._generate_tsp(tsp_id=tsp.id))
