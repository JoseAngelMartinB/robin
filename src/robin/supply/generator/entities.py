"""Entities for the supply generator module."""

import datetime
import networkx as nx
import numpy as np
import os
import random

from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from robin.supply.saver.entities import SupplySaver

from robin.supply.generator.utils import get_distance, read_yaml

from geopy.distance import geodesic
from pathlib import Path
from typing import Any, FrozenSet, List, Mapping, Set, Union, Tuple


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
        seed (Union[int, None]): Seed for the random number generator.
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
        seed: Union[int, None] = None
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
            seed (Union[int, None], optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            self.set_seed(seed)
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
        seed: Union[int, None] = None
    ) -> 'SupplyGenerator':
        """
        Create a SupplyGenerator object from YAML configuration files.

        Args:
            path_config_supply (str): Path to the supply configuration YAML file.
            path_config_generator (str): Path to the generator configuration YAML file.
            seed (Union[int, None], optional): Seed for the random number generator. Defaults to None.
        
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
        services = SupplySaver._get_services(data, lines, tsps, time_slots, seats, rolling_stocks, key='service')
        config = read_yaml(path_config_generator)
        return cls(stations, time_slots, corridors, lines, seats, rolling_stocks, tsps, services, config, seed)

    def _build_graph(self, tree: Mapping[Station, ...], graph=None) -> nx.Graph:
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
                self._build_graph({destination_station: branches[destination_station]}, graph)

        return graph

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
        return self._sample_from_config(key='lines')

        # TODO: Review this, tt? dt? travel time? departure time? Why we use 0.4 and 0.5? Is not directly to take a line from the supply?
        # NOTE: I think this is not needed as we now take the line from the supply
        # timetable = {}
        # tt_randomizer = np.random.uniform(low=0.0, high=0.4)
        # dt_randomizer = np.random.uniform(low=0.0, high=0.5)
        # for i, station in enumerate(line.timetable):
        #     arrival, departure = line.timetable[station]
        #     if i == 0:
        #         prev_dt = departure
        #     else:
        #         ref_stop_time = departure - arrival
        #         travel_time = arrival - prev_dt
        #         arrival = float(round(prev_dt + (travel_time + travel_time * tt_randomizer)))
        #         departure = float(round(arrival + ref_stop_time + ref_stop_time * dt_randomizer))
        #     timetable[station] = (arrival, departure)

        # Encode timetable to string (Hash or something) for unique line id based on timetable
        # line_id = str(hash(str(timetable.values())))
        # return Line(f'Line_{line_id}', line.name, line.corridor, timetable)

    def _get_edges_from_path(self, path: List[Station]) -> Set[FrozenSet[Station]]:
        """
        Returns the set of edges (as frozensets) traversed in the given path.

        Args:
            path (List[Station]): A list of Station objects representing the path.

        Returns:
            A set of frozensets, each containing two Station objects that form an edge.
        """
        edges: Set[FrozenSet[Station]] = set()
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edges.add(frozenset((u, v)))
        return edges

    def _generate_tsp(self) -> TSP:
        """
        Generate a TSP based on the configuration probabilities.

        Returns:
            TSP: Generated TSP based on the configuration probabilities.
        """
        return self._sample_from_config(key='tsps')

    def _generate_time_slot(self) -> TimeSlot:
        """
        Generate a time slot based on the configuration probabilities.

        Returns:
            TimeSlot: Generated time slot based on the configuration probabilities.
        """
        return self._sample_from_config(key='time_slots')

        # TODO: Same as for the line
        # ts_probabilities = self.config['time_slots']['probabilities']
        # hour = random.choices(list(ts_probabilities.keys()), weights=list(ts_probabilities.values()))[0]
        # minutes = random.randint(0, 59)
        # start_time = datetime.timedelta(hours=hour, minutes=minutes)
        # end_time = start_time + datetime.timedelta(minutes=10)
        # if end_time >= datetime.timedelta(hours=24):
        #     # Decrease time by a full day
        #     end_time -= datetime.timedelta(days=1)
        # time_slot_id = f'{start_time.seconds}'
        # time_slot = TimeSlot(time_slot_id, start_time, end_time)
        # return time_slot

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
            distance = get_distance(line, origin_sta, destination_sta)
            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = str(round(price_calc, 2)) if price_calc < max_price else max_price

        return prices

    def _generate_service_id(self, line: Line, date: datetime.date, time_slot: TimeSlot) -> str:
        """
        Generate a unique service ID based on the line, date, and time slot.

        Args:
            line (Line): Line of the service.
            date (datetime.date): Date of the service.
            time_slot (TimeSlot): Time slot of the service.

        Returns:
            str: Unique service ID.
        """
        date_str = date.strftime('%Y-%m-%d')
        time_str = '.'.join(str(time_slot.start).split(':')[:2])
        return f'{line.id}_{date_str}-{time_str}'

    def _generate_service(self) -> Service:
        """
        Generate a service based on the configuration probabilities.

        It checks if the service is feasible and generates a new one if not. (TODO: Implement feasibility check)

        Returns:
            Service: Generated service based on the configuration probabilities.
        """
        feasible = False

        while not feasible:
            date = self._generate_date()
            line = self._generate_line()
            tsp = self._generate_tsp()
            time_slot = self._generate_time_slot()
            rolling_stock = self._generate_rolling_stock(tsp)
            prices = self._generate_prices(line, rolling_stock, tsp)
            service_id = self._generate_service_id(line, date, time_slot)
            service = Service(service_id, date, line, tsp, time_slot, rolling_stock, prices)

            # TODO: Check if the service is feasible
            feasible = self._is_feasible(service, self.services)

        return service

    def _sample_from_config(self, key: str, id: Union[str, None] = None) -> Any:
        """
        Sample an item from the configuration probabilities for the given key.

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
        sampled_item = np.random.choice(items, p=probabilities)
        assert sampled_item in getattr(self, key), f'{sampled_item} not found in {key}'
        return getattr(self, key)[sampled_item]

    def _infer_path(self, service: Service) -> List[Station]:
        """
        Infers the path of a service based on its line and corridor.

        Args:
            service (Service): The service to infer.

        Returns:
            List[Station]: The inferred path of the service.
        """
        paths = []
        for path in service.line.corridor.paths:
            if all(station in path for station in service.line.stations):
                paths.append(path)

        # Return the shortest path
        if paths:
            path = min(paths, key=len)
        else:
            raise ValueError("No valid path found for the service.")

        # Clip the path from origin to destination
        origin_index = path.index(service.line.stations[0])
        destination_index = path.index(service.line.stations[-1])
        return path[origin_index:destination_index + 1]

    def _is_feasible(
            self,
            new_service: Service,
            safety_headway: int = 10,
    ) -> bool:
        """
        Check if the service is feasible.

        Args:
            service (Service): Service to check.
            services (List[Service]): List of existing services.
            safety_headway (int): Safety headway in minutes.

        Returns:
            bool: True if the service is feasible, False otherwise.
        """
        # Check each pair of services once
        for service in self.services:
            shared_edges = self._shared_edges_between_services(
                self._infer_path(service),
                self._infer_path(new_service)
            )
            if not shared_edges:
                continue

            for edge in shared_edges:
                station1, station2 = list(edge)

                departure1, arrival1 = sorted([new_service.schedule[station1], new_service.schedule[station2]])
                departure2, arrival2 = sorted([service.schedule[station1], service.schedule[station2]])

                # Expand time intervals with buffer
                interval1_start = departure1 + new_service.date - safety_headway
                interval1_end = arrival1 + new_service.date + safety_headway
                interval2_start = departure2 + service.date - safety_headway
                interval2_end = arrival2 + service.date + safety_headway

                # Check overlap
                if interval1_start <= interval2_end and interval2_start <= interval1_end:
                    return False

        return True

    def _shared_edges_between_services(
            self,
            path1: List[Station],
            path2: List[Station]
    ) -> Set[FrozenSet[Station]]:
        """Checks whether two services (given by their paths) share any track segments.

        Args:
            path1: A list of Station objects for the first service.
            path2: A list of Station objects for the second service.

        Returns:
            A set of shared edges (track segments), if any.
        """
        edges1 = self._get_edges_from_path(path1)
        edges2 = self._get_edges_from_path(path2)
        return edges1.intersection(edges2)

    def generate(
        self,
        file_name: str,
        n_services: int = 1,
        n_services_by_ru: Mapping[str, int] = None,
        seed: int = None,
        safety_gap: int = 10,
    ) -> List[Service]:
        """
        Generate a list of services.

        If the optional parameter n_services_by_ru is provided (a mapping from RU id to number of services),
        then for each RU the specified number of services will be generated. Otherwise, n_services will be used
        as a global counter.

        Args:
            file_name (str): Name of the output file.
            n_services (int, optional): Number of services to generate (if n_services_by_ru is not provided). Defaults to 1.
            n_services_by_ru (Mapping[str, int], optional): Mapping of RU id (TSP id) to the number of services to generate.
            seed (int, optional): Seed for the random number generator.

        Returns:
            List[Service]: List of generated Service objects.
        """
        self.safety_gap = safety_gap

        services = []
        # Generate services per RU if a mapping is provided
        #if n_services_by_ru is not None:
        #    for ru_id, count in n_services_by_ru.items():
        #        for j in range(count):
        #            service_id = f"{ru_id}_{j}"
        #            service = self._generate_service_for_ru(ru_id, service_id)
        #            services.append(service)
        #else:
        for _ in range(n_services):
            services.append(self._generate_service())

        self.services = services
        #self.to_yaml(output_path=file_name)
        return services

    def set_seed(seed: int) -> None:
        """
        Set seed for the random number generator.

        Args:
            seed (int): Seed for the random number generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    path_config_supply = 'configs/supply_generator/supply_data.yaml'
    path_config_generator = 'configs/supply_generator/config.yaml'
    generator = SupplyGenerator.from_yaml(path_config_supply, path_config_generator)
    print('Config:', generator.config)
    print('Random date:', generator._generate_date())
    print('Random line:', generator._generate_line())
    print('Random TSP:', generator._generate_tsp())
    print('Random time slot:', generator._generate_time_slot())
    print('Random rolling stock:', generator._generate_rolling_stock(generator._generate_tsp()))
