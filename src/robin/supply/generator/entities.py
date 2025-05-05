"""Entities for the supply generator module."""

import datetime
import hashlib
import json
import numpy as np
import os
import random

from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from robin.supply.saver.entities import SupplySaver

from robin.supply.generator.exceptions import UnfeasibleServiceException
from robin.supply.generator.constants import DEFAULT_MAX_RETRY, DEFAULT_SAFETY_GAP
from robin.supply.generator.utils import build_segments_for_service, get_stations_positions, read_yaml, segments_conflict

from geopy.distance import geodesic
from loguru import logger
from tqdm import tqdm
from typing import Any, List, Mapping, Union, Tuple


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
        services = SupplySaver._get_services(data, lines, tsps, time_slots, seats, rolling_stocks, key='service')
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
            timetable_json = json.dumps(str(timetable.values()), sort_keys=True)
            line_id = hashlib.md5(timetable_json.encode()).hexdigest()[:5]

            # Create a new line with the updated timetable
            line = Line(line_id, line.name, line.corridor, timetable)
        return line

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

    def _generate_service(self, safety_gap: int, max_retry: int) -> Union[Service, None]:
        """
        Generate a service based on the configuration probabilities.

        It checks if the service is feasible and generates a new one if not until max_retry is reached.

        Args:
            safety_gap (int): Safety gap of the segments in minutes.
            max_retry (int): Maximum number of retries to generate a feasible service.

        Returns:
            Union[Service, None]: Generated service if possible, None if max retries are reached.
        """
        retries = 0
        feasible = False
        while not feasible:
            date = self._generate_date()
            line = self._generate_line()
            tsp = self._generate_tsp()
            time_slot = self._generate_time_slot()
            rolling_stock = self._generate_rolling_stock(tsp)
            prices = self._generate_prices(line, rolling_stock, tsp)
            service_id = self._generate_service_id()
            service = Service(service_id, date, line, tsp, time_slot, rolling_stock, prices)
            feasible = self._is_feasible(service, safety_gap)
            retries += 1
            if retries > max_retry:
                logger.warning(f'Max retries reached. A feasible service could not be generated.')
                raise UnfeasibleServiceException
        return service

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

    def _is_feasible(self, new_service: Service, safety_gap: int = DEFAULT_SAFETY_GAP) -> bool:
        """
        Check if the new service is feasible with respect to the existing services.

        It checks if the new service conflicts with any existing service with a safety gap.

        Args:
            new_service (Service): Service to check if feasible.
            safety_gap (int): Safety gap of the segments in minutes. Defaults to 10.

        Returns:
            bool: True if the service is feasible, False otherwise.
        """
        # Get paths of the corridor visited by the new service
        paths = []
        for path in new_service.line.corridor.paths:
            if sum([station in path for station in new_service.line.stations]) > 1:
                paths.append(path)

        for path in paths:
            new_service_segments = build_segments_for_service(new_service, get_stations_positions(path))
            for service in self.services:
                if sum([station in path for station in service.line.stations]) <= 1:
                    continue

                stations_positions = get_stations_positions(path)

                # Precompute segments per service
                service_segments = build_segments_for_service(service, stations_positions)

                # Test all segment pairs
                for seg1 in new_service_segments:
                    for seg2 in service_segments:
                        if segments_conflict(seg1, seg2, safety_gap):
                            return False

        return True

    def generate(
        self,
        n_services: int = 1,
        n_services_by_tsp: Mapping[str, int] = None,
        output_path: Union[str, None] = None,
        seed: Union[int, None] = None,
        progress_bar: bool = True,
        safety_gap: int = DEFAULT_SAFETY_GAP,
        max_retry: int = DEFAULT_MAX_RETRY,
    ) -> List[Service]:
        """
        Generate a list of services.

        If the optional parameter n_services_by_ru is provided (a mapping from RU id to number of services),
        then for each RU the specified number of services will be generated. Otherwise, n_services will be used
        as a global counter.

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
        if seed:
            self.set_seed(seed)

        # Generate services per RU if a mapping is provided
        #if n_services_by_ru is not None:
        #    for ru_id, count in n_services_by_ru.items():
        #        for j in range(count):
        #            service_id = f"{ru_id}_{j}"
        #            service = self._generate_service_for_ru(ru_id, service_id)
        #            services.append(service)
        #else:

        self.services: List[Service] = []
        iterator = range(n_services)
        if progress_bar:
            # Interoperability of the loguru logger with tqdm
            logger.remove()
            logger.add(lambda msg: tqdm.write(msg, end=''), colorize=True)
            iterator = tqdm(iterator, desc='Generating services', unit='service')
        for _ in iterator:
            try:
                generated_service = self._generate_service(safety_gap, max_retry)
                self.services.append(generated_service)
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
    path_config_supply = 'configs/supply_generator/supply_data.yaml'
    path_config_generator = 'configs/supply_generator/config.yaml'
    generator = SupplyGenerator.from_yaml(path_config_supply, path_config_generator)
    print('Config:', generator.config)
    print('Random date:', generator._generate_date())
    print('Random line:', generator._generate_line())
    tsp = generator._generate_tsp()
    print('Random TSP:', tsp)
    print('Random time slot:', generator._generate_time_slot())
    print('Random rolling stock:', generator._generate_rolling_stock(tsp))
