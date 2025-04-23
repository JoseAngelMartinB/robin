"""Entities for the supply generator module."""

import datetime
import numpy as np
import os
import random
import yaml

from robin.supply.entities import Seat, TimeSlot, TSP, Line, RollingStock, Service
from robin.supply.saver.entities import SupplySaver

from robin.supply.generator.utils import get_distance

from pathlib import Path
from typing import Dict, List, Mapping, Tuple


class SupplyGenerator(SupplySaver):
    """
    """

    def __init__(self, services: List[Service]) -> None:
        """
        """
        SupplySaver.__init__(self, services)

    def _generate_service(self) -> Service:
        """
        """
        feasible = False

        while not feasible:
            line = self._get_random_line()
            time_slot = self._get_random_time_slot()
            tsp = self._get_random_tsp()
            rolling_stock = self._get_random_rs(tsp)
            date = self._get_random_date()
            prices = self._get_random_prices(line, rolling_stock, tsp)
            date_str = date.strftime('%Y-%m-%d')
            time_str = '.'.join(str(time_slot.start).split(':')[:2])
            service_id = f'{line.id}_{date_str}-{time_str}'
            service = Service(
                id_=service_id, date=date, line=line, time_slot=time_slot,
                tsp=tsp, rolling_stock=rolling_stock, prices=prices
            )

            # TODO: Check if the service is feasible
            feasible = True

        self.services.append(service)
        return service

    def _set_config(self, path_config: Path):
        """
        Set config file

        Args:
            path_config (Path): Path to config file
        """
        with open(path_config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

    def _get_random_rs(self, tsp: TSP) -> RollingStock:
        """
        Get a random rolling stock from a TSP

        Args:
            tsp (TSP): TSP object

        Returns:
            RollingStock: Rolling stock object randomly selected from  the specified TSP
        """
        return random.choice(tsp.rolling_stock)

    def _get_random_tsp(self) -> TSP:
        """
        Get a random TSP

        Returns:
            TSP: TSP object randomly selected from the available TSPs
        """
        #tsp_probabilities = self.config['tsps']['probabilities']
        #return random.choices(list(self.tsps.values()), weights=list(tsp_probabilities.values()))[0]
        return random.choices(list(self.tsps.values()))[0]

    def _get_random_date(self) -> datetime.date:
        """
        This function will return a random datetime between two datetime objects.

        Returns:
            datetime: Random datetime in range [start, end] specified in config file
        """
        start, end = self.config['services']['dates']['min'], self.config['services']['dates']['max']

        delta = end - start
        int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
        random_second = random.randrange(int_delta)
        return start + datetime.timedelta(seconds=random_second)

    def _get_random_prices(self,
                           line: Line,
                           rolling_stock: RollingStock,
                           tsp: TSP
        ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Get prices for a service for a given line, rolling stock and TSP

        Args:
            line: Line object
            rolling_stock: RollingStock object
            tsp: TSP object

        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Prices for each pair of stations
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
            # print(f'Distance between {origin_sta.name} and {destination_sta.name}: {distance} km')
            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = str(round(price_calc, 2)) if price_calc < max_price else max_price

        return prices

    def _get_random_time_slot(self) -> TimeSlot:
        """
        Get a random time slot

        Returns:
            TimeSlot: Time slot object
        """
        ts_probabilities = self.config['time_slots']['probabilities']
        hour = random.choices(list(ts_probabilities.keys()), weights=list(ts_probabilities.values()))[0]
        minutes = random.randint(0, 59)
        start_time = datetime.timedelta(hours=hour, minutes=minutes)
        end_time = start_time + datetime.timedelta(minutes=10)
        if end_time >= datetime.timedelta(hours=24):
            # Decrease time by a full day
            end_time -= datetime.timedelta(days=1)
        time_slot_id = f'{start_time.seconds}'
        time_slot = TimeSlot(time_slot_id, start_time, end_time)
        return time_slot

    def _get_random_line(self) -> Line:
        """
        """
        probs = self.config['lines']['probabilities'].values()
        line = random.choices(list(self.lines.values()), weights=list(probs))[0]

        timetable = {}
        tt_randomizer = np.random.uniform(low=0.0, high=0.4)
        dt_randomizer = np.random.uniform(low=0.0, high=0.5)
        for i, station in enumerate(line.timetable):
            arrival, departure = line.timetable[station]

            if i == 0:
                prev_dt = departure
            else:
                ref_stop_time = departure - arrival
                travel_time = arrival - prev_dt
                arrival = float(round(prev_dt + (travel_time + travel_time * tt_randomizer)))
                departure = float(round(arrival + ref_stop_time + ref_stop_time * dt_randomizer))

            timetable[station] = (arrival, departure)

        # Encode timetable to string (Hash or something) for unique line id based on timetable
        line_id = str(hash(str(timetable.values())))
        return Line(f'Line_{line_id}', line.name, line.corridor, timetable)

    def generate(
        self,
        file_name: Path,
        path_config: Path,
        n_services: int = 1,
        n_services_by_ru: Mapping[str, int] = None,
        seed: int = None
    ) -> List[Service]:
        """
        Generate a list of services.

        If the optional parameter n_services_by_ru is provided (a mapping from RU id to number of services),
        then for each RU the specified number of services will be generated. Otherwise, n_services will be used
        as a global counter.

        Args:
            file_name (Path): Name of the output file.
            path_config (Path): Path to the config file.
            n_services (int, optional): Number of services to generate (if n_services_by_ru is not provided). Defaults to 1.
            n_services_by_ru (Mapping[str, int], optional): Mapping of RU id (TSP id) to the number of services to generate.
            seed (int, optional): Seed for the random number generator.

        Returns:
            List[Service]: List of generated Service objects.
        """
        if seed is not None:
            self.set_seed(seed)
        self._set_config(path_config)

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
        self.save_to_yaml(services, file_name)
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
