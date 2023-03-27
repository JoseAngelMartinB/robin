"""Entities for the serivce generator module."""

import ast
import datetime
import numpy as np
import os
import pandas as pd
import random
from src.robin.supply.entities import Station, Corridor, Seat, TimeSlot, TSP, Line, RollingStock, Service
from src.data_generator.utils import _get_distance
from src.scraping.utils import write_to_yaml, station_to_dict, seat_to_dict, corridor_to_dict, line_to_dict, \
    rolling_stock_to_dict, time_slot_to_dict, tsp_to_dict, service_to_dict
from src.data_generator.utils import _get_end_time, _get_start_time, _to_station, _build_service
import time
from typing import Tuple, List, Dict
import yaml


class ServiceGenerator:
    """
    Generate random services from a YAML config file

    Attributes:
        stations (Dict[str, Station]): Dict of stations
        corridors (Dict[str, Corridor]): Dict of corridors
        seats (Dict[str, Seat]): Dict of seats
        timetable (pd.DataFrame): Timetable
        rolling_stocks (Dict[str, RollingStock]): Dict of rolling stocks
        tsps (Dict[str, TSP]): Dict of TSPs
        lines (Dict[str, Line]): Dict of lines
        time_slots (Dict[str, TimeSlot]): Dict of time slots
        services (List[Service]): List of services generated in the current session

    Methods:
        generate() (List[Service])
    """

    def __init__(self):
        self.stations = self._get_stations()
        self.corridors = self._get_corridors()
        self.seats = self._get_seats()
        self.timetable = self._get_timetable()
        self.rolling_stocks = self._get_rolling_stocks()
        self.tsps = self._get_tsps()
        self.lines = {}
        self.time_slots = {}
        self.services = []

    def generate(self, file_name: str, path_config: str, n_services: int = 1, seed: int = None) -> List[Service]:
        """
        Generate a list of services

        Args:
            file_name (str): Name of the output file
            path_config (str): Path to the config file
            n_services (int, optional): Number of services to generate. Defaults to 1.
            seed (int, optional): Seed for the random number generator. Defaults to None.

        Returns:
            List[Service]: List of services
        """
        if seed is not None:
            self.set_seed(seed)
        self._set_config(path_config)

        services = []
        for _ in range(n_services):
            services.append(self._generate_service())

        self.services += services
        self.save_to_yaml(services, file_name)
        return services

    def save_to_yaml(self, services: List[Service], file_name: str) -> None:
        """
        Save the data to a yaml file

        Args:
            filename (str): Name of the file
            services (List[Service]): List of Service objects

        Returns:
            None
        """
        rolling_stocks = list(set([rs for tsp in self.tsps.values() for rs in tsp.rolling_stock]))

        yaml_dict = {'stations': [station_to_dict(stn) for stn in self.stations.values()],
                     'seat': [seat_to_dict(s) for s in self.seats.values()],
                     'corridor': [corridor_to_dict(corr) for corr in self.corridors.values()],
                     'line': [line_to_dict(ln) for ln in self.lines.values()],
                     'rollingStock': [rolling_stock_to_dict(rs) for rs in rolling_stocks],
                     'trainServiceProvider': [tsp_to_dict(tsp) for tsp in self.tsps.values()],
                     'timeSlot': [time_slot_to_dict(s) for s in self.time_slots.values()],
                     'service': [service_to_dict(serv) for serv in services]}

        print(yaml_dict['service'][0])
        self._write_to_yaml(file_name, yaml_dict)

    def _check_collisions(self, new_service, listed_service):
        """
        Check if two services collide

        Args:
            new_service (Service): New service
            listed_service (Service): Service in the list of services

        Returns:
            bool: True if collision, False otherwise
        """

        # TODO: Check other types of collisions (different train but same line, etc.)
        if new_service.rolling_stock == listed_service.rolling_stock:
            start_dt_sl = _get_start_time(listed_service)
            end_dt_sl = _get_end_time(listed_service)
            start_dt_ns = _get_start_time(new_service)
            end_dt_ns = _get_end_time(new_service)

            if start_dt_ns <= end_dt_sl and end_dt_ns >= start_dt_sl:
                return True

            ref_time_left = self._get_ref_time(new_service, listed_service)
            ref_time_right = self._get_ref_time(listed_service, new_service)

            if end_dt_ns > start_dt_sl:
                if end_dt_sl + ref_time_right > start_dt_ns:
                    return True
            else:
                if end_dt_ns + ref_time_left > start_dt_sl:
                    return True
        return False

    # Remove seed from config file
    def _generate_service(self) -> Service:
        """
        Generate a random service

        Returns:
            Service: Service object
        """
        allow_collisions = self.config['services']['allow_collisions']

        while True:
            corridor = self._get_random_corridor()
            line = self._get_random_line(corridor)
            time_slot = self._get_random_time_slot()
            tsp = self._get_random_tsp()
            rs = self._get_random_rs(tsp)
            date = self._get_random_date()
            prices = self._get_random_prices(line, rs, tsp)  # prices: Dict[Tuple[str, str], Dict[Seat, float]]
            service = _build_service(date, line, time_slot, tsp, rs, prices)

            if allow_collisions:
                self.services.append(service)
                return service

            if not any(self._check_collisions(service, s) for s in self.services):
                self.services.append(service)
                return service

    def _get_ref_time(self, service_1: Service, service_2: Service):
        """
        Get reference time between two services

        Args:
            service_1 (Service): First service
            service_2 (Service): Second service

        Returns:
            datetime.timedelta: Reference time between stations
        """
        last_stop_ns = service_1.line.stations[-1].id
        first_stop_sl = service_2.line.stations[0].id

        if last_stop_ns == first_stop_sl:
            return datetime.timedelta(0)

        if (last_stop_ns, first_stop_sl) in self.timetable.keys():
            ref_time = self.timetable[(last_stop_ns, first_stop_sl)]
        elif (first_stop_sl, last_stop_ns) in self.timetable.keys():
            ref_time = self.timetable[(first_stop_sl, last_stop_ns)]
        else:
            raise ValueError(f'No timetable entry for {(last_stop_ns, first_stop_sl)} or '
                             f'{(first_stop_sl, last_stop_ns)}')

        hours = int(ref_time / 60)
        minutes = int(ref_time % 60)
        seconds = int((ref_time - int(ref_time)) * 60)
        return datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

    def _set_config(self, path_config: str):
        """
        Set config file

        Args:
            path_config (str): Path to config file
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

    def _get_random_tsp(self):
        """
        Get a random TSP

        Returns:
            TSP: TSP object randomly selected from the available TSPs
        """
        return random.choice(list(self.tsps.values()))

    def _get_random_corridor(self) -> Corridor:
        """
        Get a random corridor

        Returns:
            Corridor: Corridor object randomly selected from the available corridors
        :return:
        """
        # corr_id = self.config.corridors.set_corridor
        corr_id = self.config['corridors']['set_corridor']

        if corr_id:
            return self.corridors[corr_id]
        return random.choice(list(self.corridors.values()))

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
            distance = _get_distance(line, origin_sta, destination_sta)
            # print(f'Distance between {origin_sta.name} and {destination_sta.name}: {distance} km')

            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = str(round(price_calc, 2)) if price_calc < max_price else max_price

        return prices

    def _get_random_time_slot(self, size: int = 10) -> TimeSlot:
        """
        Get a random time slot

        Args:
            size (int): Size of the time slot in minutes

        Returns:
            TimeSlot: Time slot object
        """
        start_time = datetime.timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))
        td_size = datetime.timedelta(minutes=size)
        end_time = start_time + td_size

        if end_time > datetime.timedelta(hours=23, minutes=59):
            hours = float(end_time.seconds // 3600)
            minutes = float((end_time.seconds // 60) % 60)
            end_time = datetime.timedelta(hours=hours, minutes=minutes)

        hours = str(start_time.seconds // 3600)
        minutes = str((start_time.seconds // 60) % 60)
        td_minutes = str(td_size.seconds // 60)
        id_ = f'{hours}:{minutes}_{td_minutes}'

        if id_ in self.time_slots:
            return self.time_slots[id_]
        else:
            time_slot = TimeSlot(id_=id_, start=start_time, end=end_time)
            self.time_slots[id_] = time_slot
            return time_slot

    def _get_random_line(self, corridor: Corridor) -> Line:
        """
        Get random line from corridor

        Args:
            corridor: Corridor object

        Returns:
            Line: Line object
        """
        path = random.choice(corridor.paths)  # path = List[Station]

        if len(path) < 2:
            raise ValueError('Path must have at least 2 stations')

        if not self.config['lines']['sample']:
            line_path = path
        else:
            max_stops = self.config['lines'].get('max_stops', None)
            min_stations = 0 if not max_stops else len(path) - max_stops
            num_delete = random.randint(min_stations, len(path) - 2)
            white_list = self.config['lines'].get('white_list', [])

            i = 0
            ids_delete = []
            while i < num_delete:
                id_delete = random.randint(0, len(path))
                if id_delete in white_list:
                    continue
                ids_delete.append(id_delete)
                i += 1

            line_path = []
            for i, sta in enumerate(path):
                if i not in ids_delete:
                    line_path.append(sta)

        line_bool_tuple = list(map(lambda s: 1 if s in line_path else 0, path))
        new_id = line_path[0].id + '_' + line_path[-1].id + '_' + ''.join(map(str, line_bool_tuple))

        if new_id in self.lines:
            line = self.lines[new_id]
        else:
            name = f'Line {line_path[0].shortname}-{line_path[-1].shortname}'

            line_ids = [sta.id for sta in line_path]
            line_times = {}
            for i, sta in enumerate(line_path):
                if i == 0:
                    line_times[sta.id] = (0.0, 0.0)
                else:
                    start_time = line_times[line_ids[i - 1]][1] + self.timetable[(line_ids[i - 1], line_ids[i])]
                    line_times[sta.id] = (round(start_time, 2),
                                          round(start_time + 3.0, 2))

            # Generate random timetable
            line = Line(id_=new_id, name=name, corridor=corridor, timetable=line_times)

        self.lines[line.id] = line
        return line

    def _get_tsps(self) -> Dict[int, TSP]:
        """
        Load TSPs

        Returns:
            Dict[int, TSP]: Dictionary of TSPs
        """
        tsps_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/tsps.csv")
        df = pd.read_csv(tsps_file, delimiter=',', dtype={'id': int, 'name': str, 'rolling_stock': str})

        def parse_rolling_stock(rolling_stock_str):
            if rolling_stock_str:
                rolling_stock_list = rolling_stock_str.split('_')
                rolling_stock = [self.rolling_stocks[int(rs_id)] for rs_id in rolling_stock_list]
                return rolling_stock
            else:
                return []

        df['rolling_stock'] = df['rolling_stock'].apply(parse_rolling_stock)

        tsp_dict = {}
        for index, row in df.iterrows():
            tsp = TSP(row['id'], row['name'], row['rolling_stock'])
            tsp_dict[row['id']] = tsp

        return tsp_dict

    def _get_rolling_stocks(self) -> Dict[int, RollingStock]:
        """
        Load timetable with reference travel times (in minutes) between each pair of stations from csv file

        Returns:
            Dict[Tuple[str, str], float]: Dict of travel times {(origin_id, destination_id): time}
        """
        rolling_stocks_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/rollingstocks.csv")
        df = pd.read_csv(rolling_stocks_file, delimiter=',', dtype={'id': int, 'name': str, 'seats': str})

        def parse_seats(seats_str):
            if seats_str:
                seats_dict = {}
                for pair in seats_str.split('_'):
                    key, value = pair.split(':')
                    seats_dict[int(key)] = int(value)
                return seats_dict
            else:
                return {}

        df['seats'] = df['seats'].apply(parse_seats)

        rolling_stock_dict = {}
        for index, row in df.iterrows():
            rolling_stock = RollingStock(row['id'], row['name'], row['seats'])
            rolling_stock_dict[row['id']] = rolling_stock

        return rolling_stock_dict

    def _get_timetable(self) -> Dict[Tuple[str, str], float]:
        """
        Load timetable with reference travel times (in minutes) between each pair of stations from csv file

        Returns:
            Dict[Tuple[str, str], float]: Dict of travel times {(origin_id, destination_id): time}
        """
        timetable_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/timetable.csv")
        df = pd.read_csv(timetable_file, delimiter=',', dtype={'origin': str, 'destination': str, 'time': float})

        timetable = {}
        for index, pair_time in df.iterrows():
            origin_id = str(pair_time['origin'])
            destination_id = str(pair_time['destination'])
            time = float(pair_time['time'])

            timetable[(origin_id, destination_id)] = time

        return timetable

    def _get_stations(self) -> Dict[str, Station]:
        """
        Load stations from csv file

        Returns:
            Dict[str, Station]: Dict of Station objects {station_id: Station object}
        """
        stations_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/stations.csv")
        df = pd.read_csv(stations_file, delimiter=',', dtype={'stop_id': str})

        stations = {}
        for index, station in df.iterrows():
            id_ = station['stop_id']
            name = station['stop_name']
            city = station['stop_name']
            shortname = str(station['stop_name'][:3]).upper()
            coords = tuple(station[['stop_lat', 'stop_lon']])

            stations[id_] = Station(id_, name, city, shortname, coords)

        return stations

    def _get_corridors(self) -> Dict[int, Corridor]:
        """
        Load corridors from csv file

        Args:
            stations (Dict[str, Station]): Dict of Station objects {station_id: Station object}

        Returns:
            Dict[int, Corridor]: Dict of Corridor objects {corridor_id: Corridor object}
        """
        corridors_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/corridors.csv")
        df = pd.read_csv(corridors_file, delimiter=',', dtype={'corridor_id': int})

        corridors = {}
        for index, corr in df.iterrows():
            id_ = corr['corridor_id']
            name = corr['corridor_name']
            tree = ast.literal_eval(corr['tree'])
            stations_tree = _to_station(tree, self.stations)

            corridors[id_] = Corridor(id_, name, stations_tree)

        return corridors

    def _get_seats(self) -> Dict[int, Seat]:
        """
        Load seat types from csv file

        Returns:
            Dict[int, Seat]: Dict of Seat objects {seat_id: Seat object}
        """
        seats_file = os.path.join(os.path.dirname(__file__), "../../data/datagen/seats.csv")
        df = pd.read_csv(seats_file, delimiter=',', dtype={'id': int, 'hard_type': int, 'soft_type': int})

        seats = {}
        for index, s in df.iterrows():
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), "Incomplete Seat data"

            seats[s['id']] = Seat(s['id'], s['name'], s['hard_type'], s['soft_type'])

        return seats

    @staticmethod
    def _write_to_yaml(filename, objects):
        """
        Write objects to yaml file

        Args:
            filename (str): Name of the file
            objects (Dict): Dict of objects to write

        Returns:
            None
        """

        if not os.path.exists(filename):
            with open(filename, "w"):
                pass

        with open(filename, 'a') as f:
            yaml.safe_dump(objects, f, sort_keys=False, allow_unicode=True)

    @staticmethod
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
    config_path = 'config.yml'
    init_time = time.time()
    r = ServiceGenerator()
    services = r.generate(file_name="../../data/test.yml", path_config=config_path, n_services=200, seed=1)
    print(services[0])
    end_time = time.time()
    print(f'Time elapsed: {end_time - init_time} seconds')
