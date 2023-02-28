from src.robin.supply.entities import Station, Corridor, Seat, TimeSlot, TSP, Line, RollingStock, Service
from src.data_generator.yaml_utils import *
from math import sin, cos, acos, radians
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import datetime
import random
import time
import yaml
import ast
import os


class ServiceGenerator:
    """
    Generate random services from a YAML config file

    Attributes:
        stations (Dict[str, Station]): Dict of stations
        corridors (Dict[str, Corridor]): Dict of corridors
        seats (Dict[str, Seat]): Dict of seats
        timetable (pd.DataFrame): Timetable
        tsps (Dict[str, TSP]): Dict of TSPs
        lines (Dict[str, Line]): Dict of lines
        time_slots (Dict[str, TimeSlot]): Dict of time slots

    Methods:
        generate_service() -> Service: Generate a random service
    """

    def __init__(self):
        self.stations = self._get_stations()  # Dict[str, Station]
        self.corridors = self._get_corridors()
        self.seats = self._get_seats()
        self.timetable = self._get_timetable()
        self.tsps = self._get_tsps()
        self.lines = {}  # Dict[str, Line]
        self.time_slots = {}  # Dict[str, TimeSlot]
        self.services = []

    def _set_config(self, path_config: str):
        """
        Set config file

        Args:
            path_config (str): Path to config file
        """
        with open(path_config, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

    def generate(self, file_name: str, path_config: str, n_services: int = 1, seed: int = None) -> List[Service]:
        if seed is not None:
            self.set_seed(seed)
        self._set_config(path_config)

        services = []
        for _ in range(n_services):
            services.append(self._generate_service())

        self.services += services
        self.save_to_yaml(services, file_name)
        return services

    # Remove seed from config file
    def _generate_service(self) -> Service:
        """
        Generate a random service

        Returns:
            Service: Service object
        """
        allow_collisions = self.config['services']['allow_collisions']

        def _get_start_time(s):
            return datetime.datetime.combine(s.date, datetime.datetime.min.time()) + s.time_slot.start

        def _get_end_time(s):
            end_td = s.schedule[-1][-1]
            return datetime.datetime.combine(s.date, datetime.datetime.min.time()) + s.time_slot.start + end_td

        def _check_collisions(ns, sl):
            if ns.rolling_stock == sl.rolling_stock:

                start_dt_sl = _get_start_time(sl)
                end_dt_sl = _get_end_time(sl)
                start_dt_ns = _get_start_time(ns)
                end_dt_ns = _get_end_time(ns)

                if start_dt_ns <= end_dt_sl or end_dt_ns >= start_dt_sl:
                    return True

                last_stop_ns = ns.stops[-1].id
                first_stop_sl = sl.stops[0].id

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
                ref_time = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

                if end_dt_ns + ref_time >= start_dt_sl or start_dt_ns <= end_dt_sl + ref_time:
                    return True
            return False

        while True:
            corridor = self._get_random_corridor()
            line = self._get_random_line(corridor)
            time_slot = self._get_random_time_slot()
            tsp = self._get_random_tsp()
            rs = self._get_random_rs(tsp)
            date = self._get_random_date()
            prices = self._get_random_prices(line, rs, tsp)  # prices: Dict[Tuple[str, str], Dict[Seat, float]]
            service = self._create_service(date, line, time_slot, tsp, rs, prices)

            if allow_collisions:
                self.services.append(service)
                return service

            if any(list(filter(lambda s: _check_collisions(service, s), self.services))):
                continue
            self.services.append(service)
            return service

    def _create_service(self, date, line, time_slot, tsp, rs, prices):
        return Service(id_=f'{line.id}_{time_slot.id}',
                       date=str(date),
                       line=line,
                       time_slot=time_slot,
                       tsp=tsp,
                       rolling_stock=rs,
                       prices=prices)

    def _get_random_rs(self, tsp):
        return random.choice(tsp.rolling_stock)

    def _get_random_tsp(self):
        return random.choice(list(self.tsps.values()))

    def _get_random_corridor(self) -> Corridor:
        # corr_id = self.config.corridors.set_corridor
        corr_id = self.config['corridors']['set_corridor']

        if corr_id:
            return self.corridors[corr_id]
        return random.choice(list(self.corridors.values()))

    def _get_random_date(self):
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

    def _get_distance(self, line: Line, origin: Station, destination: Station) -> float:
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

    def _get_random_prices(self, line: Line, rolling_stock: RollingStock, tsp: TSP):
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
            distance = self._get_distance(line, origin_sta, destination_sta)
            # print(f'Distance between {origin_sta.name} and {destination_sta.name}: {distance} km')

            prices[pair] = {}
            for seat in seats:
                price_calc = (base_price + distance * distance_factor) * seat_factor[seat.id] * tsp_factor[tsp.id]
                prices[pair][seat] = round(price_calc, 2) if price_calc < max_price else max_price

        return prices

    def _get_random_time_slot(self, size: int = 10) -> TimeSlot:
        # Generate random time slot
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
        tsps = {}
        ave_rs = [RollingStock(id_="11", name='AVE RollingStock 1', seats={1: 200, 2: 50}),
                  RollingStock(id_="12", name='AVE RollingStock 1', seats={1: 250, 2: 50}),
                  RollingStock(id_="13", name='AVE RollingStock 1', seats={1: 300, 2: 50})]

        tsps[1] = TSP(id_=1, name='Renfe AVE', rolling_stock=ave_rs)

        avlo_rs = [RollingStock(id_="21", name='AVLO RollingStock 1', seats={1: 300}),
                   RollingStock(id_="22", name='AVLO RollingStock 1', seats={1: 350})]

        tsps[2] = TSP(id_=2, name='Renfe AVLO', rolling_stock=avlo_rs)

        ouigo_rs = [RollingStock(id_="31", name='OUIGO RollingStock 1', seats={1: 300, 2: 50}),
                    RollingStock(id_="32", name='OUIGO RollingStock 1', seats={2: 450})]

        tsps[3] = TSP(id_=3, name='OUIGO', rolling_stock=ouigo_rs)

        iryo_rs = [RollingStock(id_="41", name='OUIGO RollingStock 1', seats={1: 300, 2: 50}),
                   RollingStock(id_="42", name='OUIGO RollingStock 1', seats={1: 350, 2: 50})]

        tsps[5] = TSP(id_=4, name='IRYO', rolling_stock=iryo_rs)

        return tsps

    def _get_timetable(self) -> Dict[Tuple[str, str], float]:
        """
        Load timetable with reference travel times (in minutes) between each pair of stations from csv file

        Returns:
            Dict[Tuple[str, str], float]: Dict of travel times {(origin_id, destination_id): time}
        """
        timetable_file = os.path.join(os.path.dirname(__file__), "timetable.csv")
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
        stations_file = os.path.join(os.path.dirname(__file__), "stations.csv")
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
        corridors_file = os.path.join(os.path.dirname(__file__), "corridors.csv")
        df = pd.read_csv(corridors_file, delimiter=',', dtype={'corridor_id': int})

        corridors = {}
        for index, corr in df.iterrows():
            id_ = corr['corridor_id']
            name = corr['corridor_name']
            tree = ast.literal_eval(corr['tree'])

            def to_station(tree: Dict, sta_dict: Dict[str, Station]) -> Dict:
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

                return {sta_dict[node]: to_station(tree[node], sta_dict) for node in tree}

            def tree_to_yaml(dict_tree: Dict[Station, Dict]) -> List[Dict]:
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
                    return [{'org': node, 'des': tree_to_yaml(dict_tree[node])} for node in dict_tree]

            stations_tree = to_station(tree, self.stations)

            corridors[id_] = Corridor(id_, name, stations_tree)

        return corridors

    def _get_seats(self) -> Dict[int, Seat]:
        """
        Load seat types from csv file

        Returns:
            Dict[int, Seat]: Dict of Seat objects {seat_id: Seat object}
        """
        seats_file = os.path.join(os.path.dirname(__file__), "seats.csv")
        df = pd.read_csv(seats_file, delimiter=',', dtype={'id': int, 'hard_type': int, 'soft_type': int})

        seats = {}
        for index, s in df.iterrows():
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), "Incomplete Seat data"

            seats[s['id']] = Seat(s['id'], s['name'], s['hard_type'], s['soft_type'])

        return seats

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

        self._write_to_yaml(file_name, yaml_dict)

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

    services = r.generate(file_name="../../data/test.yml", path_config=config_path, n_services=8, seed=1)

    print(services[0])

    end_time = time.time()

    print(f'Time elapsed: {end_time - init_time} seconds')

    print("Finish")
