"""Entities for the supply module."""

from src.robin.supply.utils import get_time, get_date, format_td
from copy import deepcopy

from typing import List, Tuple, Dict, Union
import datetime
import yaml


class Station(object):
    """
    Station: Railway facility where trains stop to load or unload passengers, freight or both.

    Attributes:
        id (str): Station ID
        name (str): Station name
        shortname (str): Station short name
        coords (Tuple[float, float]): Station coordinates (latitude, longitude)
    """

    def __init__(self, id_: str, name: str, city: str, shortname: str, coords: Tuple[float, float] = None):
        self.id = id_
        self.name = name
        self.city = city
        self.shortname = shortname
        self.coords = coords

    def add_coords(self, coords: Tuple[float, float]) -> None:
        """
        Add coordinates to a Station object

        Args:
            coords: Tuple of floats (latitude, longitude)
        """
        self.coords = coords

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.shortname}, {self.coords}]'


class TimeSlot(object):
    """
    TimeSlot: Discrete time interval

    Attributes:
        id (int): Time slot ID
        start (datetime.timedelta): Time slot start time
        end (datetime.timedelta): Time slot end time
        class_mark (datetime.timedelta): Time slot class mark
        size (datetime.timedelta): Time slot size
    """

    def __init__(self, id_: Union[int, str], start: datetime.timedelta, end: datetime.timedelta):
        self.id = id_
        self.start = start
        self.end = end
        self.class_mark = self._get_class_mark()
        self.size = self._get_size()

    def _get_class_mark(self) -> datetime.timedelta:
        """
        Get class mark of time slot

        Returns:
            class mark (datetime.timedelta): Time slot class mark
        """

        if self.end < self.start:
            return (self.start + self.end + datetime.timedelta(days=1)) / 2 - datetime.timedelta(days=1)
        return (self.start + self.end) / 2

    def _get_size(self) -> datetime.timedelta:
        """
        Get size of time slot

        Returns:
            size (datetime.timedelta): Time slot size
        """
        if self.end < self.start:
            return (self.end + datetime.timedelta(days=1)) - self.start
        return self.end - self.start

    def __str__(self):
        return f'[{self.id}, {self.start}, {self.end}, {self.class_mark}, {self.size}]'


class Corridor(object):
    """
    Corridor: Tree of stations.

    Attributes:
        id (int): Corridor ID
        name (str): Corridor name
        tree (Dict[Station, Dict]): Tree of stations (with Station objects)
        paths (List[List[Station]]): List of paths (list of stations)
        stations (Dict[str, Station]): Dictionary of stations (with Station IDs as keys)

    tree dummy example*: [{'org': 'MAD', 'des': [{'org': 'BAR', 'des': []}, {'org': 'BIL', 'des': []}]}]
    *In the real tree, the Station objects are used instead of the Station IDs
    """

    def __init__(self, id_: Union[int, str], name: str, tree: Dict[Station, Dict]):
        self.id = id_
        self.name = name
        self.tree = tree
        self.paths = self._get_paths(self.tree)
        self.stations = self._dict_stations(self.tree)

    def _get_paths(self,
                   tree: Dict[Station, Dict],
                   path: List[Station] = None,
                   paths: List[List[Station]] = None) -> List[List[Station]]:
        """
        Get all paths from a tree of stations

        Args:
            tree: Tree of stations
            path (List[Station]): Current path (list of stations)
            paths (List[List[Station]]): List of paths (list of stations)

        Returns:
            paths (List[List[Station]]): List of paths (list of stations)
        """
        if path is None:
            path = []

        if paths is None:
            paths = []

        if not tree:
            paths.append(path.copy())
            return paths

        for node in tree:
            org = node
            path.append(org)
            self._get_paths(tree[node], path, paths)
            path.pop()

        return paths

    def _dict_stations(self, tree: Dict[Station, Dict], sta=None) -> Dict[str, Station]:
        """
        Get dictionary of stations (with Station IDs as keys)

        Args:
            tree (List[Dict]): Tree of stations

        Returns:
            sta (Dict[str, Station]): Dictionary of stations, with Station IDs as keys, and Station objects as values
        """
        if sta is None:
            sta = {}

        for node in tree:
            org = node
            sta[org.id] = org
            self._dict_stations(tree[node], sta)

        return sta

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.stations}]'


class Line(object):
    """
    Line: Sequence of stations being served by a train with a timetable.

    Attributes:
        id (int): Line ID
        name (str): Line name
        corridor (Corridor): Corridor ID where the Line belongs to
        timetable (Dict[str, Tuple[float, float]]): {station ID: (arrival (float), departure (float)}
        stations (List[Station]): List of Stations being served by the Line
        pairs (Dict[Tuple[str, str], Tuple[Station, Station]]): Dict with pairs of stations (origin, destination) \
        with (origin ID, destination ID) as keys, and (origin Station, destination Station) as values
    """

    def __init__(self, id_: Union[int, str], name: str, corridor: Corridor, timetable: Dict[str, Tuple[float, float]]):
        self.id = id_
        self.name = name
        self.corridor = corridor
        self.timetable = timetable
        self.stations = list(map(lambda sid: self.corridor.stations[sid], list(self.timetable.keys())))
        self.pairs = self._get_pairs()

    def _get_pairs(self):
        """
        Private method to get each pair of stations of the Line, using the station list

        Returns:
            List of tuple pairs (Station, Station): (origin, destination)
        """
        return {(a.id, b.id): (a, b) for i, a in enumerate(self.stations) for b in self.stations[i + 1:]}

    def __str__(self):
        return f'[{self.id}, {self.name}, Corridor id: {self.corridor}, {self.timetable}]'


class Seat(object):
    """
    Seat: Seat type of train.

    Attributes:
        id (int): Seat ID
        name (str): Seat type name
        hard_type (int): Hard seat type
        soft_type (int): Soft seat type
    """

    def __init__(self, id_: Union[int, str], name: str, hard_type: int, soft_type: int):
        self.id = id_
        self.name = name
        self.hard_type = hard_type
        self.soft_type = soft_type

    def __repr__(self):
        return f'{self.id}'

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.hard_type}, {self.soft_type}]'


class RollingStock(object):
    """
    Rolling Stock: Locomotives, Carriages, Wagons, or other vehicles used on a railway.

    Attributes:
        id (int): Rolling Stock ID
        name (str): Rolling Stock name
        seats (Dict[int, int]): Number of seats for each hard_type {hard_type1: quantity1, hard_type2: quantity2}
    """

    def __init__(self, id_: Union[int, str], name: str, seats: Dict[int, int]):
        self.id = id_
        self.name = name
        self.seats = seats
        self.total_capacity = sum(seats.values())

    def __str__(self):
        return f'[{self.id},{self.name},{self.seats}]'


class TSP(object):
    """
    TSP: Train Service Provider. Company that provides train services.

    Attributes:
        id (int): Train Service Provider ID
        name (name): Name of the Train Service Provider
        rolling_stock List[RollingStock]: List of RollingStock objects
    """

    def __init__(self, id_: Union[int, str], name: str, rolling_stock: List[RollingStock] = None):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock if rolling_stock is not None else []

    def add_rolling_stock(self, rs: RollingStock):
        """
        Method to add new Rolling stock object to a TSP object

        Args:
            rs (RollingStock): New Rolling Stock
        """
        self.rolling_stock.append(rs)

    def __str__(self):
        return f'[{self.id}, {self.name}, {[rs.id for rs in self.rolling_stock]}]'


class Service(object):
    """
    Service: Travel options provided by a TSP between stations in a Line with a timetable.
    Capacity is defined by the Rolling Stock. Capacity constraints may apply between some pairs of stations.
    Different type of seats may be offered, each with a specific price.

    Attributes:
        id (str): Service ID
        date (datetime.date): Day of service (year, month, daty, without time)
        line (Line): Line in which the service is provided
        tsp (TSP): Train Service Provider which provides the service
        time_slot (TimeSlot): Time Slot. Defines the start time of the service
        schedule (List[Tuple[datetime.timedelta, datetime.timedelta]]): List of tuples with arrival-departure times
        service_departure_time (float): Service departure time in hours
        service_arrival_time (float): Service arrival time in hours
        rolling_stock (RollingStock): Rolling Stock used in the service
        prices (Dict[Tuple[str, str], Dict[Seat, float]]): Prices for each pair of stations and each Seat type
        capacity_constraints (Dict[Tuple[str, str], Dict[int, int]]): Constrained capacity (limit seats available
        between a specific pair of stations)
        lift_constraints (int): Minimum anticipation (days) to lift capacity constraints
    """

    def __init__(self,
                 id_: Union[int, str],
                 date: datetime.date,
                 line: Line,
                 tsp: TSP,
                 time_slot: TimeSlot,
                 rolling_stock: RollingStock,
                 prices: Dict[Tuple[str, str], Dict[Seat, float]],
                 capacity_constraints: Dict[Tuple[str, str], Dict[int, int]] = None,
                 lift_constraints: int = 1):

        self.id = id_
        self.date = date
        self.line = line
        self.tsp = tsp
        self.time_slot = time_slot
        self.schedule = self._get_absolute_schedule()
        self.service_departure_time = self.schedule[0][0].seconds / 3600  # Service departure time in hours
        self.service_arrival_time = self.schedule[-1][0].seconds / 3600  # Service arrival time in hours
        self.rolling_stock = rolling_stock
        self.capacity_constraints = capacity_constraints
        self.lift_constraints = lift_constraints
        self.prices = prices

        _seats = set([s for d in self.prices.values() for s in d.keys()])

        self._capacity_log = {p: {ht: 0 for ht in self.rolling_stock.seats.keys()} for p in self.line.pairs}
        self.seats_log = {p: {s: 0 for s in _seats} for p in self.line.pairs}
        self.hardtype_log = self._get_hardtype_log()
        self.seats_reduce_log = {s: 0 for s in _seats}
        self.rs_reduce_log = {ht: 0 for ht in self.rolling_stock.seats.keys()}

    def _get_hardtype_log(self) -> Dict[Tuple[str, str], Dict[int, int]]:
        """
        Private method to get the hard type log of the service

        Args:
            self (Service): Service object

        Returns:
            Dict[Tuple[str, str], Dict[int, int]]: Hard type log of the service
        """
        hardtype_log = {}

        for p in self.seats_log:
            hardtype_log[p] = {}
            for s in self.seats_log[p].keys():
                if s.hard_type not in hardtype_log[p]:
                    hardtype_log[p][s.hard_type] = 0
                else:
                    hardtype_log[p][s.hard_type] += self.seats_log[p][s]

        return hardtype_log

    def _get_absolute_schedule(self) -> List[Tuple[datetime.timedelta, datetime.timedelta]]:
        """
        Private method to get the absolute schedule of the service, using the relative schedule and the time slot
        start time

        Args:
            self (Service): Service object

        Returns:
            List of tuples (datetime.timedelta, datetime.timedelta): [(departure time, arrival time)]
        """
        absolute_schedule = []
        for dt, at in list(self.line.timetable.values()):
            abs_dt = datetime.timedelta(seconds=dt*60) + self.time_slot.start
            abs_at = datetime.timedelta(seconds=at*60) + self.time_slot.start
            absolute_schedule.append((abs_dt, abs_at))

        return absolute_schedule

    def buy_ticket(self, origin: str, destination: str, seat: Seat, anticipation: int) -> bool:
        """
        Method to buy a ticket for a service

        Args:
            self (Service): Service object
            origin (str): Origin station ID
            destination (str): Destination station ID
            seat (Seat): Seat type
            anticipation (int): Days of anticipation in the purchase of the ticket

        Returns:
            True if the ticket was bought, False if not
        """
        stations_ids = list(self.line.timetable.keys())

        service_route = set(range(stations_ids.index(origin), stations_ids.index(destination)))

        if not self.tickets_available(origin, destination, seat, anticipation):
            return False

        for pair in self.line.pairs:  # pairs attribute is a dictionary with all the pairs of stations
            origin_id, destination_id = pair
            stations_in_pair = set(range(stations_ids.index(origin_id), stations_ids.index(destination_id)))
            if service_route.intersection(stations_in_pair):
                if self._capacity_log[pair][seat.hard_type] < self.rolling_stock.seats[seat.hard_type]:
                    self._capacity_log[pair][seat.hard_type] += 1

        self.seats_log[(origin, destination)][seat] += 1
        self.seats_reduce_log[seat] += 1
        self.rs_reduce_log[seat.hard_type] += 1

        return True

    def tickets_available(self, origin: str, destination: str, seat: Seat, anticipation: int) -> bool:
        """
        Method to check the availability of seats in a service

        Args:
            self (Service): Service object
            origin (str): Origin station ID
            destination (str): Destination station ID
            seat (Seat): Seat type
            anticipation (int): Days of anticipation in the purchase of the ticket

        Returns:
            bool: True if available, False if not available
        """
        occupied_seats = self._capacity_log[(origin, destination)][seat.hard_type]

        if self.capacity_constraints and anticipation <= self.lift_constraints:
            if (origin, destination) in self.capacity_constraints:
                constrained_capacity = self.capacity_constraints[(origin, destination)][seat.hard_type]
                if occupied_seats < constrained_capacity:
                    return True
        else:
            max_capacity = self.rolling_stock.seats[seat.hard_type]
            if occupied_seats < max_capacity:
                return True

        return False

    def __str__(self):
        new_line = "\n\t\t"
        return f'Service id: {self.id} \n' \
               f'\tDate of service: {self.date} \n' \
               f'\tStops: {[sta.id for sta in self.line.stations]} \n' \
               f'\tLine times (relative): {list(self.line.timetable.values())} \n' \
               f'\tLine times (absolute): {[(format_td(at), format_td(dt)) for at, dt in self.schedule]} \n' \
               f'\tTrain Service Provider: {self.tsp} \n' \
               f'\tTime Slot: {self.time_slot} \n' \
               f'\tRolling Stock: {self.rolling_stock} \n' \
               f'\tPrices: \n' \
               f'\t\t{new_line.join(f"{key}: {value}" for key, value in self.prices.items())} \n' \
               f'\tTickets sold per each pair (hard type): {self.hardtype_log} \n' \
               f'\tTickets sold per each pair (seats): {self.seats_log} \n' \
               f'\tTickets sold (count, seats): {self.seats_reduce_log} \n' \
               f'\tTickets sold (count, hard type): {self.rs_reduce_log} \n' \
               f'\tCapacity constraints: {self.capacity_constraints} \n'


class Supply(object):
    """
    Supply: List of Service's available in the system

    Attributes:
        services List[Service]: List of services available in the system
    """

    def __init__(self, services: List[Service]):
        self.services = services

    @classmethod
    def from_yaml(cls, path: str):
        """
        Class method to create a Supply object (List[Service]) from a yaml file

        Args:
            path (str): Path to the yaml file

        Returns:
            Supply object List[Service]
        """
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        stations = Supply._get_stations(data, key='stations')
        time_slots = Supply._get_time_slots(data, key='timeSlot')
        corridors = Supply._get_corridors(data, stations, key='corridor')
        lines = Supply._get_lines(data, corridors, key='line')
        seats = Supply._get_seats(data, key='seat')
        rolling_stock = Supply._get_rolling_stock(data, seats, key='rollingStock')
        tsps = Supply._get_tsps(data, rolling_stock, key='trainServiceProvider')
        services = Supply._get_services(data, lines, tsps, time_slots, seats, rolling_stock, key='service')

        return cls(list(services.values()))

    def filter_service_by_id(self, service_id: str) -> Service:
        """
        Filters a Service by ID

        Args:
            service_id (str): Service ID

        Returns:
            Service: Service object
        """
        for s in self.services:
            if s.id == service_id:
                return s

    def filter_services(self, origin: str, destination: str, date: datetime.date) -> List[Service]:
        """
        Filters a List of Services available in the system that meet the users requirements:
        From origin station ID, to destination station ID, on date D

        Args:
            origin (str): Origin Station ID
            destination (str): Destination Station ID
            date (datetime.date): Date of service (day, month, year, without time)

        Returns:
            List[Service]: List of Service objects that meet the user requests
        """
        filtered_services = []

        for s in self.services:
            if s.date == date and (origin, destination) in s.line.pairs.keys():
                filtered_services.append(s)

        return filtered_services

    @classmethod
    def _get_stations(cls, data, key='stations') -> Dict[str, Station]:
        """
        Private method to build a dict of Station objects from YAML data

        Args:
            data: YAML data as nested dict
            key (str): Key to access the data in the YAML file. Default: 'stations'.

        Returns:
            Dict[str, Station]: Dict of Station objects {station_id: Station object}
        """
        stations = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'short_name', 'city')), "Incomplete Station data"

            coords = tuple(s.get('coordinates', {'lat': None, 'lon': None}).values())

            stations[s['id']] = Station(s['id'], s['name'], s['city'], s['short_name'], coords)
        return stations

    @classmethod
    def _get_time_slots(cls, data, key='timeSlot') -> Dict[str, TimeSlot]:
        """
        Private method to build a dict of TimeSlot objects from YAML data

        Args:
            data: YAML data as nested dict
            key (str): Key to access the data in the YAML file. Default: 'timeSlot'.

        Returns:
            Dict[str, TimeSlot]: Dict of TimeSlot objects {time_slot_id: TimeSlot object}
        """
        time_slots = {}
        for ts in data[key]:
            assert all(k in ts.keys() for k in ('id', 'start', 'end')), "Incomplete TimeSlot data"

            time_slots[ts['id']] = TimeSlot(ts['id'], get_time(ts['start']), get_time(ts['end']))
        return time_slots

    @classmethod
    def _get_corridors(cls, data, stations, key='corridor') -> Dict[str, Corridor]:
        """
        Private method to build a dict of Corridor objects from YAML data

        Args:
            data: YAML data as nested dict
            stations (Dict[str, Station]): Dict of Station objects {station_id: Station object}
            key (str): Key to access the data in the YAML file. Default: 'corridor'.

        Returns:
            Dict[str, Corridor]: Dict of Corridor objects {corridor_id: Corridor object}
        """

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

        def set_stations_ids(tree, sta_set=None):
            """
            Recursive function to build a set of station IDs from a tree of station IDs

            Args:
                tree (Mapping): Tree of station IDs
                sta_set (set): Set of station IDs. Default: None

            Returns:
                Set of station IDs
            """
            if sta_set is None:
                sta_set = set()

            if not tree:
                return

            for node in tree:
                sta_set.add(node)
                set_stations_ids(tree[node], sta_set)

            return sta_set

        def convert_tree_to_dict(tree):
            if len(tree) == 1:
                node = tree[0]
                return {node['org']: convert_tree_to_dict(node['des'])}
            else:
                return {node['org']: convert_tree_to_dict(node['des']) for node in tree}

        corridors = {}
        for c in data[key]:
            assert all(k in c.keys() for k in ('id', 'name', 'stations')), "Incomplete Corridor data"

            tree_dictionary = convert_tree_to_dict(c['stations'])

            corr_stations_ids = set_stations_ids(tree_dictionary)

            assert all(s in stations.keys() for s in corr_stations_ids), "Station not found in Station list"

            stations_tree = to_station(deepcopy(tree_dictionary), stations)

            corridors[c['id']] = Corridor(c['id'], c['name'], stations_tree)

        return corridors

    @classmethod
    def _get_lines(cls, data, corridors, key='line'):
        """
        Private method to build a dict of Line objects from YAML data

        Args:
            data: YAML data
            corridors: Dict of Corridor objects {corridor_id: Corridor object}
            key (str): Key to access the data in the YAML file. Default: 'line'.

        Returns:
            Dict[str, Line]: Dict of Line objects {line_id: Line object}
        """
        lines = {}
        for ln in data[key]:
            assert all(k in ln.keys() for k in ('id', 'name', 'corridor', 'stops')), "Incomplete Line data"

            assert ln['corridor'] in corridors.keys(), "Corridor not found in Corridor list"
            corr = corridors[ln['corridor']]

            for stn in ln['stops']:
                assert all(k in stn for k in ('station', 'arrival_time', 'departure_time')), "Incomplete Stops data"

            corr_stations_ids = [s.id for s in corr.stations.values()]
            assert all(s['station'] in corr_stations_ids for s in ln['stops']), "Station not found in Corridor list"

            timetable = {s['station']: (float(s['arrival_time']), float(s['departure_time']))
                         for s in ln['stops']}

            lines[ln['id']] = Line(ln['id'], ln['name'], corr, timetable)

        return lines

    @classmethod
    def _get_seats(cls, data, key='seat'):
        """
        Private method to build a dict of Seat objects from YAML data

        Args:
            data: YAML data
            key (str): Key to access the data in the YAML file. Default: 'seat'.

        Returns:
            Dict[str, Seat]: Dict of Seat objects {seat_id: Seat object}
        """
        seats = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), "Incomplete Seat data"

            seats[s['id']] = Seat(s['id'], s['name'], s['hard_type'], s['soft_type'])

        return seats

    @classmethod
    def _get_rolling_stock(cls, data, seats, key='rollingStock'):
        """
        Private method to build a dict of RollingStock objects from YAML data

        Args:
            data: YAML data
            seats (Dict[str, Seat]): Dict of Seat objects {seat_id: Seat object}
            key (str): Key to access the data in the YAML file. Default: 'rollingStock'.

        Returns:
            Dict[str, RollingStock]: Dict of RollingStock objects {rolling_stock_id: RollingStock object}
        """
        rolling_stock = {}
        for rs in data[key]:
            assert all(k in rs.keys() for k in ('id', 'name', 'seats')), "Incomplete RollingStock data"

            for st in rs['seats']:
                assert all(k in st for k in ('hard_type', 'quantity')), "Incomplete seats data for RS"

            assert all(s['hard_type'] in [s.hard_type for s in seats.values()] for s in rs['seats']), \
                "Invalid hard_type for RS"

            rs_seats = {s['hard_type']: s['quantity'] for s in rs['seats']}

            rolling_stock[rs['id']] = RollingStock(rs['id'],
                                                   rs['name'],
                                                   rs_seats)

        return rolling_stock

    @classmethod
    def _get_tsps(cls, data, rolling_stock, key='trainServiceProvider'):
        """
        Private method to build a dict of TSP objects from YAML data

        Args:
            data: YAML data
            rolling_stock: Dict of RollingStock objects {rolling_stock_id: RollingStock object}
            key (str): Key to access the data in the YAML file. Default: 'trainServiceProvider'.

        Returns:
            Dict[str, TSP]: Dict of TSP objects {tsp_id: TSP object}
        """
        tsp = {}
        for op in data[key]:
            assert all(k in op.keys() for k in ('id', 'name', 'rolling_stock')), "Incomplete TSP data"
            assert all(i in rolling_stock.keys() for i in op['rolling_stock']), "Unknown RollingStock ID"

            tsp[op['id']] = TSP(op['id'], op['name'], [rolling_stock[i] for i in op['rolling_stock']])

        return tsp

    @classmethod
    def _get_services(cls, data, lines, tsps, time_slots, seats, rolling_stock, key='service'):
        """
        Private method to build a dict of Service objects from YAML data

        Args:
            data: YAML data
            lines: Dict of Line objects {line_id: Line object}
            tsps: Dict of TSP objects {tsp_id: TSP object}
            time_slots: Dict of TimeSlot objects {time_slot_id: TimeSlot object}
            rolling_stock: Dict of RollingStock objects {rolling_stock_id: RollingStock object}
            key (str): Key to access the data in the YAML file. Default: 'service'.

        Returns:
            Dict[str, Service]: Dict of Service objects {service_id: Service object}
        """
        services = {}
        for s in data[key]:
            service_keys = ('id', 'date', 'line', 'train_service_provider', 'time_slot', 'rolling_stock',
                            'origin_destination_tuples', 'capacity_constraints')

            assert all(k in s.keys() for k in service_keys), "Incomplete Service data"

            service_id = s['id']
            service_date = get_date(s['date'])

            assert s['line'] in lines.keys(), "Line not found"
            service_line = lines[s['line']]

            assert s['train_service_provider'] in tsps.keys(), "TSP not found"
            service_tsp = tsps[s['train_service_provider']]

            assert s['time_slot'] in time_slots.keys(), "TimeSlot not found"
            service_time_slot = time_slots[s['time_slot']]

            assert s['rolling_stock'] in rolling_stock.keys(), "RollingStock not found"
            service_rs = rolling_stock[s['rolling_stock']]

            service_prices = {}
            for od in s['origin_destination_tuples']:
                assert all(k in od.keys() for k in ('origin', 'destination', 'seats')), "Incomplete Service prices"

                org = od['origin']
                des = od['destination']
                assert all(s in service_line.corridor.stations.keys() for s in (org, des)), "Invalid station in Service"

                for st in od['seats']:
                    assert all(k in st for k in ('seat', 'price')), "Incomplete seats data for Service"
                    assert st['seat'] in seats, "Invalid seat in Service prices"

                prices = {seats[st['seat']]: st['price'] for st in od['seats']}

                service_prices[(org, des)] = prices

            capacity_constraints = s['capacity_constraints']

            if capacity_constraints:
                cc_constraints = {}
                for cc in capacity_constraints:
                    assert all(k in cc for k in ('origin', 'destination', 'seats')), \
                        "Incomplete capacity constraints data for Service"

                    assert all(s in service_line.corridor.stations.keys() for s in (cc['origin'], cc['destination'])), \
                        "Invalid station in capacity constraints"

                    for st in cc['seats']:
                        assert all(k in st for k in ('hard_type', 'quantity')), "Incomplete seats data for Service"
                        assert st['hard_type'] in service_rs.seats.keys(), "Invalid hard type in capacity constraints"

                    cc_constraints[(cc['origin'], cc['destination'])] = {st['hard_type']: st['quantity']
                                                                         for st in cc['seats']}
            else:
                cc_constraints = None

            services[service_id] = Service(service_id,
                                           service_date,
                                           service_line,
                                           service_tsp,
                                           service_time_slot,
                                           service_rs,
                                           service_prices,
                                           cc_constraints)

        return services
