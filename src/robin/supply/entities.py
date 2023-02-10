"""Entities for the supply module."""

from src.robin.supply.utils import get_time, get_date, format_td
from copy import deepcopy

from typing import List, Tuple, Mapping
import datetime
import yaml


class Station(object):
    """
    Station: Railway facility where trains stop to load or unload passengers, freight or both.

    Attributes:
        id (int): Station ID
        name (str): Station name
        shortname (str): Station short name
        coords (Tuple[float, float]): Station coordinates (latitude, longitude)
    """

    def __init__(self, id_: int, name: str, city: str, shortname: str, coords: Tuple[float, float] = None):
        self.id = id_
        self.name = name
        self.city = city
        self.shortname = shortname
        self.coords = coords

    def add_coords(self, coords: Tuple[float, float]):
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

    def __init__(self, id_: int, start: str, end: str):
        self.id = id_
        self.start = get_time(start)
        self.end = get_time(end)
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
        tree (List[Mapping]): Tree of stations (with Station objects)
        paths (List[List[Station]]): List of paths (list of stations)
        stations (Dict[str, Station]): Dictionary of stations (with Station IDs as keys)
    """

    def __init__(self, id_: int, name: str, tree: List[Mapping]):
        self.id = id_
        self.name = name
        self.tree = tree
        self.paths = self._get_paths(self.tree)
        self.stations = self._dict_stations(self.tree)

    def _get_paths(self, tree, path=None, paths=None):
        if path is None:
            path = []

        if paths is None:
            paths = []

        if not tree:
            paths.append(path)
            return

        for node in tree:
            self._get_paths(node['des'], path + [node['org']], paths)

        return paths

    def _dict_stations(self, tree, sta=None):
        if sta is None:
            sta = {}

        if not tree:
            return
        else:
            for node in tree:
                sta[node['org'].id] = node['org']
                self._dict_stations(node['des'], sta)

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
        stations (List[Station]): List of Stations being served by the Line
        timetable (Mapping[Station, Tuple[float, float]]): {station ID: (arrival (float), departure (float)}
        pairs (List[Tuple[Station, Station]]): List with attended pairs of stations (origin, destination)
    """

    def __init__(self, id_: int, name: str, corridor: Corridor, timetable: Mapping[str, Tuple[float, float]]):
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

    def __init__(self, id_: int, name: str, hard_type: int, soft_type: int):
        self.id = id_
        self.name = name
        self.hard_type = hard_type
        self.soft_type = soft_type

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.hard_type}, {self.soft_type}]'


class RollingStock(object):
    """
    Rolling Stock: Locomotives, Carriages, Wagons, or other vehicles used on a railway.

    Attributes:
        id (int): Rolling Stock ID
        name (str): Rolling Stock name
        seats (Mapping[int, int]): Number of seats for each hard_type {hard_type1: quantity1, hard_type2: quantity2}
    """

    def __init__(self, id_: int, name: str, seats: Mapping[int, int]):
        self.id = id_
        self.name = name
        self.seats = seats

    def __str__(self):
        return f'[{self.id},{self.name},{self.seats}]'


class TSP(object):
    """
    TSP: Train Service Provider. Company that provides train services.

    Attributes:
        id (int): Train Service Provider ID
        name (name): Name of the Train Service Provider
        rolling_stock List[RollingStock]: List of RollingStock's ID's
    """

    def __init__(self, id_: int, name: str, rolling_stock: List[RollingStock] = None):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock if rolling_stock is not None else []

    def add_rolling_stock(self, rs: RollingStock):
        """
        Method to add new Rolling stock ID to a TSP object

        Args:
            rs (RollingStock): New Rolling Stock
        """
        self.rolling_stock.append(rs)

    def __str__(self):
        return f'[{self.id}, {self.name}, {[rs.id for rs in self.rolling_stock]}]'


class Service(object):
    """
    Service: Travel options provided by a TSP between stations in a Line with a timetable,
    capacity constrains defined by the Rolling Stock, capacity managed by Train or ???
    and with the possibility with different seat types, each with a specific price.

    Attributes:
        id (str): Service ID
        date (datetime.date): Day of service (day, month, year, without time)
        line (Line): Line
        tsp (TSP): TSP
        timeSlot (TimeSlot): Time Slot
        rollingStock (RollingStock): Rolling Stock
        prices Mapping[Tuple[str, str], Mapping[str, float]]: {(org station ID, dest station ID): {seat_type ID: price}}
        capacity (str): String with capacity type
    """

    def __init__(self,
                 id_: str,
                 date: str,
                 line: Line,
                 tsp: TSP,
                 time_slot: TimeSlot,
                 rolling_stock: RollingStock,
                 prices: Mapping[Tuple[str, str], Mapping[Seat, float]],
                 capacity_type: str):

        # {0: {}} # Train capacity
        # Constrained capacity (limit seats available between some pairs of stations)
        # {1: {('MAD', 'ZAR'): {hard_type 1: 20, hard_type 2: 50}, {('ZAR', 'CAL'): {1: 10, 2: 30}}}

        self.id = id_
        self.date = get_date(date)
        self.line = line
        self.tsp = tsp
        self.timeSlot = time_slot
        self.schedule = self._get_absolute_schedule()
        self.rollingStock = rolling_stock
        self.capacityType = capacity_type

        if self.capacityType == 'Train':
            self.capacity = deepcopy(self.rollingStock.seats)  # Mapping[int, int]: {hard_type: quantity}
        else:
            raise NotImplementedError('Not implemented yet - capacityType != Train')

        self.prices = prices

    def _get_absolute_schedule(self):
        absolute_schedule = []
        for dt, at in list(self.line.timetable.values()):
            abs_dt = datetime.timedelta(seconds=dt*60) + self.timeSlot.start
            abs_at = datetime.timedelta(seconds=at*60) + self.timeSlot.start
            absolute_schedule.append((abs_dt, abs_at))

        return absolute_schedule

    def buy_ticket(self, origin: str, destination: str, seat_type: Seat):
        if self.capacityType == 'Train':
            if self.capacity[seat_type.hard_type] > 0:
                self.capacity[seat_type.hard_type] -= 1
        else:
            raise NotImplementedError('Not implemented yet - capacityType != Train')

    def availability(self, origin: str, destination: str, seat_type: Seat):
        if self.capacityType == 'Train':
            return self.capacity[seat_type.hard_type]
        else:
            raise NotImplementedError('Not implemented yet - capacityType != Train')

    def __str__(self):
        new_line = "\n\t\t"
        return f'Service id: {self.id} \n' \
               f'\tDate of service: {self.date} \n' \
               f'\tStops: {[sta.id for sta in self.line.stations]} \n' \
               f'\tLine times (relative): {list(self.line.timetable.values())} \n' \
               f'\tLine times (absolute): {[(format_td(at), format_td(dt)) for at, dt in self.schedule]} \n' \
               f'\tTrain Service Provider: {self.tsp} \n' \
               f'\tTime Slot: {self.timeSlot} \n' \
               f'\tRolling Stock: {self.rollingStock} \n' \
               f'\tPrices: \n' \
               f'\t\t{new_line.join(f"{key}: {value}" for key, value in self.prices.items())} \n' \
               f'\tCapacity: {self.capacity} \n' \
               f'\tCapacity type: {self.capacityType} \n'


class Supply(object):
    """
    Supply: Services available in the system that meet the users requirements:
    From origin station ID (str), to destination station ID (str), on date D as datetime.date object (YYYY-MM-DD)

    Attributes:
        data (Mapping[Mapping, Mapping]): Nested dictionary with the required data for the supply module following the
        structure from supply_data_example.yml file
    """

    def __init__(self, data: Mapping[str, Mapping]):
        self._data = data if data is not None else {}
        self._stations = {} if 'stations' not in self._data else self._get_stations(key='stations')
        self._timeSlots = {} if 'timeSlot' not in self._data else self._get_time_slots(key='timeSlot')
        self._corridors = {} if 'corridor' not in self._data else self._get_corridors(key='corridor')
        self._lines = {} if 'line' not in self._data else self._get_lines(key='line')
        self._seats = {} if 'seat' not in self._data else self._get_seats(key='seat')
        self._rollingStock = {} if 'rollingStock' not in self._data else self._get_rolling_stock(key='rollingStock')
        self._tsps = {} if 'trainServiceProvider' not in self._data else self._get_tsps(key='trainServiceProvider')

        self.services = {} if 'service' not in self._data else self._get_services(key='service')

    @classmethod
    def from_yaml(cls, path: str):
        """
        Class method to create a Supply object from a yaml file

        Args:
            path (str): Path to the yaml file

        Returns:
            Supply object
        """
        with open(path, 'r') as file:
            return cls(yaml.safe_load(file))

    def generate(self, origin: str, destination: str, date: datetime.date) -> List[Service]:
        """
        Generate a List of Services available in the system that meet the users requirements:
        From origin station ID, to destination station ID, on date D

        Args:
            origin (str): Origin Station ID
            destination (str): Destination Station ID
            date (datetime.date): Date of service (day, month, year, without time)

        Returns:
            List[Service]: List of deep copies of Service objects that meet the user requests
        """
        filtered_services = []

        for s in self.services.values():

            if s.date == date and (origin, destination) in s.line.pairs.keys():
                new_s = deepcopy(s)
                new_s.prices = {p: new_s.prices[p] for p in new_s.prices if p == (origin, destination)}
                filtered_services.append(new_s)

        return filtered_services

    def _get_stations(self, key='stations') -> Mapping[str, Station]:
        """
        Private method to build a dict of Station objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'stations'.

        Returns:
            Mapping[str, Station]: Dict of Station objects {station_id: Station object}
        """
        stations = {}
        for s in self._data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'short_name', 'city')), "Incomplete Station data"

            coords = tuple(s.get('coordinates', {'lat': None, 'lon': None}).values())

            stations[s['id']] = Station(s['id'], s['name'], s['city'], s['short_name'], coords)
        return stations

    def _get_time_slots(self, key='timeSlot'):
        """
        Private method to build a dict of TimeSlot objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'timeSlot'.

        Returns:
            Mapping[str, TimeSlot]: Dict of TimeSlot objects {time_slot_id: TimeSlot object}
        """
        time_slots = {}
        for ts in self._data[key]:
            assert all(k in ts.keys() for k in ('id', 'start', 'end')), "Incomplete TimeSlot data"

            time_slots[ts['id']] = TimeSlot(ts['id'], ts['start'], ts['end'])
        return time_slots

    def _get_corridors(self, key='corridor'):
        """
        Private method to build a dict of Corridor objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'corridor'.

        Returns:
            Mapping[str, Corridor]: Dict of Corridor objects {corridor_id: Corridor object}
        """

        def to_station(tree: Mapping, stations: Mapping[str, Station]):
            """
            Recursive function to build a tree of Station objects from a tree of station IDs

            Args:
                tree (Mapping): Tree of station IDs
                stations (Mapping[str, Station]): Dict of Station objects {station_id: Station object}

            Returns:
                List[Mapping]: Tree of Station objects
            """
            if not tree:
                return

            for node in tree:
                node['org'] = stations[node['org']]
                to_station(node['des'], stations)

            return tree

        def set_stations_ids(tree, sta=None):
            if sta is None:
                sta = set()

            if not tree:
                return

            else:
                for node in tree:
                    sta.add(node['org'])
                    set_stations_ids(node['des'], sta)

            return sta

        corridors = {}
        for c in self._data[key]:
            assert all(k in c.keys() for k in ('id', 'name', 'stations')), "Incomplete Corridor data"

            corr_stations_ids = set_stations_ids(c['stations'])

            assert all(s in self._stations.keys() for s in corr_stations_ids), "Station not found in Station list"

            stations_tree = to_station(deepcopy(c['stations']), self._stations)

            corridors[c['id']] = Corridor(c['id'], c['name'], stations_tree)

        return corridors

    def _get_lines(self, key='line'):
        """
        Private method to build a dict of Line objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'line'.

        Returns:
            Mapping[str, Line]: Dict of Line objects {line_id: Line object}
        """
        lines = {}
        for ln in self._data[key]:
            assert all(k in ln.keys() for k in ('id', 'name', 'corridor', 'stops')), "Incomplete Line data"

            assert ln['corridor'] in self._corridors.keys(), "Corridor not found in Corridor list"
            corr = self._corridors[ln['corridor']]

            for stn in ln['stops']:
                assert all(k in stn for k in ('station', 'arrival_time', 'departure_time')), "Incomplete Stops data"

            corr_stations_ids = [s.id for s in corr.stations.values()]
            assert all(s['station'] in corr_stations_ids for s in ln['stops']), "Station not found in Corridor list"

            timetable = {s['station']: (float(s['arrival_time']), float(s['departure_time']))
                         for s in ln['stops']}

            lines[ln['id']] = Line(ln['id'], ln['name'], corr, timetable)

        return lines

    def _get_seats(self, key='seat'):
        """
        Private method to build a dict of Seat objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'seat'.

        Returns:
            Mapping[str, Seat]: Dict of Seat objects {seat_id: Seat object}
        """
        seats = {}
        for s in self._data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), "Incomplete Seat data"

            seats[s['id']] = Seat(s['id'], s['name'], s['hard_type'], s['soft_type'])

        return seats

    def _get_rolling_stock(self, key='rollingStock'):
        """
        Private method to build a dict of RollingStock objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'rollingStock'.

        Returns:
            Mapping[str, RollingStock]: Dict of RollingStock objects {rolling_stock_id: RollingStock object}
        """
        rolling_stock = {}
        for rs in self._data[key]:
            assert all(k in rs.keys() for k in ('id', 'name', 'seats')), "Incomplete RollingStock data"

            for st in rs['seats']:
                assert all(k in st for k in ('hard_type', 'quantity')), "Incomplete seats data for RS"

            assert all(s['hard_type'] in [s.hard_type for s in self._seats.values()] for s in rs['seats']), \
                "Invalid hard_type for RS"

            rs_seats = {s['hard_type']: s['quantity'] for s in rs['seats']}

            rolling_stock[rs['id']] = RollingStock(rs['id'],
                                                   rs['name'],
                                                   rs_seats)

        return rolling_stock

    def _get_tsps(self, key='trainServiceProvider'):
        """
        Private method to build a dict of TSP objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'trainServiceProvider'.

        Returns:
            Mapping[str, TSP]: Dict of TSP objects {tsp_id: TSP object}
        """
        tsp = {}
        for op in self._data[key]:
            assert all(k in op.keys() for k in ('id', 'name', 'rolling_stock')), "Incomplete TSP data"
            assert all(i in self._rollingStock.keys() for i in op['rolling_stock']), "Unknown RollingStock ID"

            tsp[op['id']] = TSP(op['id'], op['name'], [self._rollingStock[i] for i in op['rolling_stock']])

        return tsp

    def _get_services(self, key='service'):
        """
        Private method to build a dict of Service objects from YAML data

        Args:
            key (str): Key to access the data in the YAML file. Default: 'service'.

        Returns:
            Mapping[str, Service]: Dict of Service objects {service_id: Service object}
        """
        services = {}
        for s in self._data[key]:
            service_keys = ('id', 'date', 'line', 'train_service_provider', 'time_slot', 'rolling_stock',
                            'origin_destination_tuples', 'type_of_capacity')

            assert all(k in s.keys() for k in service_keys), "Incomplete Service data"

            service_id = s['id']
            service_date = s['date']

            assert s['line'] in self._lines.keys(), "Line not found"
            service_line = self._lines[s['line']]

            assert s['train_service_provider'] in self._tsps.keys(), "TSP not found"
            service_tsp = self._tsps[s['train_service_provider']]

            assert s['time_slot'] in self._timeSlots.keys(), "TimeSlot not found"
            service_time_slot = self._timeSlots[s['time_slot']]

            assert s['rolling_stock'] in self._rollingStock.keys(), "RollingStock not found"
            service_rs = self._rollingStock[s['rolling_stock']]

            service_prices = {}
            for od in s['origin_destination_tuples']:
                assert all(k in od.keys() for k in ('origin', 'destination', 'seats')), "Incomplete Service prices"

                org = od['origin']
                des = od['destination']
                for st in od['seats']:
                    assert all(k in st for k in ('seat', 'price')), "Incomplete seats data for Service"

                prices = {st['seat']: st['price'] for st in od['seats']}

                service_prices[(org, des)] = prices

            service_capacity = s['type_of_capacity']

            services[service_id] = Service(service_id,
                                           service_date,
                                           service_line,
                                           service_tsp,
                                           service_time_slot,
                                           service_rs,
                                           service_prices,
                                           service_capacity)

        return services
