"""Entities for the supply module."""

from src.robin.supply.utils import get_time, get_date

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
    Corridor: Sequence of stations.

    Attributes:
        id (int): Corridor ID
        name (str): Corridor name
        stations (List[Station]): List of Station's of which the Corridor is composed
    """

    # TODO: Corridor as a tree structure (parent-child relationship between stations - List[Tuple[str, str]]??)
    def __init__(self, id_: int, name: str, stations: List[Station]):
        self.id = id_
        self.name = name
        self.stations = stations

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.stations}]'


class Line(object):
    """
    Line: Sequence of stations being served by a train with a timetable.

    Attributes:
        id (int): Line ID
        name (str): Line name
        corridor (Corridor): Corridor ID where the Line belongs to
        stops (List[Station]): List of Station IDs being served by the Line
        timetable (Mapping[Station, Tuple[float, float]]): {Station (Station): (arrival (str), departure (str)}
        pairs (List[Tuple[Station, Station]]): List of pairs of stations (origin, destination)
    """
    def __init__(self, id_: int, name: str, corridor: Corridor, timetable: Mapping[Station, Tuple[float, float]]):
        self.id = id_
        self.name = name
        self.corridor = corridor
        self.stops = list(timetable.keys())
        self.timetable = timetable
        self.pairs = self._getpairs()

    def _getpairs(self):
        """
        Private method to get each pair of stations of the line, using the stops list

        Returns:
            List of tuple pairs (origin, destination)
        """
        return [(a, b) for i, a in enumerate(self.stops) for b in self.stops[i + 1:]]

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
        rolling_stock List[int]: List of RollingStock's ID's
    """
    def __init__(self, id_: int, name: str, rolling_stock: List[int] = None):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock if rolling_stock is not None else []

    def add_rolling_stock(self, rs_id: int):
        """
        Method to add new Rolling stock ID to a TSP object

        Args:
            rs_id (int): New Rolling Stock ID
        """
        self.rolling_stock.append(rs_id)

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.rolling_stock}]'


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
                 prices: Mapping[Tuple[Station, Station], Mapping[Seat, float]],
                 capacity: str):  # TODO: Check capacity in docs

        self.id = id_
        self.date = get_date(date)
        self.line = line
        self.tsp = tsp
        self.timeSlot = time_slot
        self.rollingStock = rolling_stock
        self.prices = prices
        self.capacity = capacity

    def __str__(self):
        new_line = "\n\t\t"
        return f'Service id: {self.id} \n' \
               f'\tDate of service: {self.date} \n' \
               f'\tStops: {self.line.stops} \n' \
               f'\tTrain Service Provider: {self.tsp} \n' \
               f'\tTime Slot: {self.timeSlot} \n' \
               f'\tRolling Stock: {self.rollingStock} \n' \
               f'\tPrices: \n' \
               f'\t\t{new_line.join(f"{key}: {value}" for key, value in self.prices.items())} \n' \
               f'\tCapacity type: {self.capacity}'


class Supply(object):
    """
    Supply: Intended to provide a list of services that meet the user requests (origin, destination, date)

    Attributes:
        services (List[Service]): List of all Services available objects
    """
    # data attribute will be a dict of variable shape
    def __init__(self, data: None):  # TODO: Check data type annotation
        self.data = data if data is not None else {}
        self.stations = self._get_stations()
        self.timeSlots = self._get_time_slots()
        self.corridors = self._get_corridors()
        self.lines = self._get_lines()
        self.seats = self._get_seats()
        self.rollingStock = self._get_rolling_stock()
        self.tsps = self._get_tsps()
        self.services = self._get_services()

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as file:
            return cls(yaml.safe_load(file))

    def generate(self, origin: Station, destination: Station, date: datetime.date) -> List[Service]:
        """
        Method to get the services that meet the user requests (origin, destination, date)

        Args:
            origin (Station): Origin Station object
            destination (Station): Destination Station object
            date (datetime.date): Date of service (day, month, year, without time)

        Returns:
            List[Service]: Filtered List of Service objects
        """
        filtered_services = []

        for s in self.services.values():
            if s.date == date and (origin, destination) in s.line.pairs:
                filtered_services.append(s)

        return filtered_services

    def _get_stations(self, key='stations'):
        return {tuple(s.values())[0]: Station(*s.values()) for s in self.data[key]}

    def _get_time_slots(self, key='timeSlot'):
        return {tuple(ts.values())[0]: TimeSlot(*list(ts.values())[:-1]) for ts in self.data[key]}

    def _get_corridors(self, key='corridor'):
        corridors = {}
        for c in self.data[key]:
            corridor_data = list(c.values())
            corr_stations = list(filter(lambda s: s in corridor_data[2], self.stations))
            corr_stations = [s for s in corr_stations]

            corridors[corridor_data[0]] = Corridor(corridor_data[0],
                                                   corridor_data[1],
                                                   corr_stations)

        return corridors

    def _get_lines(self, key='line'):
        lines = {}
        for line in self.data[key]:
            line_data = list(line.values())
            timetable = {tuple(s.values())[0]: tuple(s.values())[1:] for s in line_data[3]}

            lines[line_data[0]] = Line(line_data[0],
                                       line_data[1],
                                       self.corridors[line_data[2]],
                                       timetable)

        return lines

    def _get_seats(self, key='seat'):
        return {tuple(s.values())[0]: Seat(*s.values()) for s in self.data[key]}

    def _get_rolling_stock(self, key='rollingStock'):
        rolling_stock = {}
        for rs in self.data[key]:
            rs_data = list(rs.values())
            rs_seats = {tuple(s.values())[0]: tuple(s.values())[1] for s in rs_data[2]}

            rolling_stock[rs_data[0]] = RollingStock(rs_data[0],
                                                     rs_data[1],
                                                     rs_seats)

        return rolling_stock

    def _get_tsps(self, key='trainServiceProvider'):
        tsp = {}
        for op in self.data[key]:
            op_data = list(op.values())
            tsp[op_data[0]] = TSP(op_data[0], op_data[1], op_data[2])

        return tsp

    def _get_services(self, key='service'):
        services = {}
        for s in self.data[key]:
            service_data = list(s.values())
            service_id, service_date = service_data[:2]
            service_line = self.lines[service_data[2]]
            service_tsp = self.tsps[service_data[3]]
            service_time_slot = self.timeSlots[service_data[4]]
            service_rs = self.rollingStock[service_data[5]]
            service_stops = service_data[6]

            service_prices = {}
            for s in service_stops:
                org, des, prices = tuple(s.values())
                prices = {tup[0]: tup[1] for tup in [tuple(t.values()) for t in prices]}

                service_prices[(org, des)] = prices
            service_capacity = service_data[7]

            services[service_id] = Service(service_id,
                                           service_date,
                                           service_line,
                                           service_tsp,
                                           service_time_slot,
                                           service_rs,
                                           service_prices,
                                           service_capacity)

        return services
