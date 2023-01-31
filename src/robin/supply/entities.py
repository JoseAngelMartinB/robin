from typing import List, Tuple, Mapping
import datetime


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
    def __init__(self, id_: int, start: str, end: str):
        self.id = id_

        start_hour, start_min = start.split(':')
        self.start = datetime.timedelta(hours=int(start_hour), minutes=int(start_min))

        end_hour, end_min = end.split(':')
        self.end = datetime.timedelta(hours=int(end_hour), minutes=int(end_min))

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
        stations (List[int]): List of Station IDs of which the Corridor is composed
    """

    # TODO: Corridor as a tree structure (parent-child relationship between stations)
    def __init__(self, id_: int, name: str, stations: List[int]):
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
        corridor (int): Corridor ID where the Line belongs to
        stops (List[int]): List of Station IDs being served by the Line
        timetable (Mapping[str, Tuple[float, float]]): {Station ID (str): (arrival (str), departure (str)}
        pairs (List[Tuple[str, str]]): List of pairs of stations (origin, destination)
    """
    def __init__(self, id_: int, name: str, corridor: int, timetable: Mapping[str, Tuple[float, float]]):
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
        rolling_stock List[int]: List of RollingStock id's (int)
    """
    def __init__(self, id_: int, name: str, rolling_stock_ids: List[int] = None):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock_ids if rolling_stock_ids is not None else []

    def add_rolling_stock(self, rs_id: int):
        """
        Method to add new Rolling stock ID to a TSP object

        Args:
            rs_id: RollingStock object
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
        date (datetime.date): Day of sevice (day, month, year, without time)
        line (int): Line ID
        tsp (int): TSP ID
        timeSlot (TimeSlot) : Time Slot
        rollingStock (int): Rolling Stock ID
        prices Mapping[Tuple[str, str], Mapping[int, float]]: {(org, dest): {seat_type: price, ...}}
        capacity (str): String with capacity type
    """
    def __init__(self,
                 id_: str,
                 date: str,
                 line: Line,
                 tsp: int,
                 time_slot: TimeSlot,  # TODO: Change to TimeSlot ID
                 rolling_stock: int,
                 prices: Mapping[Tuple[str, str], Mapping[int, float]],
                 capacity: str):

        self.id = id_
        self.date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
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
    Supply:

    Attributes:
        id (int): Supply ID
        origin (str): String with origin station id
        destination (str): String with destination station id
        date (datetime.date): Date of supply (day, month, year, without time)
        services (List[Service]): List of Service objects
    """
    def __init__(self, id_: int, origin: str, destination: str, date: datetime.date, services: List[Service]):
        self.id = id_
        self.origin = origin
        self.destination = destination
        self.date = date
        self.services = self.__getservices(services)  # TODO: Change to services ID'S

    def __getservices(self, services: List[Service]) -> List[Service]:
        """
        Private method to get the services that meet the user requests (origin, destination, date)

        Args:
            services (List[Service]): List of Service objects

        Returns:
            List[Service]: Filtered List of Service objects
        """
        filtered_services = []

        for s in services:
            if s.date == self.date and (self.origin, self.destination) in s.line.pairs:
                filtered_services.append(s)

        return filtered_services

