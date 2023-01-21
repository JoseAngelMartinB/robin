from typing import List, Tuple
import datetime


class Station(object):
    """
    Station class

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
    
    def __str__(self):
        return f'[{self.id}, {self.name}, {self.shortname}, {self.coords}]'
    
    
class TimeSlot(object):
    """
    Time slot class

    Attributes:
        id (int): Time slot ID
        start (float): Time slot start time
        end (float): Time slot end time
        class_mark (float): Time slot class mark
        size (float): Time slot size
    """
    def __init__(self, id_: int, start: str, end: str):
        self.id = id_
        start_hour, start_min = start.split(':')
        self.start = datetime.timedelta(hours=int(start_hour), minutes=int(start_min))

        end_hour, end_min = end.split(':')
        self.end = datetime.timedelta(hours=int(end_hour), minutes=int(end_min))

        self.class_mark = self.__get_class_mark()
        self.size = self.__get_size()

    def __get_class_mark(self) -> datetime.timedelta:
        """
        Get class mark of time slot
        :return: class mark
        """

        if self.end < self.start:
            return (self.start + self.end + datetime.timedelta(days=1)) / 2 - datetime.timedelta(days=1)
        return (self.start + self.end) / 2

    def __get_size(self) -> datetime.timedelta:
        """
        Get size of time slot
        :return: size
        """
        if self.end < self.start:
            return (self.end + datetime.timedelta(days=1)) - self.start
        return self.end - self.start

    def __str__(self):
        return f'[{self.id}, {self.start}, {self.end}, {self.class_mark}, {self.size}]'


class Corridor(object):
    """
    Corridor class

    Attributes:
        id (int): Corridor ID
        list_station (List[Station]): List of stations in corridor
    """

    # Be aware: Corridor could be a tree structure (parent-child relationship between stations)
    def __init__(self, id_: int, name: str, stations: List[Station]):
        self.id = id_
        self.name = name
        self.stations = stations

    def __str__(self):
        short_names = [s.shortname for s in self.stations]
        return f'[{self.id}, {self.name}, {short_names}]'


class Line(object):
    """
    Line class

    Attributes:
        id (int): Line ID
        name (str): Line name
        corridor (Corridor): Line corridor
        timetable (List[dict]): List of dictionaries [{stations: (arrival, departure)}]
    """
    def __init__(self, id_: int, name: str, corridor: Corridor, timetable: dict):
        # Service type is a tuple of booleans indicating which stations from the corridor are served
        self.id = id_
        self.name = name
        self.name = f'Line {id_}'
        self.corridor = corridor
        self.stops = list(timetable.keys())  # j: Train stops
        self.timetable = timetable  # t: Timetable
        self.pairs = self.__getpairs()  # w: Pairs origin-destination

    def __getstops(self, servicetype):
        """
        DEPRECATED
        Private method to get the stops of the line

        Input arguments:
            serviceType: Tuple with the service type
        Output:
            List of stops
        """
        return [s for i, s in zip(servicetype, self.corridor.stations) if i]

    def __getpairs(self):
        """
        Private method to get each pair of stations of the line

        Input arguments:
            None
        Output:
            List of pairs
        """
        return [(a, b) for i, a in enumerate(self.stops) for b in self.stops[i + 1:]]

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.corridor}, {self.timetable}]'


class Seat(object):
    def __init__(self, id_: int, name: str, hard_type: int, soft_type: int):
        self.id = id_
        self.name = name
        self.hard_type = hard_type  # E.g. Tickets available?
        self.soft_type = soft_type  # E.g. Luggage compartment?

    def __str__(self):
        return f'[{self.id}, {self.name}, {self.hard_type}, {self.soft_type}]'


class RollingStock(object):
    def __init__(self, id_: int, name: str, seats: dict):
        self.id = id_
        self.name = name
        self.seats = seats  # {seat_type: quantity}

    def __str__(self):
        return f'[{self.id},{self.name},{self.seats}]'


class TSP(object):
    def __init__(self, id_: int, name: str, rolling_stock=None):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock if rolling_stock is not None else []

    def add_rolling_stock(self, rs: RollingStock):
        self.rolling_stock.append(rs)

    def __str__(self):
        rs_ids = [rs.id for rs in self.rolling_stock]
        return f'[{self.id}, {self.name}, {rs_ids}]'


class Service(object):
    def __init__(self,
                 id_: int,
                 date: str,
                 line: Line,
                 tsp: TSP,
                 time_slot: TimeSlot,
                 rolling_stock: RollingStock,
                 prices: dict,
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
        return f'[{self.id}, {self.date}, {self.line.stops}, {self.tsp.name}, \n,' \
               f'{self.timeSlot.id}, {self.rollingStock.id}, \n,' \
               f'{self.prices}, {self.capacity}]'


class Supply(object):
    # (id, date, w(o, d), seat_offer, schedule, price on time t)
    def __init__(self, id_: int, origin: str, destination: str, date: datetime.date, services: List[Service]):
        self.id = id_
        self.origin = origin
        self.destination = destination
        self.date = date
        self.services = self.__getservices(services)

    def __getservices(self, services):
        my_services = []

        for s in services:
            if s.date == self.date:
                try:
                    if s.line.stops.index(self.origin) < s.line.stops.index(self.destination):
                        my_services.append(s)
                except ValueError:
                    pass

        return my_services

