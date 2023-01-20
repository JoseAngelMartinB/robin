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
    def __init__(self, id_: int, name: str, shortname: str, coords: Tuple[float, float] = None):
        self.id = id_
        self.name = name
        self.shortname = shortname
        self.coords = coords
    
    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname},{self.coords}]'
    
    
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
    def __init__(self, id_: int, start: float, end: float):
        self.id = id_
        self.start = start
        self.end = end
        self.class_mark = self.__get_class_mark()
        self.size = self.__get_size()

    # TODO: start and end as datetime.time?

    def __get_class_mark(self) -> float:
        """
        Get class mark of time slot
        :return: class mark
        """
        return (self.start + self.end) / 2

    def __get_size(self) -> float:
        """
        Get size of time slot
        :return: size
        """
        return self.end - self.start

    def __str__(self):
        return f'[{self.id},{self.class_mark},{self.size}]'


class Corridor(object):
    """
    Corridor class

    Attributes:
        id (int): Corridor ID
        list_station (List[Station]): List of stations in corridor
    """

    # Corridor could be a tree structure (parent-child relationship between stations)
    def __init__(self, id_: int, list_station: List[Station] = []):
        self.id = id_
        self.list_station = list_station

    def insert_station(self, station: Station):
        self.list_station.append(station)

    def __str__(self):
        return f'[{self.id},{self.list_station}]'


class Line(object):
    """
    Line class

    Attributes:
        id (int): Line ID
        name (str): Line name
        corr (Corridor): Line corridor
        lstation (List[Station]): List of stations in line
        timetable (List[Tuple]): List of tuples (arrival time, departure time)
    """
    def __init__(self, id_: int, corr: Corridor, service_type: Tuple = (), timetable: List[Tuple] = []):
        # Service type is a tuple of booleans indicating which stations from the corridor are served
        self.id = id_
        self.name = f'Line {id_}'
        self.corr = corr
        # self.lstation = lstation

        self.J = self.__getstops(service_type)  # j: Train stops
        self.W = self.__getpairs()  # w: Pairs origin-destination
        self.schedule = timetable  # TODO: - Get schedule - DEFAULT Left to Right

        # self.timetable=ltimetable # list of tuple (AT,DT)

    def __getstops(self, servicetype):
        """
        Private method to get the stops of the line

        Input arguments:
            serviceType: Tuple with the service type
        Output:
            List of stops
        """
        return [s for i, s in zip(servicetype, self.corr.list_station) if i]

    def __getpairs(self):
        """
        Private method to get each pair of stations of the line

        Input arguments:
            None
        Output:
            List of pairs
        """
        return [(a, b) for i, a in enumerate(self.J) for b in self.J[i + 1:]]
        
    def insert_station(self, station: Station, at: float, dt: float):
        self.lstation.append(station)
        self.timetable.append((at, dt))  # TODO: (AT,DT) dependent on travel way --> Not a single (AT, DT) tuple?


class Seat(object):
    def __init__(self, id_: int, name: str, hard_type: int, soft_type: int):
        self.id = id_
        self.name = name
        self.hard_type = hard_type  # E.g. Tickets available?
        self.soft_type = soft_type  # E.g. Luggage compartment?

    def __str__(self):
        return f'[{self.id}, {self.hard_type}, {self.soft_type}]'


class TSP(object):
    def __init__(self, id_: int, name: str, rolling_stock: List = []):
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock  # TODO:

    def __str__(self):
        return f'[{self.id},{self.name},{self.rolling_stock}]'


class RollingStock(object):
    def __init__(self, id_: int, name: str, tsp: TSP, seats: dict):
        self.id = id_
        self.name = name
        self.tsp = tsp
        self.seats = seats  # E.G. key: hard_type, val: quantity - {1: 50, 2: 250}

    def __str__(self):
        return f'[{self.id},{self.name},{self.tsp},{self.seats}]'


# TODO:
class Service(object):
    # (id, date, line, time_slot, seat_type, time_table, price, type_capacity, TSP_capacity, rolling_stock)
    def __init__(self, id_: int, day: datetime.datetime, line: Line):
        self.id = id_
        self.day = day
        self.line = line
        self.timeslot = None
        self.train = None
        self.day = None

    def __str__(self):
        return f'[{self.id},{self.day},{self.line}]'


# TODO:
class Supply(object):
    # (id, date, w(o, d), seat_offer, schedule, price on time t)
    def __init__(self, id_: int, date: datetime.date, w: Tuple[Station, Station]):
        self.id = id_
        self.w = w

        # Get supply for a given date and a given pair of stations

        # Retrieve information from the database
        # Build services with the information from the database
