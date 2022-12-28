from typing import List, Tuple
import datetime


class Station(object):
    def __init__(self, id_: int, name: str, shortname: str, coords: Tuple[float, float] = None):
        self.id = id_
        self.name = name
        self.shortname = shortname
        self.coords = coords
    
    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname},{self.coords}]'
    
    
class TimeSlot(object):
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
    def __init__(self, id_: int, list_station: List[Station] = []):
        self.id = id_
        self.list_station = list_station

    def insert_station(self, station: Station):
        self.list_station.append(station)

    def __str__(self):
        return f'[{self.id},{self.list_station}]'


class Line(object):
    def __init__(self, id_: int, corr: Corridor, service_type: Tuple = (), timetable: List[Tuple] = []):
        # Service type is a tuple of booleans indicating which stations from the corridor are served
        self.id = id_
        self.name = f'Line {id_}'  # TODO: get name from somewhere
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


# TODO
class Seat(object):
    # (id, hard, soft)
    def __init__(self, id_: int, name: str, hard_type: int, soft_type: int):
        # hard_type E.g. Linked to capacity constrains (e.g. tickets available)
        # soft_type - E.g. Luggage availability
        self.id = id_
        self.name = name
        self.hard_type = hard_type  # E.g. Tickets available?
        self.soft_type = soft_type  # E.g. Luggage compartment?

    def __str__(self):
        return f'[{self.id}, {self.s1}, {self.s2}]'


# TODO:
class TSP(object):
    def __init__(self, id_: int, name: str, shortname: str):
        self.id = id_
        self.name = name
        self.shortname = shortname

        # TODO: Not considered in specification table
        # Could retrieve information from RollingStock object?
        # self.rolling_stock = rolling_stock

    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'


# TODO:
class RollingStock(object):
    def __init__(self, id: int, tsp: TSP, s1: None, kmax: None):
        self.id = id
        self.tsp = tsp
        self.S1 = s1
        self.Kmax = kmax

        # TODO: FROM YML SPECS
        # self.name = name
        # self.seats = seats E.G. key: hard_type, val: quantity - {1: 50, 2: 250}

    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'


# TODO:
class Service(object):
    # (id, date, line, time_slot, seat_type, time_table, price, type_capacity, TSP_capacity, rolling_stock)
    def __init__(self, id_: int, day: datetime.datetime, line: Line):
        self.id = id_
        self.line = line
        self.timeslot = None
        self.train = None
        self.day = None


# TODO:
class Supply(object):
    # (id, date, w(o, d), seat_offer, schedule, price on time t)
    def __init__(self, id_: int, date: None, line: Line):
        self.id = id_
        self.date = None
        self.w = None
        self.seats = None
        self.seatType = None
        self.timeTable = None
        self.price = None