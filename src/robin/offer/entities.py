from typing import List, Tuple


class Station(object):
    def __init__(self, id:int, name : str, shortname : str, coords : Tuple[float, float]=None):
        self.id=id
        self.name=name
        self.shortname=shortname
        self.coords=coords
    
    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname},{self.coords}]'
    
    
class TimeSlot(object):
    def __init__(self, id:int, classmark:float, size:float):
        self.id=id
        self.classmark=classmark
        self.size=size
        
    def __str__(self):
        return f'[{self.id},{self.classmark},{self.size}]'


class Corridor(object):
    def __init__(self, id:int, liststation:List[Station]=[]):
        self.id=id
        self.liststation=liststation
    def insertStation(self,station:Station):
        self.liststation.append(station)
    def __str__(self):
        return f'[{self.id},{self.liststation}]'


class Line(object):
    def __init__(self, id:int, corr:Corridor, serviceType:Tuple=(), timeTable:List[Tuple]=[]):
        self.id = id
        self.corr = corr
        #self.lstation = lstation

        self.J = self.__getstops(serviceType)  # j: Train stops
        self.W = self.__getpairs()  # w: Pairs origin-destination
        self.schedule = timeTable  # TBD - Get schedule - DEFAULT Left to Right

        # self.timetable=ltimetable # list of tuple (AT,DT)

    def __getstops(self, serviceType):
        """
        Private method to get the stops of the line

        Input arguments:
            serviceType: Tuple with the service type
        Output:
            List of stops
        """
        return [s for i, s in zip(serviceType, self.corr.liststation) if i]

    def __getpairs(self):
        """
        Private method to get each pair of stations of the line

        Input arguments:
            None
        Output:
            List of pairs
        """
        return [(a, b) for i, a in enumerate(self.J) for b in self.J[i + 1:]]
        
    def insertStation(self,station:Station,AT:float,DT:float):
        self.lstation.append(station)
        self.timetable.append((AT,DT)) # TODO: (AT,DT) dependent on travel way --> Not a single (AT, DT) tuple?


# TODO
class Seat(object):
    # (id, hard, soft)
    def __init__(self, id:int, s1:Tuple=(), s2:Tuple=()):
        # s1: hard limit. Linked to capacity constrains (e.g. tickets avaiable)
        # s2: soft limit. (e.g. luggage availability)
        self.id = id
        self.s1 = s1
        self.s3 = s2

    def __str__(self):
        return f'[{self.id}, {self.s1}, {self.s2}]'


# TODO:
class TSP(object):
    def __init__(self, id: int, name: str, shortname: str):
        self.id = id
        self.name = name
        self.shortname = shortname

    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'


# TODO:
class RollingStock(object):
    def __init__(self, id: int, tsp:TSP, S1:None, Kmax:None):
        self.id = id
        self.tsp = tsp
        self.S1 = None
        self.Kmax = None

    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'


# TODO:
class Service(object):
    # (id, date, line, time_slot, seat_type, time_table, price, type_capacity, TSP_capacity, rolling_stock)
    def __init__(self, id: int, date:None, line:Line):
        self.id = id
        self.date = None
        self.line = None
        self.timeSlot = None
        self.seatType = None
        self.timeTable = None
        self.price = None
        self.typeCapacity = None
        self.tspCapacity = None
        self.rollingStock = None


# TODO:
class Supply(object):
    # (id, date, w(o, d), seat_offer, schedule, price on time t)
    def __init__(self, id: int, date: None, line: Line):
        self.id = id
        self.date = None
        self.w = None
        self.seats = None
        self.seatType = None
        self.timeTable = None
        self.price = None