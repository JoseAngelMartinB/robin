from typing import List, Tuple


class Station (object):
    def __init__(self,id:int,name:str,shortname:str):
        self.id=id
        self.name=name
        self.shortname=shortname
    
    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'
    
    
class TimeSlot (object):
    def __init__(self,id:int,classmark:float,size:float):
        self.id=id
        self.classmark=classmark
        self.size=size
        
    def __str__(self):
        return f'[{self.id},{self.classmark},{self.size}]'


class Corridor (object):
    def __init__(self,id:int,liststation:List[Station]=[]):
        self.id=id
        self.liststation=liststation
    def insertStation(self,station:Station):
        self.liststation.append(station)
    def __str__(self):
        return f'[{self.id},{self.liststation}]'




class Line (object):
    def __init__(self, id:int, corr:Corridor, serviceType:Tuple=(), timeTable:List[Tuple]=[]):
        self.id = id
        self.corr = corr
        #self.lstation = lstation

        self.J = self.__getStops__(serviceType)  # j: Train stops
        self.W = self.__getPairs__()  # w: Pairs origin-destination
        self.schedule = timeTable  # TBD - Get schedule - DEFAULT Left to Right

        # self.timetable=ltimetable # list of tuple (AT,DT)

    def __getStops__(self, lstation):
        """
        Input:
            - self
            - lstation: Tuple with same len() as corr, where 1 represents a station attended, 0 otherwise
        Output: Returns a list containing every train stop attended by the line instance
        """
        return [s for i, s in zip(lstation, self.corr.liststation) if i]

    def __getPairs__(self):
        """
        Input: self
        Output: Returns a list of tuples with each pair of stations attended by the line instance
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

    def __str__(self):
        return f'[{self.id},{self.name},{self.shortname}]'


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


if __name__=='__main__':
    # Dummy definition - Stations in corridor MAD-BAR
    short_names = ("MAD", "GUA", "CAL", "ZAR", "LER", "TAR", "BAR")
    names = ("Madrid", "Guadalajara", "Calatayud", "Zaragoza", "Lerida", "Tarragona", "Barcelona")
    station_number = tuple(range(len(names)))

    stations = list(Station(i, sn, n) for i, sn, n in zip(station_number, names, short_names))

    print("Stations: ")
    for s in stations:
        print(s.__str__())
    print()

    # Define corridor MAD-BAR
    corridorMB = Corridor(1, stations)

    print("Stations in Corridor Madrid-Barcelona: ")

    for s in corridorMB.liststation:
        print(s.__str__())
    print()

    # Type of Services in corridor MAD-BAR
    services = {1: (1, 1, 0, 1, 1, 1, 1),
                2: (1, 0, 1, 1, 1, 1, 1),
                3: (1, 0, 1, 1, 0, 1, 1),
                4: (1, 0, 0, 1, 0, 0, 1),
                5: (1, 0, 0, 0, 0, 0, 1),
                6: (1, 0, 0, 1, 0, 1, 1)}

    # Select service tye
    service_type = "1A" # MAD --> Bar


    # Time-table for services in corridor MAD-BAR
    # Type A: Way MAD --> BAR
    # Type B: Way MAD <-- BAR
    time_table = {
        1: ((0.0, 0.0), (15.6, 20.6), (58.6, 58.6), (79.3, 94.3), (127.0, 134.5), (153.5, 161.0), (185.1, 185.1)),
        2: ((0.0, 0.0), (15.6, 15.6), (53.6, 58.6), (79.3, 94.3), (127.0, 134.5), (153.5, 161.0), (185.1, 185.1)),
        3: ((0.0, 0.0), (15.6, 15.6), (53.6, 58.6), (79.3, 94.3), (127.0, 127.0), (146.0, 153.5), (177.6, 177.6)),
        4: ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 89.3), (122.0, 122.0), (141.0, 141.0), (165.1, 165.1)),
        5: ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 74.3), (107.0, 107.0), (126.0, 126.0), (150.1, 150.1)),
        6: ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 89.3), (122.0, 122.0), (141.0, 148.5), (172.6, 172.6))}

    # Boolean: Time table independent of travel way?
    scheduleIndTW = True
    # If not independent --> New stop and travel times must be specified!

    if scheduleIndTW:
        for k in list(time_table.keys())[:]:
            # Key name - Default travel way (List order)
            ka = str(k) + "A"

            # Key name - Reverse travel way (Reverse list order)
            kb = str(k) + "B"

            # Rename dict entry for key "n" with name "nA"
            time_table[ka] = time_table.pop(k)

            # Get stop time for each station
            stopTime = [abs(x[0] - x[1]) for x in time_table[ka]]

            # Get travel time between stations
            # Insert travel time for 1st station - [0.0]
            travelTime = [0.0] + [abs(time_table[ka][i + 1][0] - time_table[ka][i][1]) for i in
                          range(len(time_table[ka]) - 1)]

            # Get schedule for reverse travel way
            schedule = [(round(sum(travelTime[i + 1:]) + sum(stopTime[i + 1:]), 1),
                         round(sum(travelTime[i + 1:]) + sum(stopTime[i:]), 1))
                         for i in range(len(travelTime))]

            time_table[kb] = schedule

    print("Time Table")
    for item in time_table.items():
        print(item)
    print()

    # TODO: Schedule is NOT always independent on the travel-way
    lineMB = Line(1, corridorMB, services[int(service_type[0])], time_table[service_type])

    print("Train stops 'j' in Line Madrid-Barcelona - Service type: ", service_type)

    for j, schedule in zip(lineMB.J, lineMB.schedule):
        AT = schedule[0]
        DT = schedule[1]
        print(j.__str__(), "- AT: ", AT, " - DT: ", DT)
    print()

    print("Pairs of stations 'w' in Line Madrid-Barcelona - Service type: ", service_type)

    for i, w in enumerate(lineMB.W):
        print("Origin: ", w[0].__str__(), " - Destination: ", w[1].__str__())
    print()