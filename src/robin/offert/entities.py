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
    def __init__(self, id:int, corr:Corridor, serviceType:Tuple=(), ltimetable:List[Tuple]=[]):
        self.id = id
        self.corr = corr
        #self.lstation = lstation

        self.j = self.__getStops__(serviceType)  # j: Train stops
        self.w = self.__getPairs__()  # w: Pairs origin-destination

        self.schedule = self.__getSchedule__()  # TBD - Get schedule - DEFAULT Left to Right
        self.timetable=ltimetable # list of tuple (AT,DT)

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
        return [(a, b) for i, a in enumerate(self.j) for b in self.j[i + 1:]]

    # TODO: TBD
    def __getSchedule__(self):
        """
        Input:
            - self
            - timetable
        Output: Returns a list of tuple (AT, DT)
        """
        pass
        
    def insertStation(self,station:Station,AT:float,DT:float):
        self.lstation.append(station)
        self.timetable.append((AT,DT))

    
if __name__=='__main__':
    #station1=Station(1,'station1','s1')
    #station2=Station(2,'station2','s2')
    #c=Corridor(1,[station1,station2])
    #print(c)

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
    service_type = 1

    # Time-table for services in corridor MAD-BAR
    # Type A: Way MAD --> BAR
    # Type B: Way MAD <-- BAR
    time_table = {
        "1A": ((0.0, 0.0), (15.6, 20.6), (58.6, 58.6), (79.3, 94.3), (127.0, 134.5), (153.5, 161.0), (185.1, 185.1)),
        "2A": ((0.0, 0.0), (15.6, 15.6), (53.6, 58.6), (79.3, 94.3), (127.0, 134.5), (153.5, 161.0), (185.1, 185.1)),
        "3A": ((0.0, 0.0), (15.6, 15.6), (53.6, 58.6), (79.3, 94.3), (127.0, 127.0), (146.0, 153.5), (177.6, 177.6)),
        "4A": ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 89.3), (122.0, 122.0), (141.0, 141.0), (165.1, 165.1)),
        "5A": ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 74.3), (107.0, 107.0), (126.0, 126.0), (150.1, 150.1)),
        "6A": ((0.0, 0.0), (15.6, 15.6), (53.6, 53.6), (74.3, 89.3), (122.0, 122.0), (141.0, 148.5), (172.6, 172.6)),
        "1B": ((185.1, 185.1), (164.5, 169.5), (126.5, 126.5), (90.8, 105.8), (50.6, 58.1), (24.1, 31.6), (0.0, 0.0)),
        "2B": ((185.1, 185.1), (169.5, 169.5), (126.5, 131.5), (90.8, 105.8), (50.6, 58.1), (24.1, 31.6), (0.0, 0.0)),
        "3B": ((177.6, 177.6), (162.0, 162.0), (119.0, 124.0), (83.3, 98.3), (50.6, 50.6), (24.1, 31.6), (0.0, 0.0)),
        "4B": ((165.1, 165.1), (149.5, 149.5), (111.5, 111.5), (75.8, 90.8), (43.1, 43.1), (24.1, 24.1), (0.0, 0.0)),
        "5B": ((150.1, 150.1), (134.5, 134.5), (96.5, 96.5), (75.8, 75.8), (43.1, 43.1), (24.1, 24.1), (0.0, 0.0)),
        "6B": ((172.6, 172.6), (157.0, 157.0), (119.0, 119.0), (83.3, 98.3), (50.6, 50.6), (24.1, 31.6), (0.0, 0.0))}

    # TODO: Define if schedule in allways independent of the travel-way
    #timetable_line = [v for k, v in time_table.items() if str(line_type) in k]
    timetable_line = []

    lineMB = Line(1, corridorMB, services[service_type], timetable_line)

    print("Train stops 'j' in Line Madrid-Barcelona - Service type: ", service_type)

    for j in lineMB.j:
        print(j.__str__())
    print()

    print("Pairs of stations 'w' in Line Madrid-Barcelona - Service type: ", service_type)

    for w in lineMB.w:
        print(w[0].__str__(), w[1].__str__())
    print()