from entities import *

# Dummy definition - Stations in corridor MAD-BAR
short_names = ("MAD", "GUA", "CAL", "ZAR", "LER", "TAR", "BAR")
names = ("Madrid", "Guadalajara", "Calatayud", "Zaragoza", "Lerida", "Tarragona", "Barcelona")
station_number = tuple(range(len(names)))

stations = list(Station(i, sn, n) for i, sn, n in zip(station_number, names, short_names))

print("Stations: ")
for s in stations:
    print(s)
print()

# Define corridor MAD-BAR
corridorMB = Corridor(1, stations)

print("Stations in Corridor Madrid-Barcelona: ")

for s in corridorMB.list_station:
    print(s)
print()

# Type of Services in corridor MAD-BAR
# OPTIMIZE: Replace with boolean values?
# 0: No service
# 1: Service
services = {1: (1, 1, 0, 1, 1, 1, 1),
            2: (1, 0, 1, 1, 1, 1, 1),
            3: (1, 0, 1, 1, 0, 1, 1),
            4: (1, 0, 0, 1, 0, 0, 1),
            5: (1, 0, 0, 0, 0, 0, 1),
            6: (1, 0, 0, 1, 0, 1, 1)}

# Select service tye
service_type = "1A"  # MAD --> Bar

# Timetable for services in corridor MAD-BAR
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
        schedule = [(round(sum(travelTime[i + 1:]) + sum(stopTime[i + 1:]) + 0.0, 1),
                     round(sum(travelTime[i + 1:]) + sum(stopTime[i:]) + 0.0, 1))
                    for i in range(len(travelTime))]

        time_table[kb] = tuple(schedule)

print("Time Table - Corridor MAD-BAR")
for item in time_table.items():
    print(item)
print()

lineMB = Line(1, corridorMB, services[int(service_type[0])], time_table[service_type])

print("Train stops 'j' in Line Madrid-Barcelona - Service type: ", service_type)

for j, schedule in zip(lineMB.J, lineMB.schedule):
    AT = schedule[0]
    DT = schedule[1]
    print(j, "- AT: ", AT, " - DT: ", DT)
print()

print("Pairs of stations 'w' in Line Madrid-Barcelona - Service type: ", service_type)

for i, w in enumerate(lineMB.W):
    print("Origin: ", w[0], " - Destination: ", w[1])
print()

# Dummy definition TSP
renfeTSP = TSP(1071, "RENFE OPERADORA", "RENFE")  # 1071: Renfe Agency ID from GTFS
print(renfeTSP)

dictTSPs = {renfeTSP.id: renfeTSP}

# Dummy definition Rolling Stock
renfeRS = RollingStock(1, "TALGO AVRIL", dictTSPs[1071], {1: 50, 2: 75})
print(renfeRS)



