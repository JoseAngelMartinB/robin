from entities import *
import yaml

with open('../../../data/supply_data_example.yml', 'r') as file:
    supply_data_example = yaml.safe_load(file)

print(supply_data_example)

# Get list of Station objects from YML data
stations = {tuple(s.values())[0]: Station(*s.values()) for s in supply_data_example['stations']}

print("Stations: ")
for s in stations.values():
    print(s)
print()

# Get list of TimeSlot objects from YML data
timeSlots = {tuple(ts.values())[0]: TimeSlot(*list(ts.values())[:-1]) for ts in supply_data_example['timeSlot']}

print("Time Slots: ")
for ts in timeSlots.values():
    print(ts)
print()

print("Corridors: ")
corridors = {}
for c in supply_data_example['corridor']:
    corridor_data = list(c.values())

    corridors[corridor_data[0]] = Corridor(corridor_data[0],
                                           corridor_data[1],
                                           list(filter(lambda s: s.id in corridor_data[2], stations.values())))

for c in corridors.values():
    print(c)
print()

print("Lines: ")
lines = {}
for line in supply_data_example['line']:
    line_data = list(line.values())
    timetable = {tuple(s.values())[0]: tuple(s.values())[1:] for s in line_data[3]}

    lines[line_data[0]] = Line(line_data[0],
                               line_data[1],
                               corridors[line_data[2]].id,
                               timetable)

for line in lines.values():
    print(line)
print()

print("Seats: ")
seats = {tuple(s.values())[0]: Seat(*s.values()) for s in supply_data_example['seat']}

for s in seats.values():
    print(s)
print()

print("Rolling Stock: ")
rolling_stock = {}
for rs in supply_data_example['rollingStock']:
    rs_data = list(rs.values())
    rs_seats = {tuple(s.values())[0]: tuple(s.values())[1] for s in rs_data[2]}

    rolling_stock[rs_data[0]] = RollingStock(rs_data[0],
                                             rs_data[1],
                                             rs_seats)

for rs in rolling_stock.values():
    print(rs)
print()

print("TSP: ")
tsp = {}
for op in supply_data_example['trainServiceProvider']:
    op_data = list(op.values())
    tsp[op_data[0]] = TSP(op_data[0], op_data[1], op_data[2])

for op in tsp.values():
    print(op)
print()

print("Services: ")
services = {}
for s in supply_data_example['service']:
    service_data = list(s.values())
    service_id, service_date = service_data[:2]
    service_line = lines[service_data[2]]
    service_tsp = tsp[service_data[3]]
    service_time_slot = timeSlots[service_data[4]]
    service_rs = rolling_stock[service_data[5]]
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

for service in services.values():
    print(service)
print()


my_travel = Supply(1, "BCN", "MAD", datetime.datetime(day=22, month=1, year=2023).date(), services.values())

print("My Travel: ")
for s in my_travel.services:
    print(s)

exit()

# Import data from database
database = None

# Define day and a pair of stations
day = datetime.datetime(day=15, month=2, year=2023)
print("Search day: ", day)

origin = 60000
destination = 71801

my_request = Supply(id_=1, date=day, w=(origin, destination))

# Get Services from Supply
