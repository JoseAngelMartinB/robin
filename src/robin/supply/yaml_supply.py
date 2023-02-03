"""Supply entities from YAML data."""

from src.robin.supply.entities import Supply
import datetime

my_supply = Supply.from_yaml("../../../data/supply_data_example.yml")

print("Stations: ")
for s in my_supply.stations.values():
    print(s)

print("Corridors: ")
for c in my_supply.corridors.values():
    print(c)

print("Lines: ")
for ln in my_supply.lines.values():
    print(ln)

print("RollingStock: ")
for rs in my_supply.rollingStock.values():
    print(rs)

print("Seats: ")
for st in my_supply.seats.values():
    print(st)

print("TSPs: ")
for tsp in my_supply.tsps.values():
    print(tsp)

print("Services: ")
for s in my_supply.services.values():
    print(s)

print("Available stations: ")
for sid in my_supply.stations:
    print(f'Station ID: {sid} - Station name: {my_supply.stations[sid].name}')
print()

origin = "BCN"
destination = "MAD"
date = datetime.datetime(day=22, month=1, year=2023).date()

services = my_supply.generate('BCN', 'MAD', datetime.datetime(year=2023, month=1, day=22).date())

print(f'Travel options - Travel from {origin} to {destination} on {date}:')
for s in services:
    print(s)
