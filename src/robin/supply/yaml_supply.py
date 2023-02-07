"""Supply entities from YAML data."""

from src.robin.supply.entities import Supply
from src.robin.supply.utils import get_date

my_supply = Supply.from_yaml("../../../data/supply_data_example.yml")

print("Available stations: ")
for sid in my_supply.stations:
    print(f'Station ID: {sid} - Station name: {my_supply.stations[sid].name}')
print()

print("Available dates: ")
dates = [sv.date for sv in my_supply.services.values()]
print(f'From: {min(dates)} to {max(dates)}\n')

origin = input("Departure station ID: ")
destination = input("Arrival station ID: ")

date = get_date(input("Date (YYYY-MM-DD): "))

services = my_supply.generate(origin, destination, date)

print(f'Travel options - Travel from {origin} to {destination} on {date}:')
for s in services:
    print(s)
