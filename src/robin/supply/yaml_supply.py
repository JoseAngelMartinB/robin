"""Supply entities from YAML data."""

from src.robin.supply.entities import Supply
import datetime

my_supply = Supply.from_yaml("../../../data/supply_data_example.yml")

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
