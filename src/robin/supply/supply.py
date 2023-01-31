# Version 0 - Supply Module - 30/01/2023

from src.robin.supply.entities import *
from src.scraping.load_data import *
import pandas as pd

renfe_stations = pd.read_csv('data/scraping/renfe/renfe_stations.csv', delimiter=',', dtype={'stop_id': str})

# Load supply data from scraping
services, seats, corridor, tsp, rolling_stock = load_scraping(f'data/scraping/renfe/trips/trips_MADRI_BARCE_2023-02-01_2023-02-28.csv')

# Request user input: origin, destination, date
while True:
    origin = input("Origin: ")
    destination = input("Destination: ")

    try:
        date = datetime.datetime.strptime(input("Date (dd-mm-yyyy): "), "%d-%m-%Y").date()
    except ValueError:
        print("Invalid date")
        continue

    # Check if origin and destination are in the list of stations operated by Renfe
    if all(s in renfe_stations['stop_id'].values.tolist() for s in (origin, destination)) and origin != destination:
        break

dummy_service = services[0]
print("Service: ", dummy_service)
print("Line: ", dummy_service.line)

# Supply() obtains a List[Service] that match the user request (origin, destination, datetime.date)
my_travel = Supply(1, origin, destination, date, services)

print("Supply: ")

for s in my_travel.services:
    print(s)
