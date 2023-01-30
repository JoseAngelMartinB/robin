import pandas as pd

from src.robin.supply.entities import *
from src.scraping.renfetools import *
import numpy as np
import os
import glob


# Aux functions
def get_trip_price(service_id: str, price_df: pd.DataFrame):
    """
    Get trip price from prices dataframe

    Args:
        service_id: string
        price_df: dataframe with prices

    Returns:
        price: tuple of floats (three types of seats for Renfe AVE)
    """
    # Get price for service_id, If not found, return default price (Tuple of NaN values)
    try:
        price = price_df[price_df['service_id'] == service_id][['0', '1', '2']].values[0]
        price = tuple(price)
    except IndexError:
        price = tuple([float("NaN") for _ in range(3)])
    return price


def get_line(stops: pd.DataFrame, corr: Corridor):
    """
    Get line from stops dataframe
    Args:
        stops: dataframe with stops
        corr: Corridor() object
    Returns:
         Line() object
    """
    line_data = {s: (a, d) for s, a, d in zip(stops['stop_id'], stops['arrival'], stops['departure'])}

    idx = stops['service_id'].values[0].split("_")[0]

    return Line(idx, f"Line {idx}", corr, line_data)


def get_trip_line(service_id: str, lines: dict):
    """
    Get trip line from set_lines dictionary

    Args:
        service_id: string.
        lines: dictionary with lines.

    Returns:
        line: Line() object.
    """
    try:
        line = lines[service_id.split("_")[0]]
    except KeyError:
        raise KeyError(f"Line not found for service_id: {service_id}")

    return line


def get_service(service_id: str, departure: str, arrival: str, price: tuple, line: Line, tsp, rs):
    """
    Get Service() object from Renfe data

    Args:
        service_id: string
        departure: string
        arrival: string
        price: tuple of floats
        line: Line() object

    Returns:
        Service() object
    """
    id_ = service_id
    date = departure.split(" ")[0]
    departure = departure.split(" ")[1][:-3]
    arrival = arrival.split(" ")[1][:-3]
    line = line
    time_slot = TimeSlot(id_.split("_")[0], departure, arrival)

    return Service(id_,
                   date,
                   line,
                   tsp,
                   time_slot,
                   rs,
                   price,
                   "Train")


def load_scraping(file_path):
    # 0. Import data

    # Get last updated file in selected folder
    # updated_file = max(glob.iglob(f'../../datasets/scraping/renfe/trips/*.csv'), key=os.path.getmtime)

    # File with Renfe data for february 2023 (AVE)
    # E.g updated_file = '../../datasets/scraping/renfe/trips/trips_MADRI_BARCE_2022-12-30_2023-01-03.csv'
    updated_file = file_path

    # 0.1 Import trips
    trips = pd.read_csv(updated_file, delimiter=',', dtype={'trip_id': str})

    # E.g file_name = 'trips_MADRI_BARCE_2022-12-30_2023-01-03'
    file_name = updated_file.split('/')[-1].split(".")[0]

    # E.g file_name = 'MADRI_BARCE_2022-12-30_2023-01-03'
    file_name = "_".join(file_name.split("_")[1:])

    # 0.2 Import prices
    path = file_path.split("/")

    prices = pd.read_csv(f'{"/".join(path[:path.index("datasets")])}/datasets/scraping/renfe/prices/prices_{file_name}.csv', delimiter=',')

    # 0.3 Import stops
    stop_times = pd.read_csv(f'{"/".join(path[:path.index("datasets")])}/datasets/scraping/renfe/stop_times/stopTimes_{file_name}.csv',
                             delimiter=',',
                             dtype={'stop_id': str})

    # Get metadata from file name
    # E.g. origin_id = 'MADRI', destination_id = 'BARCE', start_date = '2022-12-30', end_date = '2023-01-03'
    origin_id, destination_id, start_date, end_date = file_name.split('_')

    print(f"Origin:{origin_id} - Destionation:{destination_id}\nSince: {start_date}, Until: {end_date}")

    trips['prices'] = trips['service_id'].apply(lambda x: get_trip_price(x, prices))

    # Filter trips by price column to remove trips with any NaN value
    trips = trips[trips['prices'].apply(lambda x: not any(np.isnan(x)))]

    # 1. Build Corridor MAD-BAR

    # 1.1 Group dataframe by trip_id
    grouped_df = stop_times.groupby('service_id')

    # 1.2 Get nested list with stops for each trip
    list_stations = grouped_df.apply(lambda d: list(d['stop_id'])).values.tolist()

    # 1.3 Initialize corridor with max length trip
    corridor = list_stations.pop(list_stations.index(max(list_stations, key=len)))

    # 1.4 Complete corridor with other stops that are not in the initial defined corridor
    for trip in list_stations:
        for i, s in enumerate(trip):
            if s not in corridor:
                corridor.insert(corridor.index(trip[i + 1]), s)

    # 1.5 Parse stations. Use Adif stop_id retrieve station info (name, lat, lon)
    renfe_stations = pd.read_csv(f'{"/".join(path[:path.index("datasets")])}/datasets/scraping/renfe/renfe_stations.csv', delimiter=',', dtype={'stop_id': str})

    # 1.6 Build dictionary of stations with stop_id as key and Station() object as value
    stations = {}
    for s in corridor:
        name = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
        city = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
        shortname = str(renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0])[:3].upper()
        coords = tuple(renfe_stations[renfe_stations['stop_id'] == s][['stop_lat', 'stop_lon']].values[0])

        stations[s] = Station(s, name, city, shortname, coords)

    # 1.7 Build Corridor
    first_station, last_station = tuple(stations.values())[::len(stations) - 1]
    corr_name = first_station.shortname + "-" + last_station.shortname
    corrMadBar = Corridor(1, corr_name, list(stations.values()))

    # 2. Build Lines
    routes_lines = grouped_df.apply(lambda x: get_line(x, corrMadBar))

    set_lines = {line.id: line for line in list(set(routes_lines.values.tolist()))}

    trips['lines'] = trips['service_id'].apply(lambda x: get_trip_line(x, set_lines))

    # 3. Build seats for Renfe AVE
    renfe_seats = (Seat(1, "Turista", 1, 1), Seat(2, "Turista Plus", 1, 2), Seat(3, "Preferente", 2, 1))
    renfe_seats = {s.id: s for s in renfe_seats}

    # 4. Build RollingStock for Renfe AVE
    renfe_rs = [RollingStock(1, "S-114", {1: 250, 2: 50})]

    # 5. Build TSP for Renfe
    renfe_tsp = TSP(1, "Renfe", renfe_rs)

    # 6. Build Services
    trips['service'] = trips.apply(lambda x: get_service(x['service_id'],
                                                         x['departure'],
                                                         x['arrival'],
                                                         x['prices'],
                                                         x['lines'],
                                                         renfe_tsp,
                                                         renfe_rs[0]),
                                   axis=1)

    return trips['service'].values.tolist()


if __name__ == '__main__':
    file_path = '../../datasets/scraping/renfe/trips/trips_MADRI_BARCE_2023-02-01_2023-02-28.csv'
    services = load_scraping(file_path)
    print(services[0])
