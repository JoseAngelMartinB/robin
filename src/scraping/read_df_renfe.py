from src.robin.supply.entities import *
from renfetools import *
import numpy as np
from ast import literal_eval
import os
import glob

# 0. Import data

# Get last updated file in selected folder
# updated_file = max(glob.iglob(f'../../datasets/scraping/renfe/trips/*.csv'), key=os.path.getmtime)

# File with Renfe data for february 2023 (AVE)
# E.g updated_file = '../../datasets/scraping/renfe/trips/trips_MADRI_BARCE_2022-12-30_2023-01-03.csv'
updated_file = f'../../datasets/scraping/renfe/trips/trips_MADRI_BARCE_2023-02-01_2023-02-28.csv'

# 0.1 Import trips
trips = pd.read_csv(updated_file, delimiter=',', dtype={'trip_id': str})

# E.g file_name = 'trips_MADRI_BARCE_2022-12-30_2023-01-03'
file_name = updated_file.split('/')[-1].split(".")[0]

# E.g file_name = 'MADRI_BARCE_2022-12-30_2023-01-03'
file_name = "_".join(file_name.split("_")[1:])

# 0.2 Import prices
prices = pd.read_csv(f'../../datasets/scraping/renfe/prices/prices_{file_name}.csv', delimiter=',')

# 0.3 Import stops
stop_times = pd.read_csv(f'../../datasets/scraping/renfe/stop_times/stopTimes_{file_name}.csv',
                         delimiter=',',
                         dtype={'stop_id': str})

# Get metadata from file name
# E.g. origin_id = 'MADRI', destination_id = 'BARCE', start_date = '2022-12-30', end_date = '2023-01-03'
origin_id, destination_id, start_date, end_date = file_name.split('_')

print(f"Origin:{origin_id} - Destionation:{destination_id}\nSince: {start_date}, Until: {end_date}")


def get_trip_price(service_id, prices):
    """
    Get trip price from prices dataframe

    :param
        service_id: string
        prices: dataframe with prices

    :return
        price: tuple of floats (three types of seats for Renfe AVE)
    """
    # Get price for service_id, If not found, return default price (Tuple of NaN values)
    try:
        price = prices[prices['service_id'] == service_id][['0', '1', '2']].values[0]
        price = tuple(price)
    except IndexError:
        price = tuple([float("NaN") for _ in range(3)])
    return price


trips['prices'] = trips['service_id'].apply(lambda x: get_trip_price(x, prices))

# Filter trips by price column to remove trips with any NaN value
trips = trips[trips['prices'].apply(lambda x: not any(np.isnan(x)))]

print("Head trips: ")
print(trips.head())

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
            corridor.insert(corridor.index(trip[i+1]), s)

# 1.5 Parse stations. Use Adif stop_id retrieve station info (name, lat, lon)
renfe_stations = pd.read_csv('../../datasets/scraping/renfe/renfe_stations.csv', delimiter=',', dtype={'stop_id': str})

# 1.6 Build dictionary of stations with stop_id as key and Station() object as value
stations = {}
for s in corridor:
    name = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
    city = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
    shortname = str(renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0])[:3].upper()
    coords = tuple(renfe_stations[renfe_stations['stop_id'] == s][['stop_lat', 'stop_lon']].values[0])

    stations[s] = Station(s, name, city, shortname, coords)

# 1.7 Build Corridor
first_station, last_station = tuple(stations.values())[::len(stations)-1]
corr_name = first_station.shortname + "-" + last_station.shortname
corrMadBar = Corridor(1, corr_name, list(stations.values()))

# Print stations in Corridor() object
print("Corridor: ")
print("Name: ", corrMadBar.name)
for s in corrMadBar.stations:
    print(s)


# Build Lines
def get_line(stops):
    line_data = {s: (a, d) for s, a, d in zip(stops['stop_id'], stops['arrival'], stops['departure'])}

    idx = stops['service_id'].values[0].split("_")[0]

    return Line(idx, f"Line {idx}", corrMadBar, line_data)


routes_lines = grouped_df.apply(lambda x: get_line(x))

print("Lines: ")
print(routes_lines.head())

# Build Dict of Services
"""
services[service_id] = Service(service_id,
                                   service_date,
                                   service_line,
                                   service_tsp,
                                   service_time_slot,
                                   service_rs,
                                   service_prices,
                                   service_capacity)
"""

#print(prices.head())
#print(stop_times.head())

exit()

# 1. Scrap trips range of days. DONE
# 2. Scrap prices same range of days. DONE
# 3. Load trips. DONE
# 4. For each trip: get prices --> Load prices file for day i, and get key "service_id"
# if not found, use default price dictionary with False values


# Train network to predict prices:
# Inputs: trip_id, day of week, (calendar info? type day (holidays?),today's date, date of trip
# Outputs: Prices for type of seats 0, 1, 2

# Decode dictionaries in the dataframe
converters = {'Train': literal_eval, 'Stops': literal_eval, 'Price': literal_eval}

# List of datetime cols to parse
datetime_cols = ['Departure', 'Arrival']

# Get filenames of csv files in datasets/ folder
path = 'datasets/'
filenames = [path + fn for fn in os.listdir('datasets/') if fn.endswith('.csv')]

# Get updated file
updated_file = max(filenames, key=os.path.getctime)


# Pandas read csv from file
df = pd.read_csv(updated_file, converters=converters, parse_dates=datetime_cols)

# Parse timedelta "h:m" string to timedelta object
df['Duration'] = df['Duration'].apply(lambda x: to_timedelta(x))

print(df.columns)
print(df.iloc[6])

# Stops column to nested list
stops = df['Stops'].apply(lambda d: list(d.keys())).values.tolist()

# Initialize corridor with max length trip
corridor = stops.pop(stops.index(max(stops, key=len)))

# Complete corridor with other stops that are not in the initial defined corridor
for trip in stops:
    for i, s in enumerate(trip):
        if s not in corridor:
            corridor.insert(corridor.index(trip[i-1])+1, s)

print(corridor)

stations = [Station(i, n, n[:3]) for i, n in enumerate(corridor)]

corrMadBar = Corridor(1, stations)

for s in corrMadBar.list_station:
    print(s)

# Build corridor
# Build types of trips
# Build schedules tables



