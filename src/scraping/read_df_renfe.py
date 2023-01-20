from src.robin.supply.entities import *
from renfetools import *
import numpy as np
from ast import literal_eval
import os
import glob

updated_file = max(glob.iglob(f'../../datasets/scraping/renfe/trips/*.csv'), key=os.path.getmtime)

trips = pd.read_csv(updated_file, delimiter=',')

# E.g updated_file = 'datasets/trips/trips_MADRI_BARCE_2022-12-30_2023-01-03.csv'

file_name = updated_file.split('/')[-1].split(".")[0]
# E.g file_name = 'trips_MADRI_BARCE_2022-12-30_2023-01-03'

file_name = "_".join(file_name.split("_")[1:])
# E.g file_name = 'MADRI_BARCE_2022-12-30_2023-01-03'

prices = pd.read_csv(f'../../datasets/scraping/renfe/prices/prices_{file_name}.csv', delimiter=',')
stop_times = pd.read_csv(f'../../datasets/scraping/renfe/stop_times/stopTimes_{file_name}.csv', delimiter=',')

origin_id, destination_id, departure, arrival = file_name.split('_')
# E.g. origin_id = 'MADRI', destination_id = 'BARCE', departure = '2022-12-30', arrival = '2023-01-03'

print(origin_id, destination_id, departure, arrival)

# prices = np.load("datasets/prices/prices_MAD_BAR_29-12-2022.npy", allow_pickle=True)

print(trips.head())
print(prices.head())
print(stop_times.head())

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



