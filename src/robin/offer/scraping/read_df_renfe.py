from renfetools import *
from ast import literal_eval

# Decode dictionaries in the dataframe
converters = {'Train': literal_eval, 'Stops': literal_eval, 'Price': literal_eval}

# List of datetime cols to parse
datetime_cols = ['Departure', 'Arrival']

# Pandas read csv from file
df = pd.read_csv('datasets/MAD_BAR_2022-12-13_2022-12-15.csv', converters=converters, parse_dates=datetime_cols)

# Parse timedelta "h:m" string to timedelta object
df['Duration'] = df['Duration'].apply(lambda x: to_timedelta(x))

print(df.columns)
print(df.iloc[6])

# Stops column to nested list
stops = df['Stops'].apply(lambda d: list(d.keys())).values.tolist()

# Define blacklist to remove words from stop name
blacklist = ["PTA.", "PUERTA", "CAMP", "DE"]

# Remove blacklist words from stop name and get first word of each stop name
stops = [[list(filter(lambda w: w not in blacklist, re.split(r'\W+', s)))[0] for s in trip] for trip in stops]

# Initialize corridor with max length trip
corridor = stops.pop(stops.index(max(stops, key=len)))

# Complete corridor with other stops that are not in the initial defined corridor
for trip in stops:
    for i, s in enumerate(trip):
        if s not in corridor:
            corridor.insert(corridor.index(trip[i-1])+1, s)

print(corridor)

# Build corridor
# Build types of trips
# Build schedules tables



