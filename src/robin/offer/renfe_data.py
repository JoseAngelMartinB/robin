from getRenfeData import *
from entities import *
from tqdm import tqdm # Progress Bar

savepath = 'renfe_data/'
url = 'https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip'

filename = download_data(url, savepath)

renfe_schedules = import_datasets(savepath)

print(renfe_schedules.keys())

# Parse date of service to format: Day/Month/Year
renfe_schedules['calendar_dates'].date = renfe_schedules['calendar_dates'].date.apply(parse_date)

# Filter - High speed and long distance routes
s = {'ALVIA', 'AVE', 'AVLO'}
renfe_schedules['routes'] = renfe_schedules['routes'][renfe_schedules['routes']['route_short_name'].isin(s)]

# Filter - High speed and long distance trips
s = {'GL', 'AV', 'LC'}
renfe_schedules['trips'] = renfe_schedules['trips'][renfe_schedules['trips']['route_id'].str.contains('|'.join(s))]

df = renfe_schedules['stops']
df['stop_id'] = df['stop_id'].astype(int)
df['stop_lat'] = df['stop_lat'].astype(float)
df['stop_lon'] = df['stop_lon'].astype(float)

# Dictionary
# Keys: stop id
# Values: (stop name, (lon coords, lat coords))
stopsDict = dict(zip(df.stop_id, zip(df.stop_name, zip(df.stop_lon, df.stop_lat))))

# List of Station object - Every Station serviced by Renfe
stations = [Station(int(s), stopsDict[s][0], str(str(s)[:3].upper()), stopsDict[s][1]) for s in stopsDict]

# Get stop_times dataset
df = renfe_schedules['stop_times']

# Get set of trip_id's
set_trip_ids = tuple(set(df['trip_id'].values))

# Dict of routes
# Key: trip_id
# Value: sorted tuple of stop_id's in trip
routes = {}

for tid in tqdm(set_trip_ids):
    df_lock = df.loc[df['trip_id'] == tid]
    routes[tid] = tuple(df_lock['stop_id'].values)

# Find longest route
longest_trip = max(routes.values(), key=lambda v: len(v))

# Get coordinates of each station in the longest trip
stopsCoords = [stopsDict[s][1] for s in longest_trip]
stopsCoords.append(stopsDict[longest_trip[-1]][1])

# Plot longest Route
plot_route(stopsCoords)






