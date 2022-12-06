from getRenfeData import *
from entities import *

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

# Dict of Station object - Every Station serviced by Renfe
stations = {int(s): Station(int(s), stopsDict[s][0], str(str(s)[:3].upper()), stopsDict[s][1]) for s in stopsDict}

# Get stop_times dataset
df = renfe_schedules['stop_times']

# Group dataframe by trip_id
grouped_df = df.groupby('trip_id')

# Dictionary keys: trip_id, value: list of stop_ids
routesSeries = grouped_df.apply(lambda g: tuple(g['stop_id']))

stopTimes = renfe_schedules['stop_times']

# org_id = 17000
# des_id = 79400
# Filter routesSeries that start at org_id and end at des_id
# get_trip(org_id, dest_id, stopsDict, routesSeries, df, stations, False)

org_id = 60000  # Madrid-Puerta de Atocha
des_id = 71801  # Barcelona-Sants
get_trip(org_id, des_id, stopsDict, routesSeries, stopTimes, stations, False)

