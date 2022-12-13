from src.robin.offer.gtfs.renfegtfstools import *

savepath = 'renfe_data/'
url = 'https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip'

filename = download_data(url, savepath)

renfe_schedules = import_datasets(savepath)

print(renfe_schedules.keys())

stop_times = renfe_schedules['stop_times']
stops = renfe_schedules['stops']

org_id = 60000  # Madrid-Puerta de Atocha
des_id = 71801  # Barcelona-Sants

get_trip(org_id, des_id, stop_times, stops, False)

# org_id = 17000
# des_id = 79400
# Filter routesSeries that start at org_id and end at des_id
# get_trip(org_id, dest_id, stopsDict, routesSeries, df, stations, False)



