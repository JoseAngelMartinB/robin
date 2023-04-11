from src.scraping.renfe.renfetools import get_stations
from src.scraping.renfe.gtfs.renfegtfstools import *
from bs4 import BeautifulSoup
import requests

# 1. Retrieve stations from Renfe website

# Renfe search menu
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)
print("Available stations: ", stations)

# 2. Retrieve stations from GTFS

# GTFS file
savepath = 'renfe_data/'
url = 'https://ssl.renfe.com/gtransit/Fichero_AV_LD/google_transit.zip'

filename = download_data(url, savepath)

renfe_gtfs = import_datasets(savepath)

# Get stations as dataframe from GTFS
stations_gtfs = renfe_gtfs['stops']

stations_gtfs_dict = {id_: name for id_, name in zip(stations_gtfs['stop_id'], stations_gtfs['stop_name'])}

# Search specific name in GTFS stations
# Used this to manually find stations that does not match by id (e.g. 'MADRI' vs 60000)
# DO NOT DELETE
"""
for id, name in stations_gtfs_dict.items():
    if "Madrid".lower() in name.lower():
        print(id, name)
"""

# Initialize master dict with stations that match by id between Renfe web and GTFS
master_dict = {}
count = 0
for id_, name in stations.items():
    try:
        int(id_)
    except ValueError:
        pass

    if id_ in stations_gtfs_dict.keys():
        master_dict[id_] = (stations_gtfs_dict[id_], str(id_))

print("Stations that match between Renfe web and GTFS data: ", len(master_dict))

other_dict = {'60000': ('Madrid-Puerta de Atocha', 'MADRI'),
              '10500': ('Medina del Campo', 'MEDIN'),
              '31200': ('Puebla de Sanabria', 'PUEBL'),
              '02002': ('Puente Genil-Herrera', 'PTE G'),
              '22300': ('Redondela', 'REDON'),
              '30100': ('Salamanca', 'SALAM'),
              '70101': ('San Fernando Henares', 'S FER'),
              '14223': ('Santander', 'SANTA'),
              '12100': ('Segovia', 'SEGOV'),
              '71500': ('Tarragona', 'TARRA'),
              '65000': ('Valencia-Estacio del Nord', 'VALEN'),
              '08223': ('Vigo Urzaiz', 'VIGO-'),
              '04040': ('Zaragoza-Delicias', 'ZARAG'),
              '02003': ('Antequera-Santa Ana', 'ANTEQ'),
              '71801': ('Barcelona-Sants', 'BARCE'),
              '13200': ('Bilbao-Abando Indalecio Prieto', 'BILBA'),
              '66100': ('Cuenca', 'CUENC'),
              '15410': ('Gijon', 'GIJON'),
              '70200': ('Guadalajara', 'GUADA'),
              '80100': ('Pamplona/irunya', 'IRUN-')}

# Add missing stations
master_dict.update(other_dict)

# Add latitude and longitude info to master dict
df = renfe_gtfs['stops']
for id_ in master_dict:
    lat = df[df['stop_id'] == id_]['stop_lat'].values[0]
    lon = df[df['stop_id'] == id_]['stop_lon'].values[0]
    master_dict[id_] = list(master_dict[id_]) + [lat, lon]

print(master_dict)
print("Total stations: ", len(master_dict))

stop_id = master_dict.keys()
stop_name, renfe_id, lat, lon = zip(*tuple(master_dict.values()))

df = pd.DataFrame(list(zip(stop_id, stop_name, renfe_id, lat, lon)),
                  columns=['stop_id', 'stop_name', 'renfe_id', 'stop_lat', 'stop_lon'])

df.loc[-1] = ['00000', 'Unknown', '00000', 0.0, 0.0]
df.index = df.index + 1  # shifting index
df.sort_index(inplace=True)

print(df.head())

# Save to csv
df.to_csv('../../../data/scraping/renfe/renfe_stations.csv', index=False)
