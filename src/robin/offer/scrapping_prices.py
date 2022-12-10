from bs4 import BeautifulSoup
from get_renfe_data import *
import pandas as pd
import requests

# Renfe search menu
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)
print("Available stations: ", stations)

# Set origin and destination
origin = 'Madrid (TODAS)'
destination = 'Barcelona (TODAS)'

# Get origin and destination id's to use in the search
origin_id = stations[origin]
destination_id = stations[destination]

# Renfe schedules search
url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF=2022&MF=MM&DF=DD&SF=NaN&ID=s'

print("Search url: ", url)

req = requests.get(url)

# Get table of schedules using Pandas dataframe
df = pd.read_html(req.text, converters={'Salida': str, 'Llegada': str})[0]

# Pandas unable to scrap all data, so we use BeautifulSoup to get the rest
soup = BeautifulSoup(req.text, 'html.parser')

# Retrieve date of search from the page
date = get_date(soup)
print("Date: ", date)

# Retrieve prices from the page.
prices = get_prices(soup)

# features = get_features(soup)
df['Tren / Recorrido'] = df['Tren / Recorrido'].apply(lambda x: tuple(x.split(' ')))

# Parse string dates to datetime objects
df['Salida'] = df['Salida'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))
df['Llegada'] = df['Llegada'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))

# Get duration as time delta from arrival and departure columns
df['Duración'] = df['Llegada'] - df['Salida']

# Add prices
df['Precio desde*'] = prices

# TODO: Deal with combined trips
df = df[df["Tren / Recorrido"].apply(lambda x: not any("LD" in v for v in x))]

df['Paradas'] = get_stops(soup).values() # get_stops returns a dictionary of stops

# Print dataframe
print(df.iloc[-1])
