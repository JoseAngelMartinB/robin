from scraping.renfe import *
import pandas as pd

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

# Get table of schedules using Pandas dataframe. Parse 'Arrival' and 'Departure' columns as strings
df = pd.read_html(req.text, converters={'Salida': str, 'Llegada': str})[0]

# TODO: Code optimization
"""
Function get_prices and get_stops both iter over the schedules table
Possible optimization by merging both functions
"""

# Pandas unable to scrap all data, so we use BeautifulSoup to get the rest
soup = BeautifulSoup(req.text, 'html.parser')

# Retrieve date of search from the page (header)
date = get_date(soup)
print("Date: ", date)

# features = get_features(soup)
df['Tren / Recorrido'] = df['Tren / Recorrido'].apply(lambda x: tuple(x.split(' ')))

# Parse string dates to datetime objects
df['Salida'] = df['Salida'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))

# TODO: Date could not be correct if train departs day i and arrives day i+1
df['Llegada'] = df['Llegada'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))

# Get duration as time delta from arrival and departure columns
df['Duración'] = df['Llegada'] - df['Salida']

# Add prices
df['Precio desde*'] = get_prices(soup)

# TODO: Deal with combined trips
df = df[df["Tren / Recorrido"].apply(lambda x: not any("LD" in v for v in x))]

# Add column with every stop in the trip
df['Paradas'] = get_stops(soup).values()

# Print dataframe
print(df.iloc[-1])
