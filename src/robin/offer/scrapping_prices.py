from bs4 import BeautifulSoup
from get_renfe_data import *
import pandas as pd
import requests
import json


stations = json.load(open('renfe_data/renfe.json'))['stations']
dstations = {e['station']: e['id'] for e in stations}


origin = 'Madrid (TODAS)'
destination = 'Barcelona (TODAS)'


origin_id = dstations[origin]
destination_id = dstations[destination]

url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF=2022&MF=MM&DF=DD&SF=NaN&ID=s'
print(url)

req = requests.get(url)

# Get table data using Pandas dataframe
df = pd.read_html(req.text, converters={'Salida': str, 'Llegada': str})[0]

soup = BeautifulSoup(req.text, 'html.parser')

date = get_date(soup)
print("Date: ", date)

prices = get_prices(soup)
#features = get_features(soup)

# parse float from dataframe column to time format

df['Salida'] = df['Salida'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))
df['Llegada'] = df['Llegada'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))
df['Duración'] = df['Llegada'] - df['Salida']
df['Precio desde*'] = prices

print(df.iloc[0])

