from bs4 import BeautifulSoup
import requests
import datetime
import json
import re

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

stations = json.load(open('renfe_data/renfe.json'))['stations']

dstations = {}
for e in stations:
    dstations[e['station']] = e['id']

origin = 'Madrid (TODAS)'
destination = 'Barcelona (TODAS)'

if origin not in dstations.keys():
    raise ValueError('Origin station not found')
if destination not in dstations.keys():
    raise ValueError('Destination station not found')

origin_id = dstations[origin]
destination_id = dstations[destination]

url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF=2022&MF=MM&DF=DD&SF=NaN&ID=s'
print(url)

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('div', {'class': 'irf-travellers-table__container-time'})

p = table.find_all('span', {'class': 'irf-travellers-table__txt irf-travellers-table__txt--bold'})

date = None

months = {'Enero': 1,
          'Febrero': 2,
          'Marzo': 3,
          'Abril': 4,
          'Mayo': 5,
          'Junio': 6,
          'Julio': 7,
          'Agosto': 8,
          'Septiembre': 9,
          'Octubre': 10,
          'Noviembre': 11,
          'Diciembre': 12}

date = []
for i in p:
    s = i.find('script')

    if s is not None:

        s = s.text.split('\'')[1]

        if s in months.keys():
            print(months[s])
            date.append(str(months[s]))
        if is_number(i.text):
            x = re.sub(r'\s+', '', i.text)
            print(x)
            date.append(x)

print(date)
date = '-'.join(date)

# Date to datetima object day, month and year
date = datetime.datetime.strptime(date, '%d-%m-%Y').date()
print(date)
