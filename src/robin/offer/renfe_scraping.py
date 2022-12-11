from scraping.renfe import *
import pandas as pd
import datetime

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
assert all(s in stations.keys() for s in (origin, destination)), "Invalid origin or destination"

origin_id = stations[origin]
destination_id = stations[destination]

# Renfe schedules search
url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF=2022&MF=MM&DF=DD&SF=NaN&ID=s'

print("Search url: ", url)

req = requests.get(url)

# Pandas unable to scrap all data, so we use BeautifulSoup to get the rest
soup = BeautifulSoup(req.text, 'html.parser')

# Retrieve date of search from the page (header)
date = get_date(soup)

table = get_table(soup)
# print("Table: ", table)

df = pd.DataFrame(table, columns=['Train', 'Stops', 'Departure', 'Duration', 'Price'])
df = df[df["Train"].apply(lambda x: "AVE" in x)].reset_index(drop=True)


def format_time(x):
    h, m = filter(lambda t: is_number(t), x.split(" "))
    return datetime.timedelta(hours=int(h), minutes=int(m))


df['Duration'] = df['Duration'].apply(lambda x: format_time(x))
df['Departure'] = df['Departure'].apply(lambda x: datetime.datetime.strptime(str(date)+"-"+x, '%Y-%m-%d-%H.%M'))
df['Arrival'] = df['Departure'] + df['Duration']

print(df.iloc[-1])
