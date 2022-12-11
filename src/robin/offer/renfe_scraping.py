from scraping.renfe import *
import pandas as pd
import datetime


def format_time(x):
    """ Function receives "x", a string with time formatted as "2 h. 30 m." and returns a timedelta object """
    h, m = filter(lambda t: is_number(t), x.split(" "))
    return datetime.timedelta(hours=int(h), minutes=int(m))


def to_dataframe(s, d):
    table = get_table(s)

    dfs = pd.DataFrame(table, columns=['Train', 'Stops', 'Departure', 'Duration', 'Price'])
    dfs = dfs[dfs["Train"].apply(lambda x: "AVE" in x)].reset_index(drop=True)

    dfs['Duration'] = dfs['Duration'].apply(lambda x: format_time(x))
    dfs['Departure'] = dfs['Departure'].apply(lambda x: datetime.datetime.strptime(str(d) + "-" + x, '%Y-%m-%d-%H.%M'))
    dfs['Arrival'] = dfs['Departure'] + dfs['Duration']

    return dfs


# Renfe search menu
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)
print("Available stations: ", stations)

names = tuple(list(stations.keys())[1:])

pairs = [(x, y) for x in names for y in names if x != y]

# TODO:
# - For each pair of stations get the schedules table (filtered, only AVE)
# - Set limit of days to search
# - Concatenate dataframes
# - Save dataframe to csv

# Set origin and destination
origin = 'Madrid (TODAS)'
destination = 'Barcelona (TODAS)'

# Get origin and destination id's to use in the search
assert all(s in stations.keys() for s in (origin, destination)), "Invalid origin or destination"

origin_id = stations[origin]
destination_id = stations[destination]

# Renfe schedules search
url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF=2022&MF=MM&DF=DD&SF=NaN&ID=s'

req = requests.get(url)

# Scraping with BeautifulSoup
soup = BeautifulSoup(req.text, 'html.parser')

# Retrieve date of search from the page (header)
init_date = get_date(soup)
print("Search url: ", url)
print("Date: ", init_date)

df = to_dataframe(soup, init_date)

date = init_date
for i in range(2):
    # Sum one day to date
    date += datetime.timedelta(days=1)

    # Get year, month and day from date as strings
    year, month, day = str(date).split("-")

    # Get day of week starting at sunday = 0
    weekday = date.weekday() + 1

    url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
    print("Search url: ", url)
    print("Date: ", date)

    req = requests.get(url)

    # Pandas unable to scrap all data, so we use BeautifulSoup to get the rest
    soup = BeautifulSoup(req.text, 'html.parser')

    df = pd.concat([df, to_dataframe(soup, date)])

df = df.reset_index(drop=True)
df = df[['Train', 'Stops', 'Departure', 'Arrival', 'Duration', 'Price']]

print(df.describe())
print(df.columns)
print(df.iloc[-1])

# Save dataframe to csv in datasets folder
df.to_csv(f"{origin}_{destination}_{init_date}_{date}.csv", index=False)
