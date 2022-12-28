from renfetools import *

# Renfe search menu
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)

# Set origin and destination
origin = 60000  # Madrid Puerta de Atocha
destination = 71801  # Barcelona Sants

# Read csv from parallel directory
df = pd.read_csv('datasets/renfe_stations.csv')

origin_id = df[df['stop_id'] == origin]['renfe_id'].values[0]
destination_id = df[df['stop_id'] == destination]['renfe_id'].values[0]

# Get origin and destination id's to use in the search
assert all(s in stations.keys() for s in (origin_id, destination_id)), "Invalid origin or destination"

# Get today's date
date = datetime.date.today()
date += datetime.timedelta(days=1)

# TODO: Consider saving dicts in independent files using npy format or csv
# File with: trip_id, price1, price2, price3
# File with: trip_id, sequence of stops

for i in range(1):
    # Get year, month and day from date as strings
    year, month, day = str(date).split("-")

    # Get day of week starting at sunday = 0
    weekday = date.weekday() + 1

    url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
    print("Search url: ", url)
    print("Date: ", date)

    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    df = pd.concat([df, to_dataframe(soup, date, url)])

    # Sum one day to date
    date += datetime.timedelta(days=1)

df = df.reset_index(drop=True)
df = df[['service_id', 'trip_id', 'train_type', 'stops', 'departure', 'duration', 'price']]

print(df.describe(datetime_is_numeric=True))
print(df.columns)
print(df.iloc[-1])

# Save dataframe to csv in datasets folder
# df.to_csv(f"datasets/{origin_id[:3].upper()}_{destination_id[:3].upper()}_{init_date}_{date}.csv", index=False)

