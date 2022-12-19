from renfetools import *

chrome_options = Options()
#chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")

# Renfe search menu
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

driver = webdriver.Chrome(options=chrome_options)

#driver.get(url)
#soup = BeautifulSoup(driver.page_source, 'html.parser')

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

# driver.get(url)

# Scraping with BeautifulSoup
# soup = BeautifulSoup(driver.page_source, 'html.parser')

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

# Retrieve date of search from the page (header)
init_date = get_date(soup)
print("Search url: ", url)
print("Date: ", init_date)

df = to_dataframe(soup, init_date, url)

date = init_date
for i in range(0):
    # Sum one day to date
    date += datetime.timedelta(days=1)

    # Get year, month and day from date as strings
    year, month, day = str(date).split("-")

    # Get day of week starting at sunday = 0
    weekday = date.weekday() + 1

    url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
    print("Search url: ", url)
    print("Date: ", date)

    # driver.get(url)

    # Pandas unable to scrap all data, so we use BeautifulSoup to get the rest
    # soup = BeautifulSoup(driver.page_source, 'html.parser')

    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    df = pd.concat([df, to_dataframe(soup, date, url)])

df = df.reset_index(drop=True)
df = df[['Train', 'Stops', 'Departure', 'Arrival', 'Duration', 'Price']]

print(df.describe(datetime_is_numeric=True))
print(df.columns)
print(df.iloc[-1])

# Save dataframe to csv in datasets folder
df.to_csv(f"datasets/{origin[:3].upper()}_{destination[:3].upper()}_{init_date}_{date}.csv", index=False)

# driver.close()
