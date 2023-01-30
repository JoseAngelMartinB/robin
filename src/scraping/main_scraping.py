from scraping_trips import renfe_scraping_trips
from scraping_prices import renfe_scraping_prices
from renfetools import *
import datetime

# URL Renfe drowpdown menu from station i to station j
# Use this URL to get the list of stations operated by Renfe
url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)

# Set origin and destination using GTFS station ids
origin = 60000  # Madrid Puerta de Atocha - Use station master id (from GTFS)
destination = 71801  # Barcelona Sants - Use station master id (from GTFS)

"""
Read csv of stations operated by Renfe from parallel directory

This csv contains the following columns:
stop_id, stop_name, renfe_id, stop_lat, stop_lon

stop_id: GTFS stop_id
stop_name: GTFS stop_name
renfe_id: Renfe station id - Should be the same as the stop_id, but not always
stop_lat: Station latitude
stop_lon: Station longitude
"""

df = pd.read_csv('../../data/scraping/renfe/renfe_stations.csv')

# Get Renfe stations ids from previous csv
origin_id = df[df['stop_id'] == origin]['renfe_id'].values[0]
destination_id = df[df['stop_id'] == destination]['renfe_id'].values[0]

# Assert that the origin and destination stations are in the list of stations operated by Renfe
assert all(s in stations.keys() for s in (origin_id, destination_id)), "Invalid origin or destination"

# Get today's date
# date = datetime.date.today()
# date += datetime.timedelta(days=1)

# string "dd/mm/yyyy" to datetime.date
date = datetime.datetime.strptime("01/02/2023", "%d/%m/%Y").date()
range_days = 28

renfe_scraping_trips(origin_id, destination_id, date, range_days)

renfe_scraping_prices(origin_id, destination_id, date, range_days)
