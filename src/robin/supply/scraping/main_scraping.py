from scraping_trips import renfe_scraping_trips
from scraping_prices import renfe_scraping_prices
from renfetools import *
import datetime

url = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"

req = requests.get(url)
soup = BeautifulSoup(req.text, 'html.parser')

stations = get_stations(soup)

# Set origin and destination
origin = 60000
destination = 71801

# Read csv from parallel directory
df = pd.read_csv('datasets/renfe_stations.csv')

origin_id = df[df['stop_id'] == origin]['renfe_id'].values[0]
destination_id = df[df['stop_id'] == destination]['renfe_id'].values[0]

# Get origin and destination id's to use in the search
assert all(s in stations.keys() for s in (origin_id, destination_id)), "Invalid origin or destination"

# Get today's date
date = datetime.date.today()
date += datetime.timedelta(days=1)
range_days = 150

# renfe_scraping_trips(origin_id, destination_id, date, range_days)

renfe_scraping_prices(origin_id, destination_id, date, range_days)
