from renfetools import *

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

chrome_options = Options()
chrome_options.add_argument("--disable-extensions")
#chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--headless")

# Renfe search menu
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

# Parse today date to string with format %d-%m-%Y
date_str = date.strftime("%d-%m-%Y")

# Renfe search for tickets from origin to destination on date
# Schedules table is loaded dynamically with javascript, so we need to use selenium!!!

root = "https://venta.renfe.com/vol/"
search_url = f"buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional="

url = root + search_url
print(url)

driver = webdriver.Chrome(options=chrome_options)

driver.get(url)

delay = 10  # seconds
try:
    myElem = WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'trayectoRow')))
    print("Page is ready!")
except TimeoutException:
    print("Loading took too much time!")

soup = BeautifulSoup(driver.page_source, 'html.parser')

table = soup.find("div", {"class": "tab-content"})

count = 0
trains = []
for row in table.find_all("tr"):
    if isTrain(row):
        # trip_id, train_type, origin_id, destination_id, date_service, departure_time
        trains.append(get_train(row, origin_id, destination_id, date_str))
    else:
        continue
    count += 1

for train in trains:
    print(train)

print("Total trains: ", count)
driver.close()

df = pd.DataFrame.from_records(trains, columns=['trip_id', 'train_type', 'arrival', 'departure', 'duration', 'prices'])
df["service_id"] = df.apply(lambda x: x["trip_id"] + "_" + x["departure"].strftime("%d-%m-%Y-%H.%M"), axis=1)

df = df.reset_index(drop=True)
df = df[['service_id', 'trip_id', 'train_type', 'arrival', 'departure', 'duration', 'prices']]

print(df.head())
print(df.iloc[-1])

import numpy as np

prices = dict(zip(df.service_id, df.prices))
np.save(f"datasets/prices_{origin_id[:3].upper()}_{destination_id[:3].upper()}_{date_str}.npy", prices)

