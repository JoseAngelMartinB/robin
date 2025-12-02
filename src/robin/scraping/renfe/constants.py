"""Constants for the scraping Renfe module."""

import importlib.resources

# Allowed Renfe services for scraping
ALLOWED_RENFE_SERVICES = ('AVE', 'AVLO', 'AVE INT', 'ALVIA', 'AVANT')

# Default patience of the driver when waiting for an element to load
DEFAULT_PATIENCE = 20

# Renfe main menu URL (where it is listed the stations names)
MAIN_MENU_URL = 'https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html'

# Maximum number of attempts for the Renfe scraping
MAX_ATTEMPTS = 3

# Number of minutes in a day
ONE_DAY = 24 * 60

# Renfe prices URL
PRICES_URL = 'https://venta.renfe.com/vol/buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional='

# Name of the Renfe TSP
RENFE_TSP = 'Renfe'

# Renfe stations CSV path
RENFE_STATIONS_CSV = importlib.resources.files('robin.data').joinpath('adif_renfe_stations.csv')

# Default directory for saving the scraped CSV files
SAVE_PATH = 'data/renfe'

# Renfe schedule URL
SCHEDULE_URL = 'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
