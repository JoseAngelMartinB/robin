"""Constants for the scraping Renfe module."""

# Renfe URL's
MAIN_MENU_URL = 'https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html'
SCHEDULE_URL = 'https://horarios.renfe.com/HIRRenfeWeb/'

# Renfe stations CSV path
SAVE_PATH = 'data/renfe'
RENFE_STATIONS_CSV = f'{SAVE_PATH}/renfe_stations.csv'

# Allowed Renfe services for scraping
LR_RENFE_SERVICES = ('AVE', 'AVLO', 'AVE INT', 'ALVIA', 'AVANT')

# Default patience for scraping
DEFAULT_PATIENCE = 10

# Number of minutes in a day
ONE_DAY = 24 * 60
