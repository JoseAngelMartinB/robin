"""Constants for the scraping Renfe module."""

# Renfe URL's
MAIN_MENU_URL = 'https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html'
PRICES_URL = 'https://venta.renfe.com/vol/buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional='
SCHEDULE_URL = 'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'

# Renfe stations CSV path
SAVE_PATH = 'data/renfe'
RENFE_STATIONS_CSV = f'{SAVE_PATH}/renfe_stations.csv'

# Allowed Renfe services for scraping
LR_RENFE_SERVICES = ('AVE', 'AVLO', 'AVE INT', 'ALVIA', 'AVANT')

# Default patience for scraping
DEFAULT_PATIENCE = 10

# Number of minutes in a day
ONE_DAY = 24 * 60
