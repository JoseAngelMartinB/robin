"""Constants for the scraping module."""

import importlib.resources

# Default file path where the SupplySaver will save the YAML file
OUTPUT_SUPPLY_PATH = 'supply_data.yaml'

# Starting column number for the prices in the CSV prices file
PRICES_COLUMNS = 9

# Renfe stations CSV path
RENFE_STATIONS_PATH = importlib.resources.files('robin.data').joinpath('adif_renfe_stations.csv')

# Spanish corridor stations YAML path
SPANISH_CORRIDOR_PATH = importlib.resources.files('robin.data').joinpath('spanish_corridor.yaml')

# Default size of a time slot in minutes
TIME_SLOT_SIZE = 10
