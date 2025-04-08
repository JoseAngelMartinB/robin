"""Constants for the scraping module."""

import importlib.resources

OUTPUT_SUPPLY_PATH = 'supply_data.yaml'
PRICES_COLUMNS = 8
RENFE_STATIONS_PATH = importlib.resources.files('robin.data').joinpath('adif_renfe_stations.csv')
TIME_SLOT_SIZE = 10
SPANISH_CORRIDOR_PATH = importlib.resources.files('robin.data').joinpath('spanish_corridor.yaml')
