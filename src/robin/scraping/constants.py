"""Constants for the scraping module."""

import importlib.resources

RENFE_STATIONS_PATH = importlib.resources.files('robin.data').joinpath('adif_renfe_stations.csv')
DEFAULT_SEAT_QUANTITY = {1: 250, 2: 50}
INFLATION = 1.0
