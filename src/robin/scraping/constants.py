"""Constants for the scraping module."""

import importlib.resources

PRICES_COLUMNS = 9
"""Starting column number for the prices in the CSV prices file."""

RENFE_STATIONS_PATH = importlib.resources.files('robin.data').joinpath('adif_renfe_stations.csv')
"""Renfe stations CSV path."""

SPANISH_CORRIDOR_PATH = importlib.resources.files('robin.data').joinpath('spanish_corridor.yaml')
"""Spanish corridor stations YAML path."""

TIME_SLOT_SIZE = 10
"""Default size of a time slot in minutes."""
