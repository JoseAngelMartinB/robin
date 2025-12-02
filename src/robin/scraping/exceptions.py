"""Exceptions for the scraping module."""

from typing import List


class InvalidHardTypesException(Exception):
    """Raised when hard types in seat components are not present in seat quantity."""

    def __init__(self, missing_types: List[str], seat_quantity_keys: List[str], *args, **kwargs):
        msg = f'Hard types {missing_types} not found in seat quantity keys {seat_quantity_keys}'
        super().__init__(msg, *args, **kwargs)
