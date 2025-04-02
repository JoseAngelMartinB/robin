"""Exceptions for the scraping module."""


class NotAvailableStationsException(Exception):
    """Raised when no available stations are found in the Renfe website menu."""

    def __init__(self, *args, **kwargs):
        msg = 'There are no available stations in the Renfe website menu.'
        super().__init__(msg, *args, **kwargs)
