"""Exceptions for the supply module."""


class InvalidTimeStringFormat(Exception):
    """Raise when string format doesn't match HH:MM format."""
    def __init__(self, time: str):
        msg = (f"Invalid time format: {time}. "
               'Time must be a string in format HH:SS')
        # super() inherits from Exception class
        super().__init__(msg)


class InvalidDateStringFormat(Exception):
    """Raise when string format doesn't match dd-mm-YYYY format."""
    def __init__(self, date: str):
        msg = (f"Invalid date format: {date}. "
               'Date must be a string in format dd-mm-YYYY')
        super().__init__(msg)
