"""Exceptions for the supply module."""


class InvalidTimeStringFormat(Exception):
    """Raise when string format doesn't match HH:MM:SS format."""
    
    def __init__(self, time: str):
        msg = f'Invalid time format: {time}. Time must be a string in format HH:MM:SS'
        super().__init__(msg)


class InvalidDateStringFormat(Exception):
    """Raise when string format doesn't match YYYY-mm-dd format."""
    
    def __init__(self, date: str):
        msg = f'Invalid date format: {date}. Date must be a string in format YYYY-mm-dd'
        super().__init__(msg)
