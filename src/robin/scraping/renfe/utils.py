""" Utility functions for scraping module"""

import datetime


def time_to_datetime(time: str, date: datetime.date) -> datetime.datetime:
    """
    Converts a time string with format 'HH:MM h' to a datetime object with format '%Y-%m-%d %H:%M'.

    Args:
        time (str): Time string.
        date (datetime.date): Date of the time.

    Returns:
        datetime.datetime: Datetime object of the time formatted as '%Y-%m-%d %H:%M'.
    """
    time = time.replace(' h', '')
    return datetime.datetime.strptime(str(date) + ' ' + time, '%Y-%m-%d %H:%M')


def time_to_minutes(time: str, separator: str = '.') -> int:
    """
    Convert time string formated as 'HH.MM' to minutes.

    Args:
        time (str): Time string formated as 'HH.MM'.
        separator (str): Separator between hours and minutes.

    Returns:
        int: Time in minutes.
    """
    try:
        h, m = time.split(separator)
    except ValueError:
        return 0
    return int(h) * 60 + int(m)
