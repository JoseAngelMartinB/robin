""" Utility functions for scraping module"""

import datetime


def time_str_to_minutes(duration_str: str) -> int:
    """
    Convert a duration string formatted as 'X h. Y min.' to total minutes.

    Args:
        duration_str (str): Duration string formatted as 'X h. Y min.'.

    Returns:
        int: Total duration in minutes.
    """
    # Split the string into parts.
    parts = duration_str.strip().split()

    hours = 0
    minutes = 0

    # Loop through the parts to find hours and minutes.
    for i, token in enumerate(parts):
        if token == "h.":
            if i > 0:
                hours = int(parts[i - 1])
        elif token == "min.":
            if i > 0:
                minutes = int(parts[i - 1])

    return hours * 60 + minutes


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


