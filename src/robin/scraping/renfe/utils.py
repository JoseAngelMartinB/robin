""" Utility functions for scraping module"""

import re


def format_duration(x: str) -> int:
    """
    Convert string duration to minutes.

    Args:
        x (str): String duration.

    Returns:
        int: Duration in minutes.
    """
    tuple_hour_min = tuple(filter(lambda t: is_number(t), x.split(' ')))
    if len(tuple_hour_min) == 1:
        m = tuple_hour_min[0]
        return int(m)
    h, m = tuple_hour_min
    return int(h) * 60 + int(m)


def is_number(x: str) -> bool:
    """
    Function to check if a string is a number.

    Args:
        x (str): String to check.

    Returns:
        bool: True if x is a number, False otherwise.
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def remove_blanks(x: str, replace_by: str = '') -> str:
    """
    Removes blank spaces (tabs, newlines, etc.) from a string.

    Args:
        x (str): String to remove blank spaces from.
        replace_by (str): String to replace the blank spaces with.

    Returns:
        str: String without blank spaces.
    """
    return re.sub(r'\s+', replace_by, x)


def time_to_minutes(time: str) -> int:
    """
    Convert time string formated as 'HH.MM' to minutes.

    Args:
        time (str): Time string formated as 'HH.MM'.

    Returns:
        int: Time in minutes.
    """
    try:
        h, m = time.split('.')
    except ValueError:
        return 0
    return int(h) * 60 + int(m)
