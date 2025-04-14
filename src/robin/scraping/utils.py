"""Utils for the scraping module."""

import datetime


def timedelta_to_str(time_delta: datetime.timedelta) -> str:
    """
    Convert a timedelta to a string formated 'HH.MM'.

    Args:
        time_delta: The timedelta object to convert.

    Returns:
        str: A string representation of the timedelta in 'HH.MM' format.
    """
    hours = time_delta.total_seconds() // 3600
    minutes = (time_delta.total_seconds() % 3600) / 60
    return f'{int(hours):02d}.{int(minutes):02d}'
