"""Utils for the supply module."""

from src.robin.supply.exceptions import InvalidTimeStringFormat, InvalidDateStringFormat

import datetime
import re


def get_time(time) -> datetime.timedelta:
    """
    Function which returns a datetime.timedelta object from a string time in format HH:MM:SS.

    Args:
        time: String time in format HH:MM:SS.

    Returns:
        datetime.timedelta: Timedelta object.
    """
    r = re.compile('.*:.*:.*')

    if not all([r.match(time), len(time.split(':')) == 3, all(t.isdigit() for t in time.split(':'))]):
        raise InvalidTimeStringFormat(time)

    (h, m, s) = (int(t) for t in time.split(':'))
    if h not in range(0, 24) and m not in range(0, 60) and s not in range(0, 60):
        raise InvalidTimeStringFormat(time)

    return datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(s))


def get_date(date) -> datetime.date:
    """
    Function which returns a datetime.date object from a string date in format YYYYY-mm-dd.

    Args:
        date: String date in format YYYY-mm-dd.

    Returns:
        datetime.date: Date object.
    """
    r = re.compile('.*-.*-.*')

    if not all([r.match(date), len(date.split('-')) == 3, all(t.isdigit() for t in date.split('-'))]):
        raise InvalidDateStringFormat(date)

    y, m, d = (int(t) for t in date.split('-'))
    if d not in range(0, 32) and m not in range(1, 13) and y not in range(0, 9999):
        raise InvalidDateStringFormat(date)

    # Day could be out of range for month - if so, datetime will raise a ValueError Exception
    return datetime.datetime.strptime(date, '%Y-%m-%d').date()


def format_td(td: datetime.timedelta) -> str:
    """
    Format a datetime.timedelta object to a string in format HH:MM.

    Args:
        td (datetime.timedelta): Timedelta object.

    Returns:
        str: Formatted string.
    """
    seconds = td.seconds
    hours = str(seconds // 3600)
    minutes = str((seconds % 3600) // 60)

    if len(hours) == 1:
        hours = '0' + hours
    if len(minutes) == 1:
        minutes = '0' + minutes

    return f'{hours}:{minutes}'
