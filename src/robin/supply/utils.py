"""Utils for the supply module."""

from .exceptions import InvalidTimeStringFormat, InvalidDateStringFormat

import datetime
import re


def get_time(time):
    r = re.compile('.*:.*')

    if not all([r.match(time), len(time.split(":")) == 2, all(t.isdigit() for t in time.split(":"))]):
        raise InvalidTimeStringFormat(time)

    h, m = (int(t) for t in time.split(":"))
    if h not in range(0, 24) and m not in range(0, 60):
        raise InvalidTimeStringFormat(time)

    return datetime.timedelta(hours=int(h), minutes=int(m))


def get_date(date):
    r = re.compile('.*-.*-.*')

    if not all([r.match(date), len(date.split("-")) == 3, all(t.isdigit() for t in date.split("-"))]):
        raise InvalidDateStringFormat(date)

    d, m, y = (int(t) for t in date.split("-"))
    if d not in range(0, 366) and m not in range(1, 13) and y not in range(2020, 2100):
        raise InvalidDateStringFormat(date)

    return datetime.datetime.strptime(date, "%d-%m-%Y").date()
