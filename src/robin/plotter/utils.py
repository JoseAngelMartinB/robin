"""Utils for the plotter module."""

import datetime


def get_purchase_date(purchase_day: str) -> datetime.date:
    """
    Calculate purchase date from anticipation and arrival day.

    Args:
        purchase_day (str): Purchase day in 'YYYY-MM-DD' format.

    Returns:
        datetime.date: Purchase date as datetime.date object.
    """
    return datetime.datetime.strptime(purchase_day, "%Y-%m-%d").date()
