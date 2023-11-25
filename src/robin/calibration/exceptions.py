"""Exceptions for the calibration module."""


class InvalidArrivalTimeDistribution(Exception):
    """Raised when the given distribution is invalid to optimize."""
    
    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is invalid to optimize, "
               "only 'custom_arrival_time' is supported.")
        super().__init__(msg, *args, **kwargs)
