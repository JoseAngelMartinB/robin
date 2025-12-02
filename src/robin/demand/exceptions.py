"""Exceptions for the demand module."""

from robin.demand.functions import Function


class InvalidDistributionException(Exception):
    """Raised when the given distribution is not contained in SciPy."""
    
    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not contained in SciPy. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html')
        super().__init__(msg, *args, **kwargs)


class InvalidContinuousDistributionException(Exception):
    """Raised when the given distribution is not a continuous distribution."""

    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not a continuous distribution. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions')
        super().__init__(msg, *args, **kwargs)


class InvalidDiscreteDistributionException(Exception):
    """Raised when the given distribution is not a discrete distribution."""
    
    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not a discrete distribution. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html#discrete-distributions')
        super().__init__(msg, *args, **kwargs)


class InvalidForbiddenDepartureHoursException(Exception):
    """Raised when the given forbidden departure hours are not valid."""
    
    def __init__(self, forbidden_departure_hours: tuple, *args, **kwargs):
        msg = (f"The forbidden departure hours '{forbidden_departure_hours}' are not valid. "
               'They must be a ordered tuple of two integers between 0 and 24.')
        super().__init__(msg, *args, **kwargs)


class InvalidFunctionException(Exception):
    """Raised when the given function is not contained in the ROBIN module."""
    
    def __init__(self, function_name: str, *args, **kwargs):
        functions = [member for member in dir(Function) if member[0] != '_']
        msg = (f"The function '{function_name}' is not contained in the ROBIN module. "
               f'Available functions are: {functions}.')
        super().__init__(msg, *args, **kwargs)
