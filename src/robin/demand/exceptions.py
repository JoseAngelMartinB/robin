"""Exceptions for the demand module."""

class InvalidDistributionException(Exception):
    """Raised when the given distribution is not contained in SciPy."""
    
    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not contained in SciPy. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html')
        super().__init__(msg, *args, **kwargs)


class InvalidContinuousDistributionException(Exception):
    """Raised when the given distribution is not a continuous distribution."""

    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not a discrete distribution. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html#discrete-distributions')
        super().__init__(msg, *args, **kwargs)


class InvalidDiscreteDistributionException(Exception):
    """Raised when the given distribution is not a discrete distribution."""
    
    def __init__(self, distribution_name: str, *args, **kwargs):
        msg = (f"The distribution '{distribution_name}' is not a continuous distribution. "
               'See details at: https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions')
        super().__init__(msg, *args, **kwargs)
