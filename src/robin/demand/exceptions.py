"""Exceptions for the demand module."""

class InvalidDistributionException(Exception):
    """Raised when the given distribution is not contained in SciPy."""
    pass


class InvalidContinuousDistributionException(Exception):
    """Raised when the given distribution is not a continuous distribution."""
    pass


class InvalidDiscreteDistributionException(Exception):
    """Raised when the given distribution is not a discrete distribution."""
    pass
