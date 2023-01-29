"""Utils for demand module."""

from .exceptions import InvalidDistributionException, InvalidContinuousDistributionException, InvalidDiscreteDistributionException, InvalidFunctionException
from .functions import Function

from scipy import stats
from typing import Callable


def get_function(function_name: str) -> Callable:
    """
    Returns the function from the given name.

    Args:
        function_name (str): The function name.

    Returns:
        Callable: The function from the given name.
    """
    function = getattr(Function, function_name, None)
    if not function:
        raise InvalidFunctionException(function_name)
    return function


def get_scipy_distribution(distribution_name: str, is_discrete: bool) -> Callable:
    """
    Returns the distribution function from SciPy.

    Args:
        distribution_name (str): The distribution name.
        is_discrete (bool): Whether the distribution is discrete or not.

    Returns:
        Callable: The distribution function from SciPy.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
    """
    if is_discrete and distribution_name in stats._continuous_distns._distn_names:
        raise InvalidDiscreteDistributionException(distribution_name)
    elif not is_discrete and distribution_name in stats._discrete_distns._distn_names:
        raise InvalidContinuousDistributionException(distribution_name)
    
    distribution = getattr(stats, distribution_name, None)
    if not distribution:
        raise InvalidDistributionException(distribution_name)
    return distribution
