"""Utils for the demand module."""

from robin.demand.distributions import Distribution
from robin.demand.exceptions import InvalidDistributionException, InvalidContinuousDistributionException, InvalidDiscreteDistributionException, InvalidFunctionException
from robin.demand.functions import Function

from scipy import stats
from typing import Callable, Dict, Mapping, Tuple


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


def get_scipy_distribution(distribution_name: str, is_discrete: bool, **kwargs: Mapping[str, float]) -> Tuple[Callable, Dict[str, float]]:
    """
    Returns the distribution function from SciPy.

    Args:
        distribution_name (str): The distribution name.
        is_discrete (bool): Whether the distribution is discrete or not.
        **kwargs (Mapping[str, float]): The parameters of the distribution.

    Returns:
        Callable: The distribution function from SciPy.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
    """
    # Check if the distribution is a custom distribution
    distribution = getattr(Distribution, distribution_name, None)
    if distribution:
        return distribution(**kwargs), {}

    if is_discrete and distribution_name in stats._continuous_distns._distn_names: # type: ignore
        raise InvalidDiscreteDistributionException(distribution_name)
    elif not is_discrete and distribution_name in stats._discrete_distns._distn_names: # type: ignore
        raise InvalidContinuousDistributionException(distribution_name)
    
    distribution = getattr(stats, distribution_name, None)
    if not distribution:
        raise InvalidDistributionException(distribution_name)
    return distribution, kwargs
