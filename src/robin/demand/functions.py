"""Functions for demand module."""

from functools import cache
from numpy.polynomial.polynomial import polyval
from typing import Mapping


class Function:
    """
    Functions for demand module.

    Attributes:
        polynomial (staticmethod): Polynomial function.
    """

    @staticmethod
    @cache # NOTE: This can grow very large, take care about memory usage. (100 M can be a good limit)
    def polynomial(x: float, **kwargs: Mapping[str, float]) -> float:
        """
        Polynomial function.

        Args:
            x (float): The x value.
            **kwargs (Mapping[str, float]): The coefficients of the polynomial.

        Returns:
            float: The y value.
        """
        coeff = list(kwargs.values())
        return polyval(x=x, c=coeff)
