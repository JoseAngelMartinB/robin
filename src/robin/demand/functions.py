"""Functions for demand module."""

import numpy as np

from functools import cache
from typing import Mapping


class Function:
    """
    Functions for demand module.

    Attributes:
        polynomial (staticmethod): Polynomial function.
    """

    @staticmethod
    @cache
    def polynomial(x: float, **kwargs: Mapping[str, float]) -> float:
        """
        Polynomial function.

        Args:
            x (float): The x value.
            **kwargs (Mapping[str, float]): The coefficients of the polynomial.

        Returns:
            float: The y value.
        """
        reverse_sorted = dict(sorted(kwargs.items(), key=lambda item: item[1], reverse=True))
        coeff = list(reverse_sorted.values())
        return np.polyval(x=x, p=coeff)
