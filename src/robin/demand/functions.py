"""Functions for demand module."""

from numpy.polynomial.polynomial import polyval
from typing import List, Union

class Function:
    """
    Functions for demand module.

    Attributes:
        polynomial (staticmethod): Polynomial function.
    """

    @staticmethod
    def polynomial(x: float, coeff: List[Union[int, float]]) -> float:
        """
        Polynomial function.

        Args:
            x (float): The x value.
            **kwargs (Mapping[str, float]): The coefficients of the polynomial.

        Returns:
            float: The y value.
        """
        # NOTE: Speed up the polynomial function.
        number_of_coeff = len(coeff)
        if number_of_coeff == 2:
            return coeff[0] + coeff[1] * x
        elif number_of_coeff == 3:
            return coeff[0] + coeff[1] * x + coeff[2] * x ** 2
        return polyval(x=x, c=coeff)
