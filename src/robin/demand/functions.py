"""Functions for demand module."""

class Function:
    """
    Functions for demand module.

    Attributes:
        linear (staticmethod): Linear function.
    """

    @staticmethod
    def linear(x: float, b_0: float, b_1: float) -> float:
        """
        Linear function.

        Args:
            x (float): The x value.
            b_0 (float): The y-intercept.
            b_1 (float): The slope.

        Returns:
            float: The y value.
        """
        return b_0 + b_1 * x
