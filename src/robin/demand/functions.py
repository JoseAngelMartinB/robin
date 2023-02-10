"""Functions for demand module."""

class Function:
    """
    Functions for demand module.

    Attributes:
        linear (staticmethod): Linear function.
    """

    @staticmethod
    def linear(m: float, x: float, b: float) -> float:
        """
        Linear function.

        Args:
            m (float): The slope.
            x (float): The x value.
            b (float): The y-intercept.

        Returns:
            float: The y value.
        """
        return m * x + b
