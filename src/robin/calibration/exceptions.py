"""Exceptions for the calibration module."""

class InvalidPenaltyFunction(Exception):
    """Raised when the given function is invalid to optimize."""
    
    def __init__(self, function_name: str, *args, **kwargs):
        msg = (f"The function '{function_name}' is invalid to optimize, "
               "only 'polynomial' is supported.")
        super().__init__(msg, *args, **kwargs)
