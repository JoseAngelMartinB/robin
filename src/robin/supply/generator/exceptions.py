"""Exceptions for the supply generator module."""


class UnfeasibleServiceException(Exception):
    """Raised when the generated service is not feasible."""
    
    def __init__(self, *args, **kwargs):
        msg = 'The generated service is not feasible.'
        super().__init__(msg, *args, **kwargs)
