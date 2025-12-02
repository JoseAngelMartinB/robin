"""Exceptions for the supply generator module."""


class ServiceWithConflicts(Exception):
    """Raised when a service has conflicts with existing services."""

    def __init__(self, *args, **kwargs):
        msg = 'The service has conflicts with existing services.'
        super().__init__(msg, *args, **kwargs)


class UnfeasibleServiceException(Exception):
    """Raised when the generated service is not feasible."""
    
    def __init__(self, *args, **kwargs):
        msg = 'The generated service is not feasible.'
        super().__init__(msg, *args, **kwargs)
