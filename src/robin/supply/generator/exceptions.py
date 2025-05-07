"""Exceptions for the supply generator module."""


class ServiceInMultiplePathsException(Exception):
    """Raised when a service is in multiple paths."""
    
    def __init__(self, *args, **kwargs):
        msg = 'The service is in multiple paths.'
        super().__init__(msg, *args, **kwargs)


class ServiceWithConflicts(Exception):
    """Raised when a service is in multiple paths."""

    def __init__(self, *args, **kwargs):
        msg = 'The service has conflicts'
        super().__init__(msg, *args, **kwargs)


class UnfeasibleServiceException(Exception):
    """Raised when the generated service is not feasible."""
    
    def __init__(self, *args, **kwargs):
        msg = 'The generated service is not feasible.'
        super().__init__(msg, *args, **kwargs)
