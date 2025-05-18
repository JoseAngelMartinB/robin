"""Exceptions for the plotter module."""


class NoFileProvided(Exception):
    """Raised when no file path is provided."""

    def __init__(self, *args, **kwargs):
        msg = 'No file path provided. Please provide at least supply or output file path.'
        super().__init__(msg, *args, **kwargs)
