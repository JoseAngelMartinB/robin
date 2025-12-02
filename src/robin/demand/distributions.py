"""Custom distributions for the demand module."""

from scipy import stats
from typing import Callable, Mapping


class Distribution:
    """
    Custom distributions for the demand module.

    Attributes:
        hourly (staticmethod): Custom arrival time discrete distribution given the probability of each hour.
    """

    @staticmethod
    def hourly(**kwargs: Mapping[str, float]) -> Callable:
        """
        Custom arrival time discrete distribution given the probability of each hour.
        
        Args:
            **kwargs (Mapping[str, float]): The probability of each hour.

        Returns:
            Callable: The custom arrival time discrete distribution.
            
        Raises:
            ValueError: Raised when the sum of the probabilities is not 1.
        """   
        hours = list(kwargs.keys())
        assert len(hours) == 24, 'The number of hours must be 24.'
        assert all([int(hour) == index for index, hour in enumerate(hours)]), 'The hours must be in order from 0 to 23.'
        hours = list(map(int, hours))
        probabilities = list(kwargs.values())
        return stats.rv_discrete(name='hourly', values=(hours, probabilities))
