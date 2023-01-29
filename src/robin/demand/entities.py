from .exceptions import *
from .functions import Function

from scipy import stats
from typing import Callable, Dict, Mapping, Union, Tuple


class Station:
    """
    Dummy class for a station.
    
    NOTE: This class is not yet implemented. It is just a placeholder.
    """
    pass


class Market:
    """
    A market is composed by the departure and arrival stations.

    Attributes:
        id (int): The market id.
        departure_station (Station): The departure station.
        arrival_station (Station): The arrival station.
    """

    def __init__(self, id_: int, departure_station: Station, arrival_station: Station) -> None:
        self.id = id_
        self.departure_station = departure_station
        self.arrival_station = arrival_station

    def get_departure_station(self) -> Station:
        """
        Get the departure station.

        Returns:
            Station: The departure station.
        """
        return self.departure_station
    
    def get_arrival_station(self) -> Station:
        """
        Get the arrival station.

        Returns:
            Station: The arrival station.
        """
        return self.arrival_station


class UserPattern:
    """
    A user pattern is a set of random variables and penalties that define it (e.g. business or student).

    Attributes:
        id (int): The user pattern id.
        arrival_time (str): The arrival time distribution name.
        arrival_time_kwargs (dict): The arrival time distribution parameters.
        purchase_day (str): The purchase day distribution name.
        purchase_day_kwargs (dict): The purchase day distribution named parameters.
        forbidden_departure_hours (tuple): The forbidden departure hours.
        seats (dict): The utility of the seats.
        penalty_arrival_time (str): The penalty function name for the arrival time.
        penalty_arrival_time_kwargs (dict): The penalty function named parameters.
        penalty_departure_time (str): The penalty function name for the departure time.
        penalty_departure_time_kwargs (dict): The penalty function named parameters.
        penalty_cost (str): The penalty function name for the cost.
        penalty_cost_kwargs (dict): The penalty function named parameters.
        penalty_traveling_time (str): The penalty function name for the travel time.
        penalty_traveling_time_kwargs (dict): The penalty function named parameters.
        error (str): The error distribution name.
        error_kwargs (dict): The error distribution named parameters.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
        InvalidFunctionException: Raised when the given function is not contained in the ROBIN module.
    """
    
    def __init__(
            self,
            id_: int,
            arrival_time: str,
            arrival_time_kwargs: Mapping[str, Union[int, float]],
            purchase_day: str,
            purchase_day_kwargs: Mapping[str, Union[int, float]],
            forbidden_departure_hours: Tuple[int, int],
            seats: Dict[int, float],
            penalty_arrival_time: str,
            penalty_arrival_time_kwargs: Mapping[str, Union[int, float]],
            penalty_departure_time: str,
            penalty_departure_time_kwargs: Mapping[str, Union[int, float]],
            penalty_cost: str,
            penalty_cost_kwargs: Mapping[str, Union[int, float]],
            penalty_traveling_time: str,
            penalty_traveling_time_kwargs: Mapping[str, Union[int, float]],
            error: str,
            error_kwargs: Mapping[str, Union[int, float]]
        ) -> None:
        self.id = id_
        self.arrival_time = self._get_scipy_distribution(distribution_name=arrival_time, is_discrete=False)
        self.arrival_time_kwargs = arrival_time_kwargs
        self.purchase_day = self._get_scipy_distribution(distribution_name=purchase_day, is_discrete=True)
        self.purchase_day_kwargs = purchase_day_kwargs
        self.forbidden_departure_hours = self._check_forbidden_departure_hours(forbidden_departure_hours=forbidden_departure_hours)
        self.seats = seats
        self.penalty_arrival_time = self._get_function(function_name=penalty_arrival_time)
        self.penalty_arrival_time_kwargs = penalty_arrival_time_kwargs
        self.penalty_departure_time = self._get_function(function_name=penalty_departure_time)
        self.penalty_departure_time_kwargs = penalty_departure_time_kwargs
        self.penalty_cost = self._get_function(function_name=penalty_cost)
        self.penalty_cost_kwargs = penalty_cost_kwargs
        self.penalty_traveling_time = self._get_function(function_name=penalty_traveling_time)
        self.penalty_traveling_time_kwargs = penalty_traveling_time_kwargs
        self.error = self._get_scipy_distribution(distribution_name=error, is_discrete=False)
        self.error_kwargs = error_kwargs

    def _check_forbidden_departure_hours(self, forbidden_departure_hours: tuple) -> tuple:
        """
        Checks if the given forbidden departure hours are valid.

        Args:
            forbidden_departure_hours (tuple): The forbidden departure hours.

        Returns:
            tuple: The forbidden departure hours.

        Raises:
            InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
        """
        if len(forbidden_departure_hours) != 2:
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        elif not all(isinstance(hour, int) for hour in forbidden_departure_hours):
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        elif any(hour < 0 or hour > 23 for hour in forbidden_departure_hours):
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        elif forbidden_departure_hours[0] >= forbidden_departure_hours[1]:
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        return forbidden_departure_hours

    def _get_function(self, function_name: str) -> Callable:
        """
        Returns the function from the given name.

        Args:
            function_name (str): The function name.

        Returns:
            Callable: The function from the given name.
        """
        function = getattr(Function, function_name, None)
        if not function:
            raise InvalidFunctionException(function_name)
        return function

    def _get_scipy_distribution(self, distribution_name: str, is_discrete: bool) -> Callable:
        """
        Returns the distribution function from SciPy.

        Args:
            distribution_name (str): The distribution name.
            is_discrete (bool): Whether the distribution is discrete or not.

        Returns:
            Callable: The distribution function from SciPy.

        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        """
        if is_discrete and distribution_name in stats._continuous_distns._distn_names:
            raise InvalidDiscreteDistributionException(distribution_name)
        elif not is_discrete and distribution_name in stats._discrete_distns._distn_names:
            raise InvalidContinuousDistributionException(distribution_name)
        
        distribution = getattr(stats, distribution_name, None)
        if not distribution:
            raise InvalidDistributionException(distribution_name)
        return distribution
    
    def get_arrival_time(self) -> float:
        """
        Returns the arrival time.

        Returns:
            float: The arrival time.
        """
        return self.arrival_time.rvs(**self.arrival_time_kwargs)
    
    def get_purchase_day(self) -> int:
        """
        Returns the purchase day.

        Returns:
            int: The purchase day.
        """
        return self.purchase_day.rvs(**self.purchase_day_kwargs)
    
    def get_seat_utility(self, seat: int) -> float:
        """
        Returns the utility of the given seat.

        Args:
            seat (int): The seat.

        Returns:
            float: The utility of the given seat.
        """
        return self.seats.get(seat, 0)
    
    def get_forbidden_departure_hours(self) -> tuple:
        """
        Returns the forbidden departure hours.

        Returns:
            tuple: The forbidden departure hours.
        """
        return self.forbidden_departure_hours
    
    def get_penalty_arrival_time(self) -> float:
        """
        Returns the penalty for the arrival time.

        Returns:
            float: The penalty for the arrival time.
        """
        return self.penalty_arrival_time(**self.penalty_arrival_time_kwargs)
    
    def get_penalty_departure_time(self) -> float:
        """
        Returns the penalty for the departure time.

        Returns:
            float: The penalty for the departure time.
        """
        return self.penalty_departure_time(**self.penalty_departure_time_kwargs)
    
    def get_penalty_cost(self) -> float:
        """
        Returns the penalty for the cost.

        Returns:
            float: The penalty for the cost.
        """
        return self.penalty_cost(**self.penalty_cost_kwargs)
    
    def get_penalty_traveling_time(self) -> float:
        """
        Returns the penalty for the traveling time.

        Returns:
            float: The penalty for the traveling time.
        """
        return self.penalty_traveling_time(**self.penalty_traveling_time_kwargs)
    
    def get_error(self) -> float:
        """
        Returns the error.

        Returns:
            float: The error.
        """
        return self.error.rvs(**self.error_kwargs)


class DemandPattern:
    """A demand pattern is determined by the potential demand and the distribution of user patterns."""
    pass


class Day:
    """A day is described as its actual date and demand pattern."""
    pass


class Passenger:
    """A passenger is defined by his/her user pattern, the origin-destination pair, the desired day and time of arrival and the day of purchase."""
    pass
