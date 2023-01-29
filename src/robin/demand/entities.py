"""Entities for the demand module."""

from .exceptions import InvalidForbiddenDepartureHoursException
from .utils import get_function, get_scipy_distribution

from typing import Dict, Mapping, Union, Tuple

import datetime
import numpy as np


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
        """
        Initialize a market.

        Args:
            id_ (int): The market id.
            departure_station (Station): The departure station.
            arrival_station (Station): The arrival station.
        """
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
        arrival_time (Callable): The arrival time distribution function.
        arrival_time_kwargs (Mapping[str, Union[int, float]]): The arrival time distribution parameters.
        purchase_day (Callable): The purchase day distribution function.
        purchase_day_kwargs (Mapping[str, Union[int, float]]): The purchase day distribution named parameters.
        forbidden_departure_hours (Tuple[int, int]): The forbidden departure hours.
        seats (Dict[int, float]): The utility of the seats.
        penalty_arrival_time (Callable): The penalty function for the arrival time.
        penalty_arrival_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_departure_time (Callable): The penalty function for the departure time.
        penalty_departure_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_cost (Callable): The penalty function for the cost.
        penalty_cost_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_traveling_time (Callable): The penalty function for the travel time.
        penalty_traveling_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        error (Callable): The error distribution function.
        error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.

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
        """
        Initialize a user pattern.

        Args:
            id_ (int): The user pattern id.
            arrival_time (str): The arrival time distribution name.
            arrival_time_kwargs (Mapping[str, Union[int, float]]): The arrival time distribution named parameters.
            purchase_day (str): The purchase day distribution name.
            purchase_day_kwargs (Mapping[str, Union[int, float]]): The purchase day distribution named parameters.
            forbidden_departure_hours (Tuple[int, int]): The forbidden departure hours.
            seats (Dict[int, float]): The utility of the seats.
            penalty_arrival_time (str): The penalty function name for the arrival time.
            penalty_arrival_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_departure_time (str): The penalty function name for the departure time.
            penalty_departure_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_cost (str): The penalty function name for the cost.
            penalty_cost_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_traveling_time (str): The penalty function name for the travel time.
            penalty_traveling_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            error (str): The error distribution name.
            error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.
        
        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
            InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
            InvalidFunctionException: Raised when the given function is not contained in the ROBIN module.
        """
        self.id = id_
        self.arrival_time = get_scipy_distribution(distribution_name=arrival_time, is_discrete=False)
        self.arrival_time_kwargs = arrival_time_kwargs
        self.purchase_day = get_scipy_distribution(distribution_name=purchase_day, is_discrete=True)
        self.purchase_day_kwargs = purchase_day_kwargs
        self.forbidden_departure_hours = self._check_forbidden_departure_hours(forbidden_departure_hours=forbidden_departure_hours)
        self.seats = seats
        self.penalty_arrival_time = get_function(function_name=penalty_arrival_time)
        self.penalty_arrival_time_kwargs = penalty_arrival_time_kwargs
        self.penalty_departure_time = get_function(function_name=penalty_departure_time)
        self.penalty_departure_time_kwargs = penalty_departure_time_kwargs
        self.penalty_cost = get_function(function_name=penalty_cost)
        self.penalty_cost_kwargs = penalty_cost_kwargs
        self.penalty_traveling_time = get_function(function_name=penalty_traveling_time)
        self.penalty_traveling_time_kwargs = penalty_traveling_time_kwargs
        self.error = get_scipy_distribution(distribution_name=error, is_discrete=False)
        self.error_kwargs = error_kwargs

    def _check_forbidden_departure_hours(self, forbidden_departure_hours: Tuple[int, int]) -> Tuple[int, int]:
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
    
    def get_forbidden_departure_hours(self) -> Tuple[int, int]:
        """
        Returns the forbidden departure hours.

        Returns:
            Tuple[int, int]: The forbidden departure hours.
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
    """
    A demand pattern is determined by the potential demand and the distribution of user patterns.

    Attributes:
        id (int): The demand pattern id.
        potential_demand(Callable): The potential demand distribution function.
        potential_demand_kwargs (Mapping[str, Union[int, float]]): The potential demand distribution named parameters.
        user_pattern_distribution (Dict[int, float]): The user pattern distribution.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        ValueError: Raised when the user pattern distribution does not sum up to 1.
    """
    
    def __init__(
            self,
            id_: int,
            potential_demand: str,
            potential_demand_kwargs: Mapping[str, Union[int, float]],
            user_pattern_distribution: Dict[int, float]
        ) -> None:
        """
        Initializes a demand pattern.

        Args:
            id_ (int): The demand pattern id.
            potential_demand (str): The potential demand distribution name.
            potential_demand_kwargs (dict): The potential demand distribution named parameters.
            user_pattern_distribution (dict): The user pattern distribution.

        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        """
        self.id = id_
        self.potential_demand = get_scipy_distribution(distribution_name=potential_demand, is_discrete=True)
        self.potential_demand_kwargs = potential_demand_kwargs
        self.user_pattern_distribution = user_pattern_distribution

    def get_potential_demand(self) -> int:
        """
        Returns the potential demand.

        Returns:
            int: The potential demand.
        """
        return self.potential_demand.rvs(**self.potential_demand_kwargs)
    
    def get_user_pattern(self) -> int:
        """
        Samples a user pattern according to the user pattern distribution.

        Returns:
            int: The user pattern.

        Raises:
            ValueError: Raised when the user pattern distribution does not sum up to 1.
        """
        return np.random.choice(list(self.user_pattern_distribution.keys()), p=list(self.user_pattern_distribution.values()))


class Day:
    """
    A day is described as its actual date and demand pattern.
    
    Attributes:
        id (int): The day id.
        date (datetime.date): The actual date.
        demand_pattern (DemandPattern): The associated demand pattern.
    """
    
    def __init__(self, id_: int, date: datetime.date, demand_pattern: DemandPattern) -> None:
        """
        Initializes a day.

        Args:
            id_ (int): The day id.
            date (datetime.date): The actual date.
            demand_pattern (DemandPattern): The associated demand pattern.
        """
        self.id = id_
        self.date = date
        self.demand_pattern = demand_pattern

    def get_date(self) -> datetime.date:
        """
        Returns the date.

        Returns:
            datetime.date: The date.
        """
        return self.date
    
    def get_demand_pattern(self) -> DemandPattern:
        """
        Returns the demand pattern.

        Returns:
            DemandPattern: The demand pattern.
        """
        return self.demand_pattern


class Passenger:
    """A passenger is defined by his/her user pattern, the origin-destination pair, the desired day and time of arrival and the day of purchase."""
    pass
