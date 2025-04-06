"""Entities for the demand module."""

import datetime
import numpy as np
import yaml

from robin.demand.constants import DEFAULT_SEAT_UTILITY, DEFAULT_TSP_UTILITY, DEFAULT_RVS_SIZE
from robin.demand.exceptions import InvalidForbiddenDepartureHoursException
from robin.demand.utils import get_function, get_scipy_distribution

from pathlib import Path
from typing import Any, List, Mapping, Union, Tuple


class Market:
    """
    A market is composed by the departure and arrival stations.

    Attributes:
        id (int): The market id.
        departure_station (str): The departure station id.
        arrival_station (str): The arrival station id.
    """

    def __init__(self, id: int, departure_station: str, arrival_station: str) -> None:
        """
        Initialize a market.

        Args:
            id (int): The market id.
            departure_station (Station): The departure station id.
            arrival_station (Station): The arrival station id.
        """
        self.id = id
        self.departure_station = departure_station
        self.arrival_station = arrival_station
    
    def __str__(self) -> str:
        """
        Returns a human readable string representation of the market.

        Returns:
            str: A human readable string representation of the market.
        """
        return f'{self.departure_station} - {self.arrival_station}'

    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the market.

        Returns:
            str: The debuggable string representation of the market.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'departure_station={self.departure_station}, '
            f'arrival_station={self.arrival_station})'
        )


class UserPattern:
    """
    A user pattern is a set of random variables and penalties that define it (e.g. business or student).

    Attributes:
        id (int): The user pattern id.
        name(str): The user pattern name.
        arrival_time (Callable): The arrival time distribution function.
        arrival_time_kwargs (Mapping[str, Union[int, float]]): The arrival time distribution parameters.
        purchase_day (Callable): The purchase day distribution function.
        purchase_day_kwargs (Mapping[str, Union[int, float]]): The purchase day distribution named parameters.
        forbidden_departure_hours (Tuple[int, int]): The forbidden departure hours.
        seats (Mapping[int, float]): The utility of the seats.
        penalty_arrival_time (Callable): The penalty function for the arrival time.
        penalty_arrival_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_departure_time (Callable): The penalty function for the departure time.
        penalty_departure_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_cost (Callable): The penalty function for the cost.
        penalty_cost_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        penalty_travel_time (Callable): The penalty function for the travel time.
        penalty_travel_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        error (Callable): The error distribution function.
        error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.
        default_seat_utility (float): The default utility of the seats.
        default_rvs_size (int): The default size of the random variables sample.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
        InvalidFunctionException: Raised when the given function is not contained in the ROBIN module.
    """
    
    def __init__(
        self,
        id: int,
        name: str,
        arrival_time: str,
        arrival_time_kwargs: Mapping[str, Union[int, float]],
        purchase_day: str,
        purchase_day_kwargs: Mapping[str, Union[int, float]],
        forbidden_departure_hours: Tuple[int, int],
        seats: Mapping[int, float],
        tsps: Mapping[int, float],
        penalty_arrival_time: str,
        penalty_arrival_time_kwargs: Mapping[str, Union[int, float]],
        penalty_departure_time: str,
        penalty_departure_time_kwargs: Mapping[str, Union[int, float]],
        penalty_cost: str,
        penalty_cost_kwargs: Mapping[str, Union[int, float]],
        penalty_travel_time: str,
        penalty_travel_time_kwargs: Mapping[str, Union[int, float]],
        error: str,
        error_kwargs: Mapping[str, Union[int, float]],
        default_seat_utility: float = DEFAULT_SEAT_UTILITY,
        default_tsp_utility: float = DEFAULT_TSP_UTILITY,
        default_rvs_size: int = DEFAULT_RVS_SIZE
    ) -> None:
        """
        Initialize a user pattern.

        Args:
            id (int): The user pattern id.
            name(str): The user pattern name.
            arrival_time (str): The arrival time distribution name.
            arrival_time_kwargs (Mapping[str, Union[int, float]]): The arrival time distribution named parameters.
            purchase_day (str): The purchase day distribution name.
            purchase_day_kwargs (Mapping[str, Union[int, float]]): The purchase day distribution named parameters.
            forbidden_departure_hours (Tuple[int, int]): The forbidden departure hours.
            seats (Mapping[int, float]): The utility of the seats.
            tsps (Mapping[int, float]): The utility of the train service providers.
            penalty_arrival_time (str): The penalty function name for the arrival time.
            penalty_arrival_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_departure_time (str): The penalty function name for the departure time.
            penalty_departure_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_cost (str): The penalty function name for the cost.
            penalty_cost_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            penalty_travel_time (str): The penalty function name for the travel time.
            penalty_travel_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            error (str): The error distribution name.
            error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.
            default_seat_utility (float, optional): The default utility of the seats.
            default_tsp_utility (float, optional): The default utility of the train service providers.
            default_rvs_size (int, optional): The default size of the random variables sample.
        
        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
            InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
            InvalidFunctionException: Raised when the given function is not contained in the ROBIN module.
        """
        self.id = id
        self.name = name
        self._arrival_time, self.arrival_time_kwargs = get_scipy_distribution(
            distribution_name=arrival_time, is_discrete=False, **arrival_time_kwargs
        )
        self._arrival_time_rvs = None
        self._arrival_time_rvs_idx = 0
        self._purchase_day, self.purchase_day_kwargs = get_scipy_distribution(
            distribution_name=purchase_day, is_discrete=True, **purchase_day_kwargs
        )
        self._purchase_day_rvs = None
        self._purchase_day_rvs_idx = 0
        self.forbidden_departure_hours = self._check_forbidden_departure_hours(
            forbidden_departure_hours=forbidden_departure_hours
        )
        self.seats = seats
        self.tsps = tsps
        self._penalty_arrival_time = get_function(function_name=penalty_arrival_time)
        self.penalty_arrival_time_kwargs = list(penalty_arrival_time_kwargs.values())
        self._penalty_departure_time = get_function(function_name=penalty_departure_time)
        self.penalty_departure_time_kwargs = list(penalty_departure_time_kwargs.values())
        self._penalty_cost = get_function(function_name=penalty_cost)
        self.penalty_cost_kwargs = list(penalty_cost_kwargs.values())
        self._penalty_travel_time = get_function(function_name=penalty_travel_time)
        self.penalty_travel_time_kwargs = list(penalty_travel_time_kwargs.values())
        self._error, self.error_kwargs = get_scipy_distribution(
            distribution_name=error, is_discrete=False, **error_kwargs
        )
        self._error_rvs = None
        self._error_rvs_idx = 0
        self.default_seat_utility = default_seat_utility
        self.default_tsp_utility = default_tsp_utility
        self.default_rvs_size = default_rvs_size

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
        elif any(hour < 0 or hour > 24 for hour in forbidden_departure_hours):
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        elif forbidden_departure_hours[0] >= forbidden_departure_hours[1]:
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        return forbidden_departure_hours
    
    @property
    def arrival_time(self) -> float:
        """
        Returns a random variable sample from the arrival time distribution.
        
        It implements a cache mechanism to avoid the generation of random variables at each call.

        In case the arrival time is greater than 24, it is reduced to the range [0, 24).

        Returns:
            float: A random variable sample from the distribution.
        """
        if self._arrival_time_rvs is None or self._arrival_time_rvs_idx >= self.default_rvs_size - 1:
            self._arrival_time_rvs = self._arrival_time.rvs(**self.arrival_time_kwargs, size=self.default_rvs_size)
            self._arrival_time_rvs %= 24 # Reduce the arrival time to the range [0, 24) hours
            self._arrival_time_rvs_idx = 0
        else:
            self._arrival_time_rvs_idx += 1
        return self._arrival_time_rvs[self._arrival_time_rvs_idx]
    
    @property
    def purchase_day(self) -> int:
        """
        Returns a random variable sample from the purchase day distribution.
        
        It implements a cache mechanism to avoid the generation of random variables at each call.

        Returns:
            float: A random variable sample from the distribution.
        """
        if self._purchase_day_rvs is None or self._purchase_day_rvs_idx >= self.default_rvs_size - 1:
            self._purchase_day_rvs = self._purchase_day.rvs(**self.purchase_day_kwargs, size=self.default_rvs_size)
            self._purchase_day_rvs = self._purchase_day_rvs.clip(min=0) # Clip the purchase day to the range [0, inf) days
            self._purchase_day_rvs_idx = 0
        else:
            self._purchase_day_rvs_idx += 1
        return self._purchase_day_rvs[self._purchase_day_rvs_idx]
    
    def get_seat_utility(self, seat: int) -> float:
        """
        Returns the utility of the given seat.

        Args:
            seat (int): The seat.

        Returns:
            float: The utility of the given seat.
        """
        return self.seats.get(seat, self.default_seat_utility)
    
    def get_tsp_utility(self, tsp: int) -> float:
        """
        Returns the utility of the given train service provider.

        Args:
            tsp (int): The train service provider.

        Returns:
            float: The utility of the given train service provider.
        """
        return self.tsps.get(tsp, self.default_tsp_utility)
    
    def penalty_arrival_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the arrival time.

        Args:
            x (float): The arrival time.

        Returns:
            float: The penalty function value for the arrival time.
        """
        return self._penalty_arrival_time(x=x, coeff=self.penalty_arrival_time_kwargs)
    
    def penalty_departure_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the departure time.

        Args:
            x (float): The departure time.

        Returns:
            float: The penalty function value for the departure time.
        """
        return self._penalty_departure_time(x=x, coeff=self.penalty_departure_time_kwargs)
    
    def penalty_cost(self, x: float) -> float:
        """
        Returns the value of the penalty function for the cost.

        Args:
            x (float): The cost.

        Returns:
            float: The penalty function value for the cost.
        """
        return self._penalty_cost(x=x, coeff=self.penalty_cost_kwargs)
    
    def penalty_travel_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the travel time.

        Args:
            x (float): The travel time.

        Returns:
            float: The penalty function value for the travel time.
        """
        return self._penalty_travel_time(x=x, coeff=self.penalty_travel_time_kwargs)
    
    @property
    def error(self) -> float:
        """
        Returns a random variable sample from the error distribution.
        
        It implements a cache mechanism to avoid the generation of random variables at each call.

        Returns:
            float: A random variable sample from the distribution.
        """
        if self._error_rvs is None or self._error_rvs_idx >= self.default_rvs_size - 1:
            self._error_rvs = self._error.rvs(**self.error_kwargs, size=self.default_rvs_size)
            self._error_rvs_idx = 0
        else:
            self._error_rvs_idx += 1
        return self._error_rvs[self._error_rvs_idx]

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the user pattern.

        Returns:
            str: A human readable string representation of the user pattern.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        Returns the debuggable representation of the user pattern.

        Returns:
            str: The debuggable representation of the user pattern.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'arrival_time={self._arrival_time}, '
            f'arrival_time_kwargs={self.arrival_time_kwargs}, '
            f'purchase_day={self._purchase_day}, '
            f'purchase_day_kwargs={self.purchase_day_kwargs}, '
            f'forbidden_departure_hours={self.forbidden_departure_hours}, '
            f'seats={self.seats}, '
            f'penalty_arrival_time={self._penalty_arrival_time}, '
            f'penalty_arrival_time_kwargs={self.penalty_arrival_time_kwargs}, '
            f'penalty_departure_time={self._penalty_departure_time}, '
            f'penalty_departure_time_kwargs={self.penalty_departure_time_kwargs}, '
            f'penalty_cost={self._penalty_cost}, '
            f'penalty_cost_kwargs={self.penalty_cost_kwargs}, '
            f'penalty_travel_time={self._penalty_travel_time}, '
            f'penalty_travel_time_kwargs={self.penalty_travel_time_kwargs}, '
            f'error={self._error}, '
            f'error_kwargs={self.error_kwargs})'
        )


class DemandPattern:
    """
    A demand pattern is determined by the potential demand and the distribution of user patterns in a set of markets.

    Attributes:
        id (int): The demand pattern id.
        name(str): The demand pattern name.
        markets (List[Market]): The list of markets.
        potential_demands(Mapping[Market, Callable]): The potential demand distribution for each market.
        potential_demands_kwargs (Mapping[Market, Mapping[str, Union[int, float]]]): The keyword arguments
            for the potential demand distribution for each market.
        user_patterns_distribution (Mapping[Market, Mapping[UserPattern, float]]): The distribution of user patterns
            for each market.
        default_rvs_size (int): The default size of the random variables sample.

    Raises:
        InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
        InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
        InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        ValueError: Raised when the user pattern distribution does not sum up to 1.
    """
    
    def __init__(
        self,
        id: int,
        name: str,
        markets: List[Market],
        potential_demands: List[str],
        potential_demands_kwargs: List[Mapping[str, Union[int, float]]],
        user_patterns_distribution: List[Mapping[UserPattern, float]],
        default_rvs_size: int = DEFAULT_RVS_SIZE
    ) -> None:
        """
        Initializes a demand pattern.

        Args:
            id (int): The demand pattern id.
            name(str): The demand pattern name.
            markets (List[Market]): The list of markets.
            potential_demands(List[Callable]): The list of potential demand distributions.
            potential_demands_kwargs (List[Mapping[str, Union[int, float]]]): The list of potential
                demand distribution named parameters.
            user_patterns_distribution (List[Mapping[UserPattern, float]]): The list of user pattern distributions.
            default_rvs_size (int, optional): The default size of the random variables sample.

        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        """
        same_length = len(markets) == len(potential_demands) == len(potential_demands_kwargs) == len(user_patterns_distribution)
        assert same_length, 'The number of markets must be equal to the number of potential demand distributions and the number of user pattern distributions.'
        self.id = id
        self.name = name
        self.markets = markets
        self._potential_demands = {}
        self.potential_demands_kwargs = {}
        self._user_patterns_distribution_rvs = {}
        self._user_patterns_distribution_rvs_idx = {}
        self.user_patterns_distribution = {}
        for i in range(len(markets)):
            market = markets[i]
            potential_demand = potential_demands[i]
            potential_demand_kwargs = potential_demands_kwargs[i]
            user_pattern_distribution = user_patterns_distribution[i]
            self._potential_demands[market], self.potential_demands_kwargs[market] = get_scipy_distribution(
                distribution_name=potential_demand,
                is_discrete=True,
                **potential_demand_kwargs
            )
            self._user_patterns_distribution_rvs[market] = None
            self._user_patterns_distribution_rvs_idx[market] = 0
            self.user_patterns_distribution[market] = user_pattern_distribution
        self.default_rvs_size = default_rvs_size

    def potential_demand(self, market: Market) -> int:
        """
        Returns a random variable sample from the potential demand distribution of the given market.

        Args:
            market (Market): The market.

        Returns:
            int: A random variable sample from the distribution.
        """
        potential_demand = self._potential_demands[market].rvs(**self.potential_demands_kwargs[market])
        assert potential_demand >= 0, 'The potential demand must be non-negative.'
        return potential_demand
    
    def get_user_pattern(self, market: Market) -> UserPattern:
        """
        Samples a user pattern according to the user pattern distribution of the given market.

        Args:
            market (Market): The market.

        Returns:
            UserPattern: The user pattern.

        Raises:
            ValueError: Raised when the user pattern distribution does not sum up to 1.
        """
        # NOTE: Vectorize the user pattern distribution.
        if self._user_patterns_distribution_rvs[market] is None or self._user_patterns_distribution_rvs_idx[market] >= self.default_rvs_size - 1:
            user_pattern_distribution_market = self.user_patterns_distribution[market]
            self._user_patterns_distribution_rvs[market] = np.random.choice(
                a=list(user_pattern_distribution_market.keys()),
                size=self.default_rvs_size,
                p=list(user_pattern_distribution_market.values())
            )
            self._user_patterns_distribution_rvs_idx[market] = 0
        else:
            self._user_patterns_distribution_rvs_idx[market] += 1
        return self._user_patterns_distribution_rvs[market][self._user_patterns_distribution_rvs_idx[market]]

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the demand pattern.

        Returns:
            str: A human readable string representation of the demand pattern.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the demand pattern.

        Returns:
            str: The debuggable string representation of the demand pattern.
        """
        return (
            f'DemandPattern(id={self.id}, '
            f'name={self.name}, '
            f'markets={self.markets}, '
            f'potential_demands={self._potential_demands}, '
            f'potential_demands_kwargs={self.potential_demands_kwargs}, '
            f'user_patterns_distribution={self.user_patterns_distribution})'
        )


class Day:
    """
    A day is described as a date with and associated demand pattern.
    
    Attributes:
        id (int): The day id.
        date (datetime.date): The actual date.
        demand_pattern (DemandPattern): The associated demand pattern.
    """
    
    def __init__(self, id: int, date: datetime.date, demand_pattern: DemandPattern) -> None:
        """
        Initializes a day.

        Args:
            id(int): The day id.
            date (datetime.date): The actual date.
            demand_pattern (DemandPattern): The associated demand pattern.
        """
        self.id = id
        self.date = date
        self.demand_pattern = demand_pattern

    def generate_passengers(self, id_offset: int = 1) -> List['Passenger']:
        """
        Generates passengers according to the demand pattern.

        Args:
            id_offset (int): The id offset for the generated passengers.

        Returns:
            List[Passenger]: The generated passengers.
        """
        passengers = []
        for market in self.demand_pattern.markets:
            potential_demand = self.demand_pattern.potential_demand(market)
            for i in range(potential_demand):
                user_pattern = self.demand_pattern.get_user_pattern(market)
                passengers.append(
                    Passenger(
                        id=i + id_offset,
                        user_pattern=user_pattern,
                        market=market,
                        arrival_day=self,
                        arrival_time=user_pattern.arrival_time,
                        purchase_day=user_pattern.purchase_day,
                    )
                )
            id_offset += potential_demand
        return passengers

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the day.

        Returns:
            str: A human readable string representation of the day.
        """
        return str(self.date)

    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the day.

        Returns:
            str: The debuggable string representation of the day.
        """
        return f'Day(id={self.id}, date={self.date}, demand_pattern={self.demand_pattern})'


class Passenger:
    """
    A passenger is defined by his/her user pattern, the origin-destination pair, the desired day and time of arrival and the day of purchase.
    
    Attributes:
        id (int): The passenger id.
        user_pattern (UserPattern): The user pattern that this passengers belongs.
        market (Market): The market composed by the origin-destination station pair.
        arrival_day (Day): The desired day of arrival.
        arrival_time (float): The desired time of arrival.
        purchase_day (int): The day of purchase of the train ticket.
        service (Service): The service that this passenger is assigned to.
        service_departure_time (float): The departure time of the service.
        service_arrival_time (float): The arrival time of the service.
        seat (Seat): The seat that this passenger is assigned to.
        seat_price (float): The price of the seat.
        utility (float): The utility of the seat.
        best_service (Service): The best service that fits the passenger needs.
        best_seat (Seat): The best seat from the best service for the passenger.
        best_utility (float): The utility of the best seat.
    """
    
    def __init__(
        self,
        id: int,
        user_pattern: UserPattern,
        market: Market,
        arrival_day: Day,
        arrival_time: float,
        purchase_day: int
    ) -> None:
        """
        Initializes a passenger.

        Args:
            id (int): The passenger id.
            user_pattern (UserPattern): The user pattern that this passengers belongs.
            market (Market): The market composed by the origin-destination station pair.
            arrival_day (Day): The desired day of arrival.
            arrival_time (float): The desired time of arrival.
            purchase_day (int): The day of purchase of the train ticket.
        """
        self.id = id
        self.user_pattern = user_pattern
        self.market = market
        self.arrival_day = arrival_day
        self.arrival_time = arrival_time
        self.purchase_day = purchase_day
        self.service = None
        self.service_departure_time = None
        self.service_arrival_time = None
        self.seat = None
        self.ticket_price = None
        self.utility = None
        self.best_service = None
        self.best_seat = None
        self.best_utility = None

    def _is_valid_departure_time(self, service_departure_time: float) -> bool:
        """
        Checks if the departure time is valid.

        Args:
            service_departure_time (float): The departure time of the service.

        Returns:
            bool: True if the departure time is valid, False otherwise.
        """
        forbidden_departure_hours = self.user_pattern.forbidden_departure_hours
        is_valid_departure_time_early = service_departure_time >= forbidden_departure_hours[0]
        is_valid_departure_time_later = service_departure_time <= forbidden_departure_hours[1]
        return not (is_valid_departure_time_early and is_valid_departure_time_later)

    def _get_utility_arrival_time(self, service_arrival_time: float) -> float:
        """
        Returns the utility of the passenger given the arrival time.

        Args:
            service_arrival_time (float): The arrival time of the service.

        Returns:
            float: The utility of the passenger given the arrival time.
        """
        # NOTE: This speed up the arrival time utility by avoiding using max function.
        # earlier_displacement = max(self.arrival_time - service_arrival_time, 0)
        # later_displacement = max(service_arrival_time - self.arrival_time, 0)
        earlier_displacement = self.arrival_time - service_arrival_time if self.arrival_time > service_arrival_time else 0
        later_displacement = service_arrival_time - self.arrival_time if self.arrival_time < service_arrival_time else 0
        return self.user_pattern.penalty_arrival_time(earlier_displacement + later_displacement)

    def _get_utility_departure_time(self, service_departure_time: float) -> float:
        """
        Returns the utility of the passenger given the departure time.

        Args:
            service_departure_time (float): The departure time of the service.

        Returns:
            float: The utility of the passenger given the departure time.
        """
        dt_begin = self.user_pattern.forbidden_departure_hours[0]
        dt_end = self.user_pattern.forbidden_departure_hours[1]
        # NOTE: This speed up the departure time utility by avoiding using min and max functions.
        # departure_time = min(max(dt_end - service_departure_time, 0), dt_end - dt_begin)
        _departure_time = dt_end - service_departure_time if dt_end > service_departure_time else 0
        departure_time = _departure_time if _departure_time < dt_end - dt_begin else dt_end - dt_begin
        return self.user_pattern.penalty_departure_time(departure_time)

    def _get_utility_price(self, price: float) -> float:
        """
        Returns the utility of the passenger given the price.

        Args:
            price (float): The price of the service.

        Returns:
            float: The utility of the passenger given the price.
        """
        return self.user_pattern.penalty_cost(price)

    def _get_utility_travel_time(self, service_arrival_time: float, service_departure_time: float) -> float:
        """
        Returns the utility of the passenger given the travel time.

        Args:
            service_arrival_time (float): The arrival time of the service.
            service_departure_time (float): The departure time of the service.

        Returns:
            float: The utility of the passenger given the travel time.
        """
        return self.user_pattern.penalty_travel_time(service_arrival_time - service_departure_time)

    def get_utility(
        self,
        seat: int,
        tsp: int,
        service_departure_time: float,
        service_arrival_time: float,
        price: float,
        departure_time_hard_restriction: bool = False
    ) -> float:
        """
        Returns the utility of the passenger given the seat, the arrival time, the departure time and the price.

        Args:
            seat (int): The seat of the service.
            tsp (int): The train service provider of the service.
            service_departure_time (float): The departure time of the service.
            service_arrival_time (float): The arrival time of the service.
            price (float): The price of the seat.
            departure_time_hard_restriction (bool, optional): If True, the passenger will not be
                assigned to a service with a departure time that is not valid. Defaults to False.
        
        Returns:
            float: The utility of the passenger given the seat, the arrival time, the departure time and the price.
        """
        if departure_time_hard_restriction and not self._is_valid_departure_time(service_departure_time):
            return -np.inf # Minimum utility
        seat_utility = self.user_pattern.get_seat_utility(seat)
        tsp_utility = self.user_pattern.get_tsp_utility(tsp)
        arrival_time_utility = self._get_utility_arrival_time(service_arrival_time)
        departure_time_utility = self._get_utility_departure_time(service_departure_time)
        price_utility = self._get_utility_price(price)
        travel_time_utility = self._get_utility_travel_time(service_arrival_time, service_departure_time)
        error_utility = self.user_pattern.error
        return seat_utility + tsp_utility - arrival_time_utility - departure_time_utility - price_utility - travel_time_utility + error_utility

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the passenger.

        Returns:
            str: A human readable string representation of the passenger.
        """
        return (
            f'Passenger {self.id} from {self.market.departure_station} to {self.market.arrival_station} '
            f'desired to arrive at {self.arrival_day} {self.arrival_time} '
            f'purchasing with {self.purchase_day} antelation days '
            f'with utility {self.utility}'
        )

    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the passenger.

        Returns:
            str: The debuggable string representation of the passenger.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'user_pattern={self.user_pattern}, '
            f'market={self.market}, '
            f'arrival_day={self.arrival_day}, '
            f'arrival_time={self.arrival_time}, '
            f'purchase_day={self.purchase_day}, '
            f'service={self.service}, '
            f'service_departure_time={self.service_departure_time}, '
            f'service_arrival_time={self.service_arrival_time}, '
            f'seat={self.seat}, '
            f'ticket_price={self.ticket_price}, '
            f'utility={self.utility}'
        )


class Demand:
    """
    A demand is a collection of days, where each day is associated with a demand pattern and a market.
    
    Attributes:
        days (List[Day]): The list of days.
    """
    
    def __init__(self, days: List[Day]) -> None:
        """
        Initializes a demand.

        Args:
            days (List[Day]): The list of days.
        """
        self.days = days

    @classmethod
    def _get_markets(cls, data: Mapping[str, Any]) -> Mapping[int, Market]:
        """
        Returns the markets.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.

        Returns:
            Mapping[int, Market]: The markets.
        """
        markets = {}
        for market in data['market']:
            markets[market['id']] = Market(**market)
        return markets

    @classmethod
    def _utility_list_to_dict(cls, list_: List[Mapping[str, Any]]) -> Mapping[int, float]:
        """
        Convert a list of utilities into a dictionary.

        Args:
            list_ (List[Mapping[str, Any]]): The list of utilities.

        Returns:
            Mapping[int, float]: The dictionary of utilities.
        """
        return {item['id']: item['utility'] for item in list_}

    @classmethod
    def _get_user_patterns(cls, data: Mapping[str, Any]) -> Mapping[int, UserPattern]:
        """
        Returns the user patterns.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.

        Returns:
            Mapping[int, UserPattern]: The user patterns.
        """
        user_patterns = {}
        for user_pattern in data['userPattern']:
            # Convert the forbidden departure hours into a tuple (begin, end)
            forbidden_departure_hours = tuple(user_pattern['forbidden_departure_hours'].values())
            user_pattern.pop('forbidden_departure_hours', None)

            # Convert the list of seats and tsps into a dictionary {id: utility}
            seats = Demand._utility_list_to_dict(user_pattern['seats'])
            tsps = Demand._utility_list_to_dict(user_pattern['train_service_providers'])
            user_pattern.pop('seats', None)
            user_pattern.pop('train_service_providers', None)

            user_patterns[user_pattern['id']] = UserPattern(
                forbidden_departure_hours=forbidden_departure_hours,
                seats=seats,
                tsps=tsps,
                **user_pattern
            )
        return user_patterns

    @classmethod
    def _get_demand_patterns(
        cls,
        data: Mapping[str, Any],
        markets: Mapping[int, Market],
        user_patterns: Mapping[int, UserPattern]
    ) -> Mapping[int, DemandPattern]:
        """
        Returns the demand patterns.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.
            markets (Mapping[int, Market]): The markets.
            user_patterns (Mapping[int, UserPattern]): The user patterns.

        Returns:
            Mapping[int, DemandPattern]: The demand patterns.
        """
        demand_patterns = {}
        for demand_pattern in data['demandPattern']:
            # Get the markets, the potential demands and the user patterns distribution
            markets_objects = []
            potential_demands = []
            potential_demands_kwargs = []
            user_patterns_distribution = []
            for market in demand_pattern['markets']:
                markets_objects.append(markets[market['market']])
                potential_demands.append(market['potential_demand'])
                potential_demands_kwargs.append(market['potential_demand_kwargs'])

                # Convert the list of user patterns distributions into a dictionary {user_pattern: percentage}
                user_pattern_distribution = {}
                for demand_upd in market['user_pattern_distribution']:
                    user_pattern_distribution[user_patterns[demand_upd['id']]] = demand_upd['percentage']
                user_patterns_distribution.append(user_pattern_distribution)
            
            demand_patterns[demand_pattern['id']] = DemandPattern(
                id=demand_pattern['id'],
                name=demand_pattern['name'],
                markets=markets_objects,
                potential_demands=potential_demands,
                potential_demands_kwargs=potential_demands_kwargs,
                user_patterns_distribution=user_patterns_distribution
            )
        return demand_patterns

    @classmethod
    def _get_days(
        cls,
        data: Mapping[str, Any],
        demand_patterns: Mapping[int, DemandPattern]
    ) -> Mapping[int, Day]:
        """
        Returns the days.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.
            demand_patterns (Mapping[int, DemandPattern]): The demand patterns.

        Returns:
            Mapping[int, Day]: The days.
        """
        days = {}
        for day in data['day']:
            demand_pattern = demand_patterns[day['demandPattern']]
            day.pop('demandPattern', None)
            days[day['id']] = Day(demand_pattern=demand_pattern, **day)
        return days

    @classmethod
    def from_yaml(cls, path: Path) -> 'Demand':
        """
        Creates a demand from a YAML file.

        NOTE: The YAML file must have the following keys in any order:
            - market
            - userPattern
            - demandPattern
            - day

        See data/demand_data_example.yaml for an example.

        Args:
            path (Path): The path to the YAML file.

        Returns:
            Demand: The demand created from the YAML file.
        """
        with open(path, 'r') as f:
            demand_yaml = f.read()

        data = yaml.load(demand_yaml, Loader=yaml.CSafeLoader)
        markets = Demand._get_markets(data)
        user_patterns = Demand._get_user_patterns(data)
        demand_patterns = Demand._get_demand_patterns(data, markets, user_patterns)
        days = Demand._get_days(data, demand_patterns)

        return cls(list(days.values()))
    
    def generate_passengers(self) -> List[Passenger]:
        """
        Generates the passengers for all days sorted by purchase day (descending).

        Returns:
            List[Passenger]: The generated passengers sorted by purchase day (descending).
        """
        passengers = []
        id_offset = 1
        for day in self.days:
            passengers_day = day.generate_passengers(id_offset)
            passengers += passengers_day
            id_offset += len(passengers_day)
        passengers.sort(key=lambda x: x.purchase_day, reverse=True)
        return passengers

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the demand.

        Returns:
            str: A human readable string representation of the demand.
        """
        return str(self.days)

    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the demand.

        Returns:
            str: The debuggable string representation of the demand.
        """
        return (
            f'{self.__class__.__name__}('
            f'days={self.days})'
        )
