"""Entities for the demand module."""

import datetime
import numpy as np
import yaml

from .exceptions import InvalidForbiddenDepartureHoursException
from .utils import get_function, get_scipy_distribution

from copy import deepcopy
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
        return f'{self.__class__.__name__}(id={self.id}, departure_station={self.departure_station}, arrival_station={self.arrival_station})'


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
            penalty_arrival_time: str,
            penalty_arrival_time_kwargs: Mapping[str, Union[int, float]],
            penalty_departure_time: str,
            penalty_departure_time_kwargs: Mapping[str, Union[int, float]],
            penalty_cost: str,
            penalty_cost_kwargs: Mapping[str, Union[int, float]],
            penalty_travel_time: str,
            penalty_travel_time_kwargs: Mapping[str, Union[int, float]],
            error: str,
            error_kwargs: Mapping[str, Union[int, float]]
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
        
        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
            InvalidForbiddenDepartureHoursException: Raised when the given forbidden departure hours are not valid.
            InvalidFunctionException: Raised when the given function is not contained in the ROBIN module.
        """
        self.id = id
        self.name = name
        self._arrival_time = get_scipy_distribution(distribution_name=arrival_time, is_discrete=False)
        self.arrival_time_kwargs = arrival_time_kwargs
        self._purchase_day = get_scipy_distribution(distribution_name=purchase_day, is_discrete=True)
        self.purchase_day_kwargs = purchase_day_kwargs
        self.forbidden_departure_hours = self._check_forbidden_departure_hours(forbidden_departure_hours=forbidden_departure_hours)
        self.seats = seats
        self._penalty_arrival_time = get_function(function_name=penalty_arrival_time)
        self.penalty_arrival_time_kwargs = penalty_arrival_time_kwargs
        self._penalty_departure_time = get_function(function_name=penalty_departure_time)
        self.penalty_departure_time_kwargs = penalty_departure_time_kwargs
        self._penalty_cost = get_function(function_name=penalty_cost)
        self.penalty_cost_kwargs = penalty_cost_kwargs
        self._penalty_travel_time = get_function(function_name=penalty_travel_time)
        self.penalty_travel_time_kwargs = penalty_travel_time_kwargs
        self._error = get_scipy_distribution(distribution_name=error, is_discrete=False)
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
        elif any(hour < 0 or hour > 24 for hour in forbidden_departure_hours):
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        elif forbidden_departure_hours[0] >= forbidden_departure_hours[1]:
            raise InvalidForbiddenDepartureHoursException(forbidden_departure_hours)
        return forbidden_departure_hours
    
    @property
    def arrival_time(self) -> float:
        """
        Returns a random variable sample from the arrival time distribution.

        In case the arrival time is greater than 24, it is reduced to the range [0, 24).

        Returns:
            float: A random variable sample from the distribution.
        """
        arrival_time = self._arrival_time.rvs(**self.arrival_time_kwargs)
        return arrival_time % 24
    
    @property
    def purchase_day(self) -> int:
        """
        Returns a random variable sample from the purchase day distribution.

        Returns:
            float: A random variable sample from the distribution.
        """
        return self._purchase_day.rvs(**self.purchase_day_kwargs)
    
    def get_seat_utility(self, seat: int) -> float:
        """
        Returns the utility of the given seat.

        Args:
            seat (int): The seat.

        Returns:
            float: The utility of the given seat.
        """
        return self.seats.get(seat, 0)
    
    def penalty_arrival_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the arrival time.

        Args:
            x (float): The arrival time.

        Returns:
            float: The penalty function value for the arrival time.
        """
        return self._penalty_arrival_time(x=x, **self.penalty_arrival_time_kwargs)
    
    def penalty_departure_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the departure time.

        Args:
            x (float): The departure time.

        Returns:
            float: The penalty function value for the departure time.
        """
        return self._penalty_departure_time(x=x, **self.penalty_departure_time_kwargs)
    
    def penalty_cost(self, x: float) -> float:
        """
        Returns the value of the penalty function for the cost.

        Args:
            x (float): The cost.

        Returns:
            float: The penalty function value for the cost.
        """
        return self._penalty_cost(x=x, **self.penalty_cost_kwargs)
    
    def penalty_travel_time(self, x: float) -> float:
        """
        Returns the value of the penalty function for the travel time.

        Args:
            x (float): The travel time.

        Returns:
            float: The penalty function value for the travel time.
        """
        return self._penalty_travel_time(x=x, **self.penalty_travel_time_kwargs)
    
    @property
    def error(self) -> float:
        """
        Returns a random variable sample from the error distribution.

        Returns:
            float: A random variable sample from the distribution.
        """
        return self._error.rvs(**self.error_kwargs)

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
            f'arrival_time={self.arrival_time}, '
            f'arrival_time_kwargs={self.arrival_time_kwargs}, '
            f'purchase_day={self.purchase_day}, '
            f'purchase_day_kwargs={self.purchase_day_kwargs}, '
            f'forbidden_departure_hours={self.forbidden_departure_hours}, '
            f'seats={self.seats}, '
            f'penalty_arrival_time={self.penalty_arrival_time}, '
            f'penalty_arrival_time_kwargs={self.penalty_arrival_time_kwargs}, '
            f'penalty_departure_time={self.penalty_departure_time}, '
            f'penalty_departure_time_kwargs={self.penalty_departure_time_kwargs}, '
            f'penalty_cost={self.penalty_cost}, '
            f'penalty_cost_kwargs={self.penalty_cost_kwargs}, '
            f'penalty_travel_time={self.penalty_travel_time}, '
            f'penalty_travel_time_kwargs={self.penalty_travel_time_kwargs}, '
            f'error={self.error}, '
            f'error_kwargs={self.error_kwargs})'
        )


class DemandPattern:
    """
    A demand pattern is determined by the potential demand and the distribution of user patterns.

    Attributes:
        id (int): The demand pattern id.
        name(str): The demand pattern name.
        potential_demand(Callable): The potential demand distribution function.
        potential_demand_kwargs (Mapping[str, Union[int, float]]): The potential demand distribution named parameters.
        user_pattern_distribution (Mapping[UserPattern, float]): The user pattern distribution.

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
            potential_demand: str,
            potential_demand_kwargs: Mapping[str, Union[int, float]],
            user_pattern_distribution: Mapping[UserPattern, float]
        ) -> None:
        """
        Initializes a demand pattern.

        Args:
            id (int): The demand pattern id.
            name(str): The demand pattern name.
            potential_demand (str): The potential demand distribution name.
            potential_demand_kwargs (Mapping[str, Union[int, float]]): The potential demand distribution named parameters.
            user_pattern_distribution (Mapping[UserPattern, float]): The user pattern distribution.

        Raises:
            InvalidDistributionException: Raised when the given distribution is not contained in SciPy.
            InvalidContinuousDistributionException: Raised when the given distribution is not a continuous distribution.
            InvalidDiscreteDistributionException: Raised when the given distribution is not a discrete distribution.
        """
        self.id = id
        self.name = name
        self._potential_demand = get_scipy_distribution(distribution_name=potential_demand, is_discrete=True)
        self.potential_demand_kwargs = potential_demand_kwargs
        self.user_pattern_distribution = user_pattern_distribution

    @property
    def potential_demand(self) -> int:
        """
        Returns a random variable sample from the potential demand distribution.

        Returns:
            int: A random variable sample from the distribution.
        """
        return self._potential_demand.rvs(**self.potential_demand_kwargs)
    
    def get_user_pattern(self) -> UserPattern:
        """
        Samples a user pattern according to the user pattern distribution.

        Returns:
            UserPattern: The user pattern.

        Raises:
            ValueError: Raised when the user pattern distribution does not sum up to 1.
        """
        return np.random.choice(list(self.user_pattern_distribution.keys()), p=list(self.user_pattern_distribution.values()))

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
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'potential_demand={self.potential_demand}, '
            f'potential_demand_kwargs={self.potential_demand_kwargs}, '
            f'user_pattern_distribution={self.user_pattern_distribution})'
        )


class Day:
    """
    A day is described as a date with and associated demand pattern in a market.
    
    Attributes:
        id (int): The day id.
        date (datetime.date): The actual date.
        demand_pattern (DemandPattern): The associated demand pattern.
        market (Market): The associated market.
    """
    
    def __init__(self, id: int, date: datetime.date, demand_pattern: DemandPattern, market: Market) -> None:
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
        self.market = market

    def generate_passengers(self, id_offset: int = 1) -> List['Passenger']:
        """
        Generates passengers according to the demand pattern.

        Args:
            id_offset (int): The id offset for the generated passengers.

        Returns:
            List[Passenger]: The generated passengers.
        """
        passengers = []
        for i in range(self.demand_pattern.potential_demand):
            user_pattern = self.demand_pattern.get_user_pattern()
            passengers.append(
                Passenger(
                    id=i + id_offset,
                    user_pattern=user_pattern,
                    market=self.market,
                    arrival_day=self,
                    arrival_time=user_pattern.arrival_time,
                    purchase_day=user_pattern.purchase_day,
                )
            )
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
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'date={self.date}, '
            f'demand_pattern={self.demand_pattern}), '
            f'market={self.market})'
        )


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
    """
    
    def __init__(self, id: int, user_pattern: UserPattern, market: Market, arrival_day: Day, arrival_time: float, purchase_day: int) -> None:
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

    def _is_valid_departure_time(self, service_departure_time: float) -> bool:
        """
        Checks if the departure time is valid.

        Args:
            service_departure_time (float): The departure time of the service.

        Returns:
            bool: True if the departure time is valid, False otherwise.
        """
        forbidden_departure_hours = self.user_pattern.forbidden_departure_hours
        return not (service_departure_time >= forbidden_departure_hours[0] and service_departure_time <= forbidden_departure_hours[1])

    def _get_utility_arrival_time(self, service_arrival_time: float) -> float:
        """
        Returns the utility of the passenger given the arrival time.

        Args:
            service_arrival_time (float): The arrival time of the service.

        Returns:
            float: The utility of the passenger given the arrival time.
        """
        earlier_displacement = max(self.arrival_time - service_arrival_time, 0)
        later_displacement = max(service_arrival_time - self.arrival_time, 0)
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
        departure_time = min(max(dt_end - service_departure_time, 0), dt_end - dt_begin)
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
            service_departure_time: float,
            service_arrival_time: float,
            price: float,
            departure_time_hard_restriction: bool = False
        ) -> float:
        """
        Returns the utility of the passenger given the seat, the arrival time, the departure time and the price.

        Args:
            seat (int): The seat of the service.
            service_departure_time (float): The departure time of the service.
            service_arrival_time (float): The arrival time of the service.
            price (float): The price of the seat.
        
        Returns:
            float: The utility of the passenger given the seat, the arrival time, the departure time and the price.
        """
        if departure_time_hard_restriction and not self._is_valid_departure_time(service_departure_time):
            return -np.inf # Minimum utility
        seat_utility = self.user_pattern.get_seat_utility(seat)
        arrival_time_utility = self._get_utility_arrival_time(service_arrival_time)
        departure_time_utility = self._get_utility_departure_time(service_departure_time)
        price_utility = self._get_utility_price(price)
        travel_time_utility = self._get_utility_travel_time(service_arrival_time, service_departure_time)
        error_utility = self.user_pattern.error
        return seat_utility + arrival_time_utility - departure_time_utility - price_utility - travel_time_utility + error_utility

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
        for key, value in data.items():
            if key == 'market':
                for market in value:
                    markets[market['id']] = Market(**market)
        return markets

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
        for key, value in data.items():
            if key == 'userPattern':
                for user_pattern in value:
                    # Convert the forbidden departure hours into a tuple (begin, end)
                    forbidden_departure_hours = tuple(user_pattern['forbidden_departure_hours'].values())
                    user_pattern.pop('forbidden_departure_hours', None)

                    # Convert the list of seats into a dictionary {id: utility}
                    ids = []
                    utilities = []
                    for seat in user_pattern['seats']:
                        ids.append(seat['id'])
                        utilities.append(seat['utility'])
                    seats = dict(zip(ids, utilities))
                    user_pattern.pop('seats', None)

                    user_patterns[user_pattern['id']] = UserPattern(
                        forbidden_departure_hours=forbidden_departure_hours,
                        seats=seats,
                        **user_pattern
                    )
        return user_patterns

    @classmethod
    def _get_demand_patterns(
            cls,
            data: Mapping[str, Any],
            user_patterns: Mapping[int, UserPattern]
        ) -> Mapping[int, DemandPattern]:
        """
        Returns the demand patterns.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.
            user_patterns (Mapping[int, UserPattern]): The user patterns.

        Returns:
            Mapping[int, DemandPattern]: The demand patterns.
        """
        demand_patterns = {}
        for key, value in data.items():
            if key == 'demandPattern':
                for demand_pattern in value:
                    # Convert the list of user pattern distributions into a dictionary {user_pattern: percentage}
                    user_pattern_distribution = {}
                    for demand_upd in demand_pattern['user_pattern_distribution']:
                        user_pattern_distribution[user_patterns[demand_upd['id']]] = demand_upd['percentage']
                    demand_pattern.pop('user_pattern_distribution', None)

                    demand_patterns[demand_pattern['id']] = DemandPattern(
                        user_pattern_distribution=user_pattern_distribution,
                        **demand_pattern
                    )
        return demand_patterns

    @classmethod
    def _get_days(
            cls,
            data: Mapping[str, Any],
            markets: Mapping[int, Market],
            demand_patterns: Mapping[int, DemandPattern]
        ) -> Mapping[int, Day]:
        """
        Returns the days.

        Args:
            data (Mapping[str, Any]): The data of the YAML file.
            markets (Mapping[int, Market]): The markets.
            demand_patterns (Mapping[int, DemandPattern]): The demand patterns.

        Returns:
            Mapping[int, Day]: The days.
        """
        days = {}
        for key, value in data.items():
            if key == 'day':
                for day in value:
                    id_ = day['id']
                    date = day['date']
                    demand_pattern = demand_patterns[day['demandPattern']]
                    market = markets[day['market']]
                    days[day['id']] = Day(id=id_, date=date, demand_pattern=demand_pattern, market=market)
        return days

    @classmethod
    def from_yaml(cls, path: str) -> 'Demand':
        """
        Creates a demand from a YAML file.

        NOTE: The YAML file must have the following keys in any order:
            - market
            - userPattern
            - demandPattern
            - day

        See data/demand_data_example.yaml for an example.

        Args:
            path (str): The path to the YAML file.

        Returns:
            Demand: The demand created from the YAML file.
        """
        with open(path, 'r') as f:
            demand_yaml = f.read()

        data = yaml.safe_load(demand_yaml)
        markets = Demand._get_markets(data)
        user_patterns = Demand._get_user_patterns(data)
        demand_patterns = Demand._get_demand_patterns(data, user_patterns)
        days = Demand._get_days(data, markets, demand_patterns)

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
