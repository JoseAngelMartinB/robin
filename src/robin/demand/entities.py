"""Entities for the demand module."""

from .exceptions import InvalidArrivalTimeException, InvalidForbiddenDepartureHoursException
from .utils import get_function, get_scipy_distribution

from typing import List, Mapping, Union, Tuple

import datetime
import numpy as np
import yaml


class Station:
    """
    Dummy class for a station.
    
    NOTE: This class is not yet implemented. It is just a placeholder.
    """
    pass


class TimeSlot:
    """
    Dummy class for a time slot.
    
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

    def __init__(self, id: int, departure_station: Station, arrival_station: Station) -> None:
        """
        Initialize a market.

        Args:
            id (int): The market id.
            departure_station (Station): The departure station.
            arrival_station (Station): The arrival station.
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
        penalty_traveling_time (Callable): The penalty function for the travel time.
        penalty_traveling_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
        error (Callable): The error distribution function.
        error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.
        arrival_time_maximum_iterations (int): The maximum number of iterations for the arrival time distribution.

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
            penalty_traveling_time: str,
            penalty_traveling_time_kwargs: Mapping[str, Union[int, float]],
            error: str,
            error_kwargs: Mapping[str, Union[int, float]],
            arrival_time_maximum_iterations: int = 100,
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
            penalty_traveling_time (str): The penalty function name for the travel time.
            penalty_traveling_time_kwargs (Mapping[str, Union[int, float]]): The penalty function named parameters.
            error (str): The error distribution name.
            error_kwargs (Mapping[str, Union[int, float]]): The error distribution named parameters.
            arrival_time_maximum_iterations (int): The maximum number of iterations for the arrival time distribution.
        
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
        self._penalty_traveling_time = get_function(function_name=penalty_traveling_time)
        self.penalty_traveling_time_kwargs = penalty_traveling_time_kwargs
        self._error = get_scipy_distribution(distribution_name=error, is_discrete=False)
        self.error_kwargs = error_kwargs
        self.arrival_time_maximum_iterations = arrival_time_maximum_iterations

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
        
        It takes into account the forbidden departure hours.

        Returns:
            float: A random variable sample from the distribution.

        Raises:
            InvalidArrivalTimeException: Raised when it could not be generated a valid arrival time.
        """
        arrival_time = self._arrival_time.rvs(**self.arrival_time_kwargs)
        is_valid_arrival_time = arrival_time >= self.forbidden_departure_hours[0] and arrival_time <= self.forbidden_departure_hours[1]
        iteration_count = 0

        while is_valid_arrival_time and iteration_count < self.arrival_time_maximum_iterations:
            arrival_time = self._arrival_time.rvs(**self.arrival_time_kwargs)
            is_valid_arrival_time = arrival_time >= self.forbidden_departure_hours[0] and arrival_time <= self.forbidden_departure_hours[1]
            iteration_count += 1

        if iteration_count >= self.arrival_time_maximum_iterations:
            raise InvalidArrivalTimeException(self.name, self.forbidden_departure_hours, self.arrival_time_maximum_iterations)
        
        return arrival_time
    
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
    
    @property
    def penalty_arrival_time(self) -> float:
        """
        Returns a random variable sample from the arrival time penalty function.

        Returns:
            float: The arrival time penalty function.
        """
        return self._penalty_arrival_time(**self.penalty_arrival_time_kwargs)
    
    @property
    def penalty_departure_time(self) -> float:
        """
        Returns a random variable sample from the penalty function for the departure time.

        Returns:
            float: The penalty function for the departure time.
        """
        return self._penalty_departure_time(**self.penalty_departure_time_kwargs)
    
    @property
    def penalty_cost(self) -> float:
        """
        Returns a random variable sample from the penalty function for the cost.

        Returns:
            float: The penalty function for the cost.
        """
        return self._penalty_cost(**self.penalty_cost_kwargs)
    
    @property
    def penalty_traveling_time(self) -> float:
        """
        Returns a random variable sample from the penalty function for the traveling time.

        Returns:
            float: The penalty function for the traveling time.
        """
        return self._penalty_traveling_time(**self.penalty_traveling_time_kwargs)
    
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
            f'penalty_traveling_time={self.penalty_traveling_time}, '
            f'penalty_traveling_time_kwargs={self.penalty_traveling_time_kwargs}, '
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

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the passenger.

        Returns:
            str: A human readable string representation of the passenger.
        """  
        return f'{self.id}, {self.market}, {self.user_pattern}, {self.arrival_day}, {self.arrival_time}, {self.purchase_day}'

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
            f'purchase_day={self.purchase_day})'
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
    def from_yaml(cls, path: str) -> 'Demand':
        """
        Creates a demand from a YAML file.

        TODO: This method is too long. Refactor it.

        Args:
            path (str): The path to the YAML file.

        Returns:
            Demand: The demand created from the YAML file.
        """
        with open(path, 'r') as f:
            demand_yaml = f.read()

        data = yaml.safe_load(demand_yaml)

        markets = {}
        user_patterns = {}
        demand_patterns = {}
        days = {}

        for key, value in data.items():
            if key == 'market':
                for market in value:
                    markets[market['id']] = Market(**market)
            elif key == 'userPattern':
                for userPattern in value:
                    forbidden_departure_hours = tuple(userPattern['forbidden_departure_hours'].values())
                    userPattern.pop('forbidden_departure_hours', None)
                    user_patterns[userPattern['id']] = UserPattern(forbidden_departure_hours=forbidden_departure_hours, **userPattern)
            elif key == 'demandPattern':
                for demandPattern in value:
                    user_pattern_distribution = {}
                    for demand_upd in demandPattern['user_pattern_distribution']:
                        user_pattern_distribution[user_patterns[demand_upd['id']]] = demand_upd['percentage']
                    demandPattern.pop('user_pattern_distribution', None)
                    demand_patterns[demandPattern['id']] = DemandPattern(user_pattern_distribution=user_pattern_distribution, **demandPattern)
            elif key == 'day':
                for day in value:
                    id_ = day['id']
                    date = datetime.datetime.strptime(day['date'], '%Y-%m-%d').date()
                    demand_pattern = demand_patterns[day['demandPattern']]
                    market = markets[day['market']]
                    days[day['id']] = Day(id=id_, date=date, demand_pattern=demand_pattern, market=market)

        return cls(list(days.values()))
    
    def generate_passengers(self) -> List[Passenger]:
        """
        Generates the passengers for all days.

        Returns:
            List[Passenger]: The generated passengers.
        """
        passengers = []
        id_offset = 1
        for day in self.days:
            passengers += day.generate_passengers(id_offset)
            id_offset += len(passengers)
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
