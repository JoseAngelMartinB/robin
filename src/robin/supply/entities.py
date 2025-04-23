"""Entities for the supply module."""

import datetime
import yaml

from robin.supply.constants import DEFAULT_LIFT_CONSTRAINTS
from robin.supply.utils import get_time, get_date, format_td, set_stations_ids, convert_tree_to_dict

from copy import deepcopy
from functools import cache, cached_property
from typing import Any, Dict, List, Mapping, Set, Tuple, Union


class Station:
    """
    Railway facility where trains stop to load or unload passengers, freight or both.

    Attributes:
        id (str): Station ID.
        name (str): Station name.
        city (str): Station city.
        short_name (str): Station short name.
        coordinates (Tuple[float, float]): Station coordinates (latitude, longitude).
    """

    def __init__(self, id_: str, name: str, city: str, short_name: str, coordinates: Tuple[float, float] = None) -> None:
        """
        Initialize a Station with name, city, short name and coordinates.

        Args:
            id_ (str): Station ID.
            name (str): Station name.
            city (str): Station city.
            short_name (str): Station short name.
            coordinates (Tuple[float, float]): Station coordinates (latitude, longitude).
        """
        self.id = id_
        self.name = name
        self.city = city
        self.short_name = short_name
        self.coordinates = coordinates

    def __str__(self) -> str:
        """
        Returns a human readable string representation of the station.

        Returns:
            str: A human readable string representation of the station.
        """
        return self.name

    def __repr__(self) -> str:
        """
        Returns a debuggable string representation of the station.

        Returns:
            str: A debuggable string representation of the station.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'city={self.city}, '
            f'short_name={self.short_name}, '
            f'coordinates={self.coordinates})'
        )


class TimeSlot:
    """
    Discrete time interval.

    Attributes:
        id (str): Time slot ID.
        start (datetime.timedelta): Time slot start time.
        end (datetime.timedelta): Time slot end time.
        class_mark (datetime.timedelta): Time slot class mark.
        size (datetime.timedelta): Time slot size.
    """

    def __init__(self, id_: str, start: datetime.timedelta, end: datetime.timedelta) -> None:
        """
        Initialize a TimeSlot with start and end time.

        Args:
            id_ (str): Time slot ID.
            start (datetime.timedelta): Time slot start time.
            end (datetime.timedelta): Time slot end time.
        """
        self.id = id_
        self.start = start
        self.end = end

    @cached_property
    def class_mark(self) -> datetime.timedelta:
        """
        Get class mark of time slot.

        Returns:
            datetime.timedelta: Time slot class mark.
        """
        if self.end < self.start:
            return (self.start + self.end + datetime.timedelta(days=1)) / 2 - datetime.timedelta(days=1)
        return (self.start + self.end) / 2

    @cached_property
    def size(self) -> datetime.timedelta:
        """
        Get size of time slot.

        Returns:
            datetime.timedelta: Time slot size.
        """
        if self.end < self.start:
            return (self.end + datetime.timedelta(days=1)) - self.start
        return self.end - self.start

    def __str__(self) -> str:
        """
        A human readable string representation of the time slot.

        Returns:
            str: A human readable string representation of the time slot.
        """
        return self.id
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the time slot.

        Returns:
            str: A debuggable string representation of the time slot.
        """
        return f'{self.__class__.__name__}(id={self.id}, start={self.start}, end={self.end})'


class Corridor:
    """
    Tree of stations.

    Example:
        {'MAD': {'BAR': {}, 'BIL': {}}}
        *In the real tree, the Station objects are used instead of the Station IDs

    Attributes:
        id (str): Corridor ID.
        name (str): Corridor name.
        tree (Mapping[Station, Mapping]): Tree of stations (with Station objects).
        paths (List[List[Station]]): List of paths (list of stations).
        stations (Dict[str, Station]): Dictionary of stations (with Station IDs as keys).
    """

    def __init__(self, id_: str, name: str, tree: Mapping[Station, Mapping]) -> None:
        """
        Initialize a Corridor with a tree of stations.

        Args:
            id_ (str): Corridor ID.
            name (str): Corridor name.
            tree (Mapping[Station, Mapping]): Tree of stations (with Station objects).
        """
        self.id = id_
        self.name = name
        self.tree = tree
        self.paths = self._get_paths(self.tree)
        self.stations = self._dict_stations(self.tree)

    def _get_paths(
        self,
        tree: Mapping[Station, Mapping],
        path: List[Station] = None,
        paths: List[List[Station]] = None
    ) -> List[List[Station]]:
        """
        Get all paths from a tree of stations.

        Args:
            tree: Tree of stations.
            path (List[Station]): Current path or stations.
            paths (List[List[Station]]): List of paths or stations.

        Returns:
            List[List[Station]]: List of paths or stations.
        """
        if path is None:
            path = []
        if paths is None:
            paths = []
        if not tree:
            paths.append(path.copy())
            return paths

        for node in tree:
            org = node
            path.append(org)
            self._get_paths(tree[node], path, paths)
            path.pop()
        return paths

    def _dict_stations(
        self,
        tree: Mapping[Station, Mapping],
        stations: Mapping[str, Station] = None
    ) -> Dict[str, Station]:
        """
        Get dictionary of stations (with Station IDs as keys).

        Args:
            tree (List[Mapping]): Tree of stations.
            stations (Mapping[str, Station]): Dictionary of stations (with Station IDs as keys).

        Returns:
            Dict[str, Station]: Dictionary of stations, with Station IDs as keys, and Station objects as values.
        """
        if stations is None:
            stations = {}

        for node in tree:
            org = node
            stations[org.id] = org
            self._dict_stations(tree[node], stations)
        return stations

    def __str__(self) -> str:
        """
        A human readable string representation of the corridor.

        Returns:
            str: A human readable string representation of the corridor.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the corridor.

        Returns:
            str: A debuggable string representation of the corridor.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'tree={self.tree}, '
            f'paths={self.paths}, '
            f'stations={self.stations})'
        )


class Line:
    """
    Sequence of stations being served by a train with a timetable.

    Attributes:
        id (str): Line ID.
        name (str): Line name.
        corridor (Corridor): Corridor ID where the Line belongs to.
        timetable (Mapping[str, Tuple[float, float]]): Dict with pairs of stations (origin, destination)
            with (origin ID, destination ID) as keys, and (origin time, destination time) as values.
        stations (List[Station]): List of Stations being served by the Line.
        stations_ids (List[str]): List of Station IDs being served by the Line.
        pairs (Mapping[Tuple[str, str], Tuple[Station, Station]]): Dict with pairs of stations (origin, destination)
            with (origin ID, destination ID) as keys, and (origin Station, destination Station) as values.
    """

    def __init__(self, id_: str, name: str, corridor: Corridor, timetable: Mapping[str, Tuple[float, float]]) -> None:
        """
        Initialize a Line object.

        Args:
            id_ (str): Line ID.
            name (str): Line name.
            corridor (Corridor): Corridor ID where the Line belongs to.
            timetable (Mapping[str, Tuple[float, float]]): Dict with pairs of stations (origin, destination)
                with (origin ID, destination ID) as keys, and (origin time, destination time) as values.
        """
        self.id = id_
        self.name = name
        self.corridor = corridor
        self.timetable = timetable
        self.stations = list(map(lambda sid: self.corridor.stations[sid], list(self.timetable.keys())))
        self.stations_ids = [station.id for station in self.stations]

    @cached_property
    def pairs(self) -> Dict[Tuple[str, str], Tuple[Station, Station]]:
        """
        Returns a dictionary with pairs of stations.
        
        Returns:
            Dict[Tuple[str, str], Tuple[Station, Station]]: Dictionary with pairs of stations (origin, destination)
                with (origin ID, destination ID) as keys, and (origin Station, destination Station) as values.
        """
        return {(a.id, b.id): (a, b) for i, a in enumerate(self.stations) for b in self.stations[i + 1:]}

    def __str__(self) -> str:
        """
        A human readable string representation of the line.

        Returns:
            str: A human readable string representation of the line.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the line.

        Returns:
            str: A debuggable string representation of the line.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'corridor={self.corridor}, '
            f'timetable={self.timetable}, '
            f'stations={self.stations})'
        )


class Seat:
    """
    Seat type of a train.

    Attributes:
        id (str): Seat ID.
        name (str): Seat type name.
        hard_type (int): Hard seat type.
        soft_type (int): Soft seat type.
    """

    def __init__(self, id_: str, name: str, hard_type: int, soft_type: int) -> None:
        """
        Initialize a Seat object.

        Args:
            id_ (str): Seat ID.
            name (str): Seat type name.
            hard_type (int): Hard seat type.
            soft_type (int): Soft seat type.
        """
        self.id = id_
        self.name = name
        self.hard_type = hard_type
        self.soft_type = soft_type

    def __str__(self) -> str:
        """
        A human readable string representation of the seat.

        Returns:
            str: A human readable string representation of the seat.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the seat.

        Returns:
            str: A debuggable string representation of the seat.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'hard_type={self.hard_type}, '
            f'soft_type={self.soft_type})'
        )


class RollingStock(object):
    """
    Locomotives, carriages, wagons, or other vehicles used on a railway.

    Attributes:
        id (str): Rolling Stock ID.
        name (str): Rolling Stock name.
        seats (Mapping[int, int]): Number of seats for each hard type.
        total_capacity (int): Total number of seats.
    """

    def __init__(self, id_: str, name: str, seats: Mapping[int, int]) -> None:
        """
        Constructor method for RollingStock class.

        Args:
            id_ (str): Rolling Stock ID.
            name (str): Rolling Stock name.
            seats (Mapping[int, int]): Number of seats for each hard type.
        """
        self.id = id_
        self.name = name
        self.seats = seats
        self.total_capacity = sum(seats.values())

    def __str__(self) -> str:
        """
        A human readable string representation of the rolling stock.

        Returns:
            str: A human readable string representation of the rolling stock.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the rolling stock.

        Returns:
            str: A debuggable string representation of the rolling stock.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'seats={self.seats}, '
            f'total_capacity={self.total_capacity})'
        )


class TSP:
    """
    Train Service Provider, a company that provides train services.

    Attributes:
        id (str): Train Service Provider ID.
        name (name): Name of the Train Service Provider.
        rolling_stock List[RollingStock]: List of RollingStock objects.
    """

    def __init__(self, id_: str, name: str, rolling_stock: List[RollingStock] = None) -> None:
        """
        Initialize a TSP object.

        Args:
            id_ (str): Train Service Provider ID.
            name (name): Name of the Train Service Provider.
            rolling_stock List[RollingStock]: List of RollingStock objects.
        """
        self.id = id_
        self.name = name
        self.rolling_stock = rolling_stock if rolling_stock is not None else []

    def __str__(self) -> str:
        """
        A human readable string representation of the TSP.

        Returns:
            str: A human readable string representation of the TSP.
        """
        return self.name
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the TSP.

        Returns:
            str: A debuggable string representation of the TSP.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'name={self.name}, '
            f'rolling_stock={self.rolling_stock})'
        )


class Service:
    """
    Travel options provided by a TSP between stations in a Line with a timetable.

    Capacity is defined by the Rolling Stock.
    Capacity constraints may apply between some pairs of stations.
    Different type of seats may be offered, each with a specific price.

    Attributes:
        id (str): Service ID.
        date (datetime.date): Day of service (year, month, daty, without time).
        line (Line): Line in which the service is provided.
        tsp (TSP): Train Service Provider which provides the service.
        time_slot (TimeSlot): Time Slot. Defines the start time of the service.
        schedule (Mapping[str, Tuple[datetime.timedelta, datetime.timedelta]]): Absolute schedule of the service per station.
        departure_time (Mapping[str, float]): Service departure time in hours per station.
        arrival_time (Mapping[str, float]): Service arrival time in hours per station.
        rolling_stock (RollingStock): Rolling Stock used in the service.
        capacity_constraints (Mapping[Tuple[str, str], Mapping[int, int]]): Constrained capacity (limit seats available
            between a specific pair of stations).
        lift_constraints (datetime.date): Date when capacity constraints are lifted.
        prices (Mapping[Tuple[str, str], Mapping[Seat, float]]): Prices for each pair of stations and each Seat type.
        seat_types (Mapping[str, Seat]): Seat types available in the service.
        tickets_sold_seats (Mapping[Seat, int]): Number of seats sold for each Seat type.
        tickets_sold_hard_types (Mapping[int, int]): Number of seats sold for each hard type.
        tickets_sold_pair_seats (Mapping[Tuple[str, str], Dict[Seat, int]]): Number of seats sold for each pair of stations
            and each Seat types.
        tickets_sold_pair_hard_types (Mapping[Tuple[str, str], Mapping[int, int]]): Number of seats sold for each pair of
            stations and each hard types.
        total_profit (float): Total profit of the service.
        profit_pair_seats (Mapping[Tuple[str, str], Mapping[Seat, float]]): Profit per pair of stations and each Seat type.
    """

    def __init__(
        self,
        id_: str,
        date: datetime.date,
        line: Line,
        tsp: TSP,
        time_slot: TimeSlot,
        rolling_stock: RollingStock,
        prices: Mapping[Tuple[str, str], Mapping[Seat, float]],
        capacity_constraints: Mapping[Tuple[str, str], Mapping[int, int]] = None,
        lift_constraints: int = DEFAULT_LIFT_CONSTRAINTS
    ) -> None:
        """
        Initialize a Service with a date, line, TSP, time slot, rolling stock and prices.

        Args:
            id_ (str): Service ID.
            date (datetime.date): Day of service (year, month, daty, without time).
            line (Line): Line in which the service is provided.
            tsp (TSP): Train Service Provider which provides the service.
            time_slot (TimeSlot): Time Slot. Defines the start time of the service.
            rolling_stock (RollingStock): Rolling Stock used in the service.
            prices (Mapping[Tuple[str, str], Mapping[Seat, float]]): Prices for each pair of stations and each Seat type.
            capacity_constraints (Mapping[Tuple[str, str], Mapping[int, int]]): Constrained capacity (limit seats available
                between a specific pair of stations).
            lift_constraints (int): Minimum anticipation (days) to lift capacity constraints.
        """
        self.id = id_
        self.date = date
        self.line = line
        self.tsp = tsp
        self.time_slot = time_slot
        self.schedule = self._get_absolute_schedule()
        self.arrival_time = {station.id: self.schedule[station.id][0].seconds / 3600 for station in self.line.stations}
        self.departure_time = {station.id: self.schedule[station.id][1].seconds / 3600 for station in self.line.stations}
        self.rolling_stock = rolling_stock
        self.capacity_constraints = capacity_constraints
        self.lift_constraints = self.date - datetime.timedelta(days=lift_constraints)
        self.prices = prices
        self._seat_types = tuple(dict.fromkeys([seat for seat_price in self.prices.values() for seat in seat_price.keys()]))
        self.seat_types = {seat.name: seat for seat in self._seat_types}
        self.tickets_sold_seats = {seat: 0 for seat in self._seat_types}
        self.tickets_sold_hard_types = {hard_type: 0 for hard_type in self.rolling_stock.seats.keys()}
        self.tickets_sold_pair_seats = {pair: {seat: 0 for seat in self._seat_types} for pair in self.line.pairs}
        self.tickets_sold_pair_hard_types = self._get_tickets_sold_pair_hard_type()
        self.total_profit = 0
        self.profit_pair_seats = {pair: {seat: 0 for seat in self._seat_types} for pair in self.line.pairs}
        self._pair_capacity = {
            pair: {hard_type: 0 for hard_type in self.rolling_stock.seats.keys()} for pair in self.line.pairs
        }

    def _get_tickets_sold_pair_hard_type(self) -> Mapping[Tuple[str, str], Mapping[int, int]]:
        """
        Private method to get the hard type tickets sold of the service.

        Returns:
            Mapping[Tuple[str, str], Mapping[int, int]]: Hard type tickets sold of the service.
        """
        tickets_sold_pair_hard_type = {}
        for pair in self.tickets_sold_pair_seats:
            tickets_sold_pair_hard_type[pair] = {}
            for seat in self.tickets_sold_pair_seats[pair].keys():
                if seat.hard_type not in tickets_sold_pair_hard_type[pair]:
                    tickets_sold_pair_hard_type[pair][seat.hard_type] = 0
                else:
                    tickets_sold_pair_hard_type[pair][seat.hard_type] += self.tickets_sold_pair_seats[pair][seat]
        return tickets_sold_pair_hard_type

    def _get_absolute_schedule(self) -> Mapping[str, Tuple[datetime.timedelta, datetime.timedelta]]:
        """
        Private method to get the absolute schedule of the service per station.

        Returns:
            Mapping[str, Tuple[datetime.timedelta, datetime.timedelta]]: Absolute schedule of the service per station.
        """
        absolute_schedule = {}
        for station, (departure_time, arrival_time) in self.line.timetable.items():
            abs_departure_time = datetime.timedelta(seconds=departure_time * 60) + self.time_slot.start
            abs_arrival_time = datetime.timedelta(seconds=arrival_time * 60) + self.time_slot.start
            absolute_schedule[station] = (abs_departure_time, abs_arrival_time)
        return absolute_schedule

    @cache
    def _get_affected_pairs(self, origin: str, destination: str) -> Set[Tuple[str, str]]:
        """
        Private method to get the pairs affected by origin-destination selection.

        Args:
            origin (str): Origin station ID.
            destination (str): Destination station ID.

        Returns:
            Set[Tuple[str, str]]: Set of pairs affected by origin-destination selection.
        """
        pairs = list(self.line.pairs.keys())

        # Get the index of the first pair which includes the origin station
        start_index = 0
        for i, pair in enumerate(pairs):
            if pair[0] == origin:
                start_index = i
                break
        # Get the index of the last pair which includes the destination station
        end_index = -1
        for i, pair in enumerate(pairs[::-1]):
            if pair[1] == destination:
                end_index = len(pairs) - i
                break

        affected_pairs = pairs[start_index:end_index]
        # Get stations between selected origin-destination
        origin_index = self.line.stations_ids.index(origin) + 1
        destination_index = self.line.stations_ids.index(destination)
        intermediate_stations = self.line.stations_ids[origin_index:destination_index]
        # Include pairs which depart between selected origin-destination
        affected_pairs.extend([pair for pair in pairs if pair[0] in intermediate_stations])
        return set(affected_pairs)

    def _tickets_available(self, origin: str, destination: str, seat: Seat):
        """
        Check if there are tickets available for the service without considering capacity constraints.

        Args:
            origin (str): Origin station ID.
            destination (str): Destination station ID.
            seat (Seat): Seat type.

        Returns:
            bool: True if there are tickets available, False otherwise.
        """
        # Check every pair capacity until the destination station is reached
        affected_pairs = self._get_affected_pairs(origin, destination)
        rolling_stock_seats = self.rolling_stock.seats[seat.hard_type]
        for pair in affected_pairs:
            if self._pair_capacity[pair][seat.hard_type] >= rolling_stock_seats:
                return False
        return True

    def buy_ticket(self, origin: str, destination: str, seat: Seat, purchase_date: datetime.date) -> bool:
        """
        Buy a ticket for the service.

        Args:
            origin (str): Origin station ID.
            destination (str): Destination station ID.
            seat (Seat): Seat type.
            purchase_date (datetime.date): Day of purchase of the ticket.

        Returns:
            bool: True if the ticket was bought, False otherwise.
        """
        if not self.tickets_available(origin, destination, seat, purchase_date):
            return False

        # Invalidate memoized tickets_available as the capacity will change
        self.tickets_available.cache_clear()

        # Check every pair capacity until the destination station is reached
        affected_pairs = self._get_affected_pairs(origin, destination)
        for pair in affected_pairs:
            self._pair_capacity[pair][seat.hard_type] += 1

        self.tickets_sold_pair_seats[(origin, destination)][seat] += 1
        self.tickets_sold_seats[seat] += 1
        self.tickets_sold_hard_types[seat.hard_type] += 1
        self.total_profit += self.prices[(origin, destination)][seat]
        self.profit_pair_seats[(origin, destination)][seat] += self.prices[(origin, destination)][seat]
        return True

    @cache
    def tickets_available(self, origin: str, destination: str, seat: Seat, purchase_date: datetime.date) -> bool:
        """
        Check if there are tickets available for the service.

        Args:
            origin (str): Origin station ID.
            destination (str): Destination station ID.
            seat (Seat): Seat type.
            purchase_date (datetime.date): Day of purchase of the ticket.

        Returns:
            bool: True if there are tickets available, False otherwise.
        """
        # Check if there are tickets available without considering capacity constraints
        pair_capacity = self._pair_capacity[(origin, destination)][seat.hard_type]
        tickets_available = self._tickets_available(origin=origin, destination=destination, seat=seat)
        # Check if there are tickets available considering capacity constraints
        if self.capacity_constraints and purchase_date < self.lift_constraints and (origin, destination) in self.capacity_constraints:
            constrained_capacity = self.capacity_constraints[(origin, destination)][seat.hard_type]
            if pair_capacity < constrained_capacity and tickets_available:
                return True
        return tickets_available
    
    def __str__(self) -> str:
        """
        A human readable string representation of the service.

        Returns:
            str: A human readable string representation of the service.
        """
        new_line = '\n\t\t'
        prices = ''.join(f'{new_line}{pair}: {{{", ".join(f"{seat}: {price}" for seat, price in seats.items())}}}' for pair, seats in self.prices.items())
        tickets_sold_seats = ''.join(f'{new_line}{seat}: {count}' for seat, count in self.tickets_sold_seats.items())
        tickets_sold_hard_types = ''.join(f'{new_line}{hard_type}: {count}' for hard_type, count in self.tickets_sold_hard_types.items())
        tickets_sold_pair_seats = ''.join(f'{new_line}{pair}: {{{", ".join(f"{seat}: {count}" for seat, count in seats.items())}}}' for pair, seats in self.tickets_sold_pair_seats.items())
        return (
            f'Service id: {self.id} \n'
            f'\tDate of service: {self.date} \n'
            f'\tStops: {[sta.id for sta in self.line.stations]} \n'
            f'\tLine times (relative): {list(self.line.timetable.values())} \n'
            f'\tLine times (absolute): {[(format_td(at), format_td(dt)) for at, dt in list(self.schedule.values())]} \n'
            f'\tTrain Service Provider: {self.tsp} \n'
            f'\tTime Slot: {self.time_slot} \n'
            f'\tRolling Stock: {self.rolling_stock} \n'
            f'\tPrices: {prices} \n'
            f'\tTickets sold (seats): {tickets_sold_seats} \n'
            f'\tTickets sold (hard type): {tickets_sold_hard_types} \n'
            f'\tTickets sold per each pair (seats): {tickets_sold_pair_seats} \n'
            f'\tCapacity constraints: {self.capacity_constraints} \n'
        )
    
    def __repr__(self) -> str:
        """
        A debuggable string representation of the service.

        Returns:
            str: A debuggable string representation of the service.
        """
        return (
            f'{self.__class__.__name__}('
            f'id={self.id}, '
            f'date={self.date}, '
            f'line={self.line}, '
            f'tsp={self.tsp}, '
            f'time_slot={self.time_slot}, '
            f'rolling_stock={self.rolling_stock}, '
            f'capacity_constraints={self.capacity_constraints}, '
            f'lift_constraints={self.lift_constraints}, '
            f'prices={self.prices})'
        )


class Supply:
    """
    List of Service's available in the system.

    Attributes:
        services (List[Service]): List of services available in the system.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initialize a Supply with a list of services.

        Args:
            services (List[Service]): List of services available in the system.
        """
        self.services = services
        self.stations = tuple(dict.fromkeys(station for service in services for station in service.line.corridor.stations.values()))
        self.time_slots = tuple(dict.fromkeys(service.time_slot for service in services))
        self.corridors = tuple(dict.fromkeys(service.line.corridor for service in services))
        self.lines = tuple(dict.fromkeys(service.line for service in services))
        self.seats = tuple(dict.fromkeys(seat for service in services for pair in service.prices.values() for seat in pair.keys()))
        self.rolling_stocks = tuple(dict.fromkeys(service.rolling_stock for service in services))
        self.tsps = tuple(dict.fromkeys(service.tsp for service in services))

    @classmethod
    def from_yaml(cls, path: str) -> 'Supply':
        """
        Class method to create a Supply object (List[Service]) from a YAML file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            Supply: Supply object.
        """
        with open(path, 'r') as file:
            data = yaml.load(file, Loader=yaml.CSafeLoader)

        stations = Supply._get_stations(data, key='stations')
        time_slots = Supply._get_time_slots(data, key='timeSlot')
        corridors = Supply._get_corridors(data, stations, key='corridor')
        lines = Supply._get_lines(data, corridors, key='line')
        seats = Supply._get_seats(data, key='seat')
        rolling_stock = Supply._get_rolling_stock(data, seats, key='rollingStock')
        tsps = Supply._get_tsps(data, rolling_stock, key='trainServiceProvider')
        services = Supply._get_services(data, lines, tsps, time_slots, seats, rolling_stock, key='service')

        return cls(list(services.values()))

    def get_stations_dict(self):
        """
        Get a dictionary of stations in the supply with the station id as key and the station name as value.

        Returns:
            Dict[str, str]: Dictionary of stations in the supply with the station id as key and the station name
            as value.
        """
        return {str(sta.id): sta.name for s in self.services for sta in s.line.stations}

    def filter_service_by_id(self, service_id: str) -> Service:
        """
        Filters a Service by ID.

        Args:
            service_id (str): Service ID.

        Returns:
            Service: Service object.
        """
        for service in self.services:
            if service.id == service_id:
                return service

    @cache
    def filter_services(self, origin: str, destination: str, date: datetime.date) -> List[Service]:
        """
        Filters a List of Services available in the system that meet the users requirements.

        Args:
            origin (str): Origin Station ID.
            destination (str): Destination Station ID.
            date (datetime.date): Date of service (day, month, year, without time).

        Returns:
            List[Service]: List of Service objects that meet the user requests.
        """
        filtered_services = []
        for service in self.services:
            if service.date == date and (origin, destination) in service.prices.keys():
                filtered_services.append(service)
        return filtered_services

    @classmethod
    def _get_stations(cls, data: Mapping[Any, Any], key: str = 'stations') -> Dict[str, Station]:
        """
        Private method to build a dict of Station objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'stations'.

        Returns:
            Dict[str, Station]: Dict of Station objects.
        """
        stations = {}
        for station in data[key]:
            assert all(station_fields in station.keys() for station_fields in ('id', 'name', 'short_name', 'city')), 'Incomplete Station data'
            lat, lon = tuple(station.get('coordinates', {'lat': None, 'lon': None}).values())
            if not lat or not lon:
                station_id = str(station['id'])
                stations[station_id] = Station(station_id, station['name'], station['city'], station['short_name'])
            else:
                coordinates = (float(lat), float(lon))
                stations[str(station['id'])] = Station(str(station['id']), station['name'], station['city'], station['short_name'], coordinates)
        return stations

    @classmethod
    def _get_time_slots(cls, data: Mapping[Any, Any], key: str = 'timeSlot') -> Dict[str, TimeSlot]:
        """
        Private method to build a dict of TimeSlot objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'timeSlot'.

        Returns:
            Dict[str, TimeSlot]: Dict of TimeSlot objects.
        """
        time_slots = {}
        for time_slot in data[key]:
            assert all(time_slot_fields in time_slot.keys() for time_slot_fields in ('id', 'start', 'end')), 'Incomplete TimeSlot data'
            time_slot_id = str(time_slot['id'])
            time_slots[time_slot_id] = TimeSlot(time_slot_id, get_time(time_slot['start']), get_time(time_slot['end']))
        return time_slots

    @classmethod
    def _get_corridors(
        cls,
        data: Mapping[Any, Any],
        stations: (Mapping[str, Station]),
        key: str = 'corridor'
    ) -> Dict[str, Corridor]:
        """
        Private method to build a dict of Corridor objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data as nested dict.
            stations (Mapping[str, Station]): Dict of Station objects.
            key (str): Key to access the data in the YAML file. Default: 'corridor'.

        Returns:
            Dict[str, Corridor]: Dict of Corridor objects.
        """

        def to_station(tree: Dict, sta_dict: Mapping[str, Station]) -> Dict[Station, Dict]:
            """
            Recursive function to build a tree of Station objects from a tree of station IDs.

            Args:
                tree (Mapping): Tree of station IDs.
                sta_dict (Mapping[str, Station]): Dict of Station objects {station_id: Station object}

            Returns:
                Dict[Station, Dict]: Tree of Station objects.
            """
            if not tree:
                return {}
            return {sta_dict[node]: to_station(tree[node], sta_dict) for node in tree}

        corridors = {}
        for corridor in data[key]:
            assert all(corridor_fields in corridor.keys() for corridor_fields in ('id', 'name', 'stations')), 'Incomplete Corridor data'

            tree_dictionary = convert_tree_to_dict(corridor['stations'])
            corr_stations_ids = set_stations_ids(tree_dictionary)
            assert all(station in stations.keys() for station in corr_stations_ids), 'Station not found in Station list'

            stations_tree = to_station(deepcopy(tree_dictionary), stations)
            corridor_id = str(corridor['id'])
            corridors[corridor_id] = Corridor(corridor_id, corridor['name'], stations_tree)
        return corridors

    @classmethod
    def _get_lines(cls, data: Mapping[Any, Any], corridors: Mapping[str, Corridor], key='line') -> Dict[str, Line]:
        """
        Private method to build a dict of Line objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data
            corridors (Mapping[str, Corridor]): Dict of Corridor objects.
            key (str): Key to access the data in the YAML file. Default: 'line'.

        Returns:
            Dict[str, Line]: Dict of Line objects.
        """
        lines = {}
        for line in data[key]:
            assert all(line_fields in line.keys() for line_fields in ('id', 'name', 'corridor', 'stops')), 'Incomplete Line data'

            corr_id = str(line['corridor'])
            assert corr_id in corridors.keys(), 'Corridor not found in Corridor list'
            corridor = corridors[corr_id]

            for station in line['stops']:
                assert all(stop_fields in station for stop_fields in ('station', 'arrival_time', 'departure_time')), 'Incomplete Stops data'

            corr_stations_ids = [station.id for station in corridor.stations.values()]
            assert all(station['station'] in corr_stations_ids for station in line['stops']), 'Station not found in Corridor list'

            timetable = {
                station['station']: (float(station['arrival_time']), float(station['departure_time']))
                for station in line['stops']
            }
            line_id = str(line['id'])
            lines[line_id] = Line(line_id, line['name'], corridor, timetable)
        return lines

    @classmethod
    def _get_seats(cls, data: Mapping[Any, Any], key: str = 'seat') -> Dict[str, Seat]:
        """
        Private method to build a dict of Seat objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data.
            key (str): Key to access the data in the YAML file. Default: 'seat'.

        Returns:
            Dict[str, Seat]: Dict of Seat objects.
        """
        seats = {}
        for seat in data[key]:
            assert all(seat_fields in seat.keys() for seat_fields in ('id', 'name', 'hard_type', 'soft_type')), 'Incomplete Seat data'
            seat_id = str(seat['id'])
            seats[seat_id] = Seat(seat_id, seat['name'], seat['hard_type'], seat['soft_type'])
        return seats

    @classmethod
    def _get_rolling_stock(
        cls,
        data: Mapping[Any, Any],
        seats: Mapping[str, Seat],
        key: str = 'rollingStock'
    ) -> Dict[str, RollingStock]:
        """
        Private method to build a dict of RollingStock objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data.
            seats (Mapping[str, Seat]): Dict of Seat objects.
            key (str): Key to access the data in the YAML file. Default: 'rollingStock'.

        Returns:
            Dict[str, RollingStock]: Dict of RollingStock objects.
        """
        rolling_stocks = {}
        for rolling_stock in data[key]:
            assert all(rolling_stock_fields in rolling_stock.keys() for rolling_stock_fields in ('id', 'name', 'seats')), 'Incomplete RollingStock data'

            for seat in rolling_stock['seats']:
                assert all(key in seat for key in ('hard_type', 'quantity')), 'Incomplete seats data for RS'

            assert all(seat['hard_type'] in [seat.hard_type for seat in seats.values()] for seat in
                       rolling_stock['seats']), 'Invalid hard_type for RS'

            rolling_stock_seats = {int(seat['hard_type']): int(seat['quantity']) for seat in rolling_stock['seats']}
            rolling_stock_id = str(rolling_stock['id'])
            rolling_stocks[rolling_stock_id] = RollingStock(rolling_stock_id, rolling_stock['name'], rolling_stock_seats)
        return rolling_stocks

    @classmethod
    def _get_tsps(
        cls,
        data: Mapping[Any, Any],
        rolling_stock: Mapping[str, RollingStock],
        key: str = 'trainServiceProvider'
    ) -> Dict[str, TSP]:
        """
        Private method to build a dict of TSP objects from YAML data.

        Args:
            data (Mapping[Any, Any])): YAML data
            rolling_stock (Mapping[str, RollingStock]): Dict of RollingStock objects.
            key (str): Key to access the data in the YAML file. Default: 'trainServiceProvider'.

        Returns:
            Dict[str, TSP]: Dict of TSP objects.
        """
        tsps = {}
        for tsp in data[key]:
            assert all(tsp_fields in tsp.keys() for tsp_fields in ('id', 'name', 'rolling_stock')), 'Incomplete TSP data'
            assert all(str(rolling_stock_id) in rolling_stock.keys() for rolling_stock_id in tsp['rolling_stock']), 'Unknown RollingStock ID'
            tsp_id = str(tsp['id'])
            tsps[tsp_id] = TSP(tsp_id, tsp['name'], [rolling_stock[str(rolling_stock_id)] for rolling_stock_id in tsp['rolling_stock']])
        return tsps

    @classmethod
    def _get_capacity_constraints(
        cls,
        service_line: Line,
        service_rolling_stock: RollingStock,
        yaml_capacity_constraints: Mapping
    ) -> Union[Dict, None]:
        """
        Private method to build a dict of capacity constraints from YAML data.

        Args:
            service_line (Line): Line object.
            yaml_capacity_constraints (Mapping[str, Any]): Dict of capacity constraints from YAML data.

        Returns:
            Union[Dict, None]: Dict of capacity constraints.
        """
        if yaml_capacity_constraints:
            capacity_constraints = {}
            for capacity_constraint in yaml_capacity_constraints:
                assert all(capacity_constraint_field in capacity_constraint for capacity_constraint_field in
                           ('origin', 'destination', 'seats')), 'Incomplete capacity constraints data for Service'
                assert all(station in service_line.corridor.stations.keys() for station in
                           (capacity_constraint['origin'],
                            capacity_constraint['destination'])), 'Invalid station in capacity constraints'

                for seat in capacity_constraint['seats']:
                    assert all(service in seat for service in ('hard_type', 'quantity')), 'Incomplete seats data for Service'
                    assert seat['hard_type'] in service_rolling_stock.seats.keys(), 'Invalid hard type in capacity constraints'

                origin_destination_tuple = (capacity_constraint['origin'], capacity_constraint['destination'])
                pair_constraints = {}
                for seat in capacity_constraint['seats']:
                    pair_constraints[seat['hard_type']] = seat['quantity']
                capacity_constraints[origin_destination_tuple] = pair_constraints
            return capacity_constraints
        return None

    @classmethod
    def _get_service_prices(
        cls,
        service_line: Line,
        seats: Mapping[str, Seat],
        yaml_service_prices: Mapping
    ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Private method to build a dict of service prices from YAML data.

        Args:
            service_line (Line): Line object.
            seats (Mapping[str, Seat]): Dict of Seat objects.
            yaml_service_prices (Mapping[str, Any]): Dict of service prices from YAML data.

        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Dict of service prices.
        """
        service_prices = {}
        for trip in yaml_service_prices:
            assert all(trip_fields in trip.keys() for trip_fields in ('origin', 'destination', 'seats')), 'Incomplete Service prices'

            origin = trip['origin']
            destination = trip['destination']
            assert all(station in service_line.corridor.stations.keys() for station in (origin, destination)), 'Invalid station in Service'
            for seat in trip['seats']:
                assert all(key in seat for key in ('seat', 'price')), 'Incomplete seats data for Service'
                assert str(seat['seat']) in seats, 'Invalid seat in Service prices'

            prices = {seats[str(seat['seat'])]: seat['price'] for seat in trip['seats']}
            service_prices[(origin, destination)] = prices
        return service_prices

    @classmethod
    def _get_services(
        cls,
        data: Mapping[Any, Any],
        lines: Mapping[str, Line],
        tsps: Mapping[str, TSP],
        time_slots: Mapping[str, TimeSlot],
        seats: Mapping[str, Seat],
        rolling_stock: Mapping[str, RollingStock],
        key: str = 'service'
    ) -> Dict[str, Service]:
        """
        Private method to build a dict of Service objects from YAML data.

        Args:
            data (Mapping[Any, Any]): YAML data
            lines (Mapping[str, Line]): Dict of Line objects.
            tsps (Mapping[str, TSP]): Dict of TSP objects.
            time_slots (Mapping[str, TimeSlot]): Dict of TimeSlot objects.
            seats (Mapping[str, Seat]): Dict of Seat objects.
            rolling_stock (Mapping[str, RollingStock]): Dict of RollingStock objects.
            key (str): Key to access the data in the YAML file. Default: 'service'.

        Returns:
            Dict[str, Service]: Dict of Service objects.
        """
        services = {}
        for service in data[key]:
            service_keys = (
                'id', 'date', 'line', 'train_service_provider', 'time_slot', 'rolling_stock',
                'origin_destination_tuples', 'capacity_constraints'
            )
            assert all(service_fields in service.keys() for service_fields in service_keys), 'Incomplete Service data'

            service_id = str(service['id'])
            service_date = get_date(service['date'])
            service_line_id = str(service['line'])
            assert service_line_id in lines.keys(), 'Line not found'
            service_line = lines[service_line_id]

            tsp_id = str(service['train_service_provider'])
            assert tsp_id in tsps.keys(), 'TSP not found'
            service_tsp = tsps[tsp_id]

            time_slot_id = str(service['time_slot'])
            assert time_slot_id in time_slots.keys(), 'TimeSlot not found'
            service_time_slot = time_slots[time_slot_id]

            rolling_stock_id = str(service['rolling_stock'])
            assert rolling_stock_id in rolling_stock.keys(), 'RollingStock not found'
            service_rolling_stock = rolling_stock[rolling_stock_id]

            yaml_service_prices = service['origin_destination_tuples']
            service_prices = cls._get_service_prices(
                service_line, seats, yaml_service_prices
            )

            yaml_capacity_constraints = service['capacity_constraints']
            capacity_constraints = cls._get_capacity_constraints(
                service_line, service_rolling_stock, yaml_capacity_constraints
            )

            services[service_id] = Service(
                service_id,
                service_date,
                service_line,
                service_tsp,
                service_time_slot,
                service_rolling_stock,
                service_prices,
                capacity_constraints
            )
        return services
