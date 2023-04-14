"""Entities for the supply module."""

import datetime
import yaml

from src.robin.supply.utils import get_time, get_date, format_td, set_stations_ids, convert_tree_to_dict

from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union


class Station:
    """
    Railway facility where trains stop to load or unload passengers, freight or both.

    Attributes:
        id (str): Station ID.
        name (str): Station name.
        city (str): Station city.
        shortname (str): Station short name.
        coords (Tuple[float, float]): Station coordinates (latitude, longitude).
    """

    def __init__(self, id_: str, name: str, city: str, shortname: str, coords: Tuple[float, float] = None) -> None:
        """
        Initialize a Station object.

        Args:
            id_ (str): Station ID.
            name (str): Station name.
            city (str): Station city.
            shortname (str): Station short name.
            coords (Tuple[float, float]): Station coordinates (latitude, longitude).
        """
        self.id = id_
        self.name = name
        self.city = city
        self.shortname = shortname
        self.coords = coords

    def add_coords(self, coords: Tuple[float, float]) -> None:
        """
        Add coordinates to a Station object.

        Args:
            coords (Tuple[float, float]): Station coordinates (latitude, longitude).
        """
        self.coords = coords

    def __str__(self) -> str:
        """
        String representation of a Station object.

        Returns:
            str: String representation of a Station object.
        """
        return f'[{self.id}, {self.name}, {self.shortname}, {self.coords}]'


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
        Initialize a TimeSlot object.

        Args:
            id_ (str): Time slot ID.
            start (datetime.timedelta): Time slot start time.
            end (datetime.timedelta): Time slot end time.
        """
        self.id = id_
        self.start = start
        self.end = end
        self.class_mark = self._get_class_mark()
        self.size = self._get_size()

    def _get_class_mark(self) -> datetime.timedelta:
        """
        Get class mark of time slot.

        Returns:
            datetime.timedelta: Time slot class mark.
        """
        if self.end < self.start:
            return (self.start + self.end + datetime.timedelta(days=1)) / 2 - datetime.timedelta(days=1)
        return (self.start + self.end) / 2

    def _get_size(self) -> datetime.timedelta:
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
        String representation of a TimeSlot object.

        Returns:
            str: String representation of a TimeSlot object.
        """
        return f'[{self.id}, {self.start}, {self.end}, {self.class_mark}, {self.size}]'


class Corridor:
    """
    Tree of stations.

    Example:
        {'MAD': {'BAR': {}, 'BIL': {}}}
        *In the real tree, the Station objects are used instead of the Station IDs

    Attributes:
        id (str): Corridor ID.
        name (str): Corridor name.
        tree (Dict[Station, Dict]): Tree of stations (with Station objects).
        paths (List[List[Station]]): List of paths (list of stations).
        stations (Dict[str, Station]): Dictionary of stations (with Station IDs as keys).
    """

    def __init__(self, id_: str, name: str, tree: Dict[Station, Dict]) -> None:
        """
        Initialize a Corridor object.

        Args:
            id_ (str): Corridor ID.
            name (str): Corridor name.
            tree (Dict[Station, Dict]): Tree of stations (with Station objects).
        """
        self.id = id_
        self.name = name
        self.tree = tree
        self.paths = self._get_paths(self.tree)
        self.stations = self._dict_stations(self.tree)

    def _get_paths(
            self,
            tree: Dict[Station, Dict],
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

    def _dict_stations(self, tree: Dict[Station, Dict], sta=None) -> Dict[str, Station]:
        """
        Get dictionary of stations (with Station IDs as keys).

        Args:
            tree (List[Dict]): Tree of stations.

        Returns:
            Dict[str, Station]: Dictionary of stations, with Station IDs as keys, and Station objects as values.
        """
        if sta is None:
            sta = {}

        for node in tree:
            org = node
            sta[org.id] = org
            self._dict_stations(tree[node], sta)

        return sta

    def __str__(self) -> str:
        """
        String representation of a Corridor object.

        Returns:
            str: String representation of a Corridor object.
        """
        return f'[{self.id}, {self.name}, {self.stations}]'


class Line:
    """
    Sequence of stations being served by a train with a timetable.

    Attributes:
        id (str): Line ID.
        name (str): Line name.
        corridor (Corridor): Corridor ID where the Line belongs to.
        timetable (Dict[str, Tuple[float, float]]): Dict with pairs of stations (origin, destination)
            with (origin ID, destination ID) as keys, and (origin time, destination time) as values.
        stations (List[Station]): List of Stations being served by the Line.
        pairs (Dict[Tuple[str, str], Tuple[Station, Station]]): Dict with pairs of stations (origin, destination)
            with (origin ID, destination ID) as keys, and (origin Station, destination Station) as values.
    """

    def __init__(self, id_: str, name: str, corridor: Corridor, timetable: Dict[str, Tuple[float, float]]) -> None:
        """
        Initialize a Line object.

        Args:
            id_ (str): Line ID.
            name (str): Line name.
            corridor (Corridor): Corridor ID where the Line belongs to.
            timetable (Dict[str, Tuple[float, float]]): Dict with pairs of stations (origin, destination)
                with (origin ID, destination ID) as keys, and (origin time, destination time) as values.
        """
        self.id = id_
        self.name = name
        self.corridor = corridor
        self.timetable = timetable
        self.stations = list(map(lambda sid: self.corridor.stations[sid], list(self.timetable.keys())))
        self.pairs = self._get_pairs()

    def _get_pairs(self) -> Dict[Tuple[str, str], Tuple[Station, Station]]:
        """
        Private method to get each pair of stations of the Line, using the station list.

        Returns:
            Dict[Tuple[str, str], Tuple[Station, Station]]: Dict with pairs of stations (origin, destination).
        """
        return {(a.id, b.id): (a, b) for i, a in enumerate(self.stations) for b in self.stations[i + 1:]}

    def __str__(self) -> str:
        """
        String representation of the Line object.

        Returns:
            str: String representation of the Line object.
        """
        return f'[{self.id}, {self.name}, Corridor id: {self.corridor}, {self.timetable}]'


class Seat:
    """
    Seat type of train.

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
        Returns a human readable string representation of the Seat object.

        TODO: __str__ and __repr__ are mixed up. __str__ should be human readable, __repr__ should be debuggable.

        Returns:
            str: Human readable string representation of the Seat object.
        """
        return f'[{self.id}, {self.name}, {self.hard_type}, {self.soft_type}]'

    def __repr__(self) -> str:
        """
        Returns the debuggable string representation of the Seat object.

        TODO: __str__ and __repr__ are mixed up. __str__ should be human readable, __repr__ should be debuggable.

        Returns:
            str: Debuggable string representation of the Seat object.
        """
        return self.name


class RollingStock(object):
    """
    Locomotives, Carriages, Wagons, or other vehicles used on a railway.

    Attributes:
        id (str): Rolling Stock ID.
        name (str): Rolling Stock name.
        seats (Dict[int, int]): Number of seats for each hard type.
        total_capacity (int): Total number of seats.
    """

    def __init__(self, id_: str, name: str, seats: Dict[int, int]) -> None:
        """
        Constructor method for RollingStock class.

        Args:
            id_ (str): Rolling Stock ID.
            name (str): Rolling Stock name.
            seats (Dict[int, int]): Number of seats for each hard type.
        """
        self.id = id_
        self.name = name
        self.seats = seats
        self.total_capacity = sum(seats.values())

    def __str__(self) -> str:
        """
        String representation of the RollingStock object.

        Returns:
            str: String representation of the RollingStock object.
        """
        return f'[{self.id}, {self.name}, {self.seats}]'


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

    def add_rolling_stock(self, rolling_stock: RollingStock) -> None:
        """
        Add a RollingStock object to the TSP.

        Args:
            rolling_stock (RollingStock): RollingStock object to add.
        """
        self.rolling_stock.append(rolling_stock)

    def __str__(self) -> str:
        """
        String representation of the TSP object.

        Returns:
            str: String representation of the TSP object.
        """
        return f'[{self.id}, {self.name}, {[rolling_stock.id for rolling_stock in self.rolling_stock]}]'


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
        schedule (List[Tuple[datetime.timedelta, datetime.timedelta]]): List of tuples with arrival-departure times.
        service_departure_time (float): Service departure time in hours.
        service_arrival_time (float): Service arrival time in hours.
        rolling_stock (RollingStock): Rolling Stock used in the service.
        capacity_constraints (Dict[Tuple[str, str], Dict[int, int]]): Constrained capacity (limit seats available
            between a specific pair of stations).
        lift_constraints (int): Minimum anticipation (days) to lift capacity constraints.
        prices (Dict[Tuple[str, str], Dict[Seat, float]]): Prices for each pair of stations and each Seat type.
        seat_types (Dict[str, Seat]): Seat types available in the service.
        tickets_sold_seats (Dict[Seat, int]): Number of seats sold for each Seat type.
        tickets_sold_hard_types (Dict[int, int]): Number of seats sold for each hard type.
        tickets_sold_pair_seats (Dict[Tuple[str, str], Dict[Seat, int]]): Number of seats sold for each pair of stations
            and each Seat types.
        tickets_sold_pair_hard_types (Dict[Tuple[str, str], Dict[int, int]]): Number of seats sold for each pair of
            stations and each hard types.
    """

    def __init__(
            self,
            id_: str,
            date: datetime.date,
            line: Line,
            tsp: TSP,
            time_slot: TimeSlot,
            rolling_stock: RollingStock,
            prices: Dict[Tuple[str, str], Dict[Seat, float]],
            capacity_constraints: Dict[Tuple[str, str], Dict[int, int]] = None,
            lift_constraints: int = 1
    ) -> None:
        """
        Initialize a Service object.

        Args:
            id_ (str): Service ID.
            date (datetime.date): Day of service (year, month, daty, without time).
            line (Line): Line in which the service is provided.
            tsp (TSP): Train Service Provider which provides the service.
            time_slot (TimeSlot): Time Slot. Defines the start time of the service.
            rolling_stock (RollingStock): Rolling Stock used in the service.
            prices (Dict[Tuple[str, str], Dict[Seat, float]]): Prices for each pair of stations and each Seat type.
            capacity_constraints (Dict[Tuple[str, str], Dict[int, int]]): Constrained capacity (limit seats available
                between a specific pair of stations).
            lift_constraints (int): Minimum anticipation (days) to lift capacity constraints.
        """
        self.id = id_
        self.date = date
        self.line = line
        self.tsp = tsp
        self.time_slot = time_slot
        self.schedule = self._get_absolute_schedule()
        self.service_departure_time = self.schedule[0][0].seconds / 3600  # Service departure time in hours
        self.service_arrival_time = self.schedule[-1][0].seconds / 3600  # Service arrival time in hours
        self.rolling_stock = rolling_stock
        self.capacity_constraints = capacity_constraints
        self.lift_constraints = lift_constraints
        self.prices = prices
        self._seat_types = set([seat for seat_price in self.prices.values() for seat in seat_price.keys()])
        self.seat_types = {seat.name: seat for seat in self._seat_types}
        self.tickets_sold_seats = {seat: 0 for seat in self._seat_types}
        self.tickets_sold_hard_types = {hard_type: 0 for hard_type in self.rolling_stock.seats.keys()}
        self.tickets_sold_pair_seats = {pair: {seat: 0 for seat in self._seat_types} for pair in self.line.pairs}
        self.tickets_sold_pair_hard_types = self._get_tickets_sold_pair_hard_type()
        self._pair_capacity = {pair: {hard_type: 0 for hard_type in self.rolling_stock.seats.keys()} for pair in
                               self.line.pairs}

    def _get_tickets_sold_pair_hard_type(self) -> Dict[Tuple[str, str], Dict[int, int]]:
        """
        Private method to get the hard type tickets sold of the service.

        Returns:
            Dict[Tuple[str, str], Dict[int, int]]: Hard type tickets sold of the service.
        """
        tickets_sold_pair_hard_type = {}
        for p in self.tickets_sold_pair_seats:
            tickets_sold_pair_hard_type[p] = {}
            for s in self.tickets_sold_pair_seats[p].keys():
                if s.hard_type not in tickets_sold_pair_hard_type[p]:
                    tickets_sold_pair_hard_type[p][s.hard_type] = 0
                else:
                    tickets_sold_pair_hard_type[p][s.hard_type] += self.tickets_sold_pair_seats[p][s]
        return tickets_sold_pair_hard_type

    def _get_absolute_schedule(self) -> List[Tuple[datetime.timedelta, datetime.timedelta]]:
        """
        Private method to get the absolute schedule of the service, using the relative schedule and the time slot start time.

        Returns:
            List[Tuple[datetime.timedelta, datetime.timedelta]]: Absolute schedule of the service.
        """
        absolute_schedule = []
        for dt, at in list(self.line.timetable.values()):
            abs_dt = datetime.timedelta(seconds=dt * 60) + self.time_slot.start
            abs_at = datetime.timedelta(seconds=at * 60) + self.time_slot.start
            absolute_schedule.append((abs_dt, abs_at))
        return absolute_schedule

    def buy_ticket(self, origin: str, destination: str, seat: Seat, anticipation: int) -> bool:
        """
        Buy a ticket for the service.

        Args:
            self (Service): Service object.
            origin (str): Origin station ID.
            destination (str): Destination station ID.
            seat (Seat): Seat type.
            anticipation (int): Days of anticipation in the purchase of the ticket.

        Returns:
            bool: True if the ticket was bought, False otherwise.
        """
        stations_ids = list(self.line.timetable.keys())
        service_route = set(range(stations_ids.index(origin), stations_ids.index(destination)))
        if not self.tickets_available(origin, destination, seat, anticipation):
            return False

        for pair in self.line.pairs:  # pairs attribute is a dictionary with all the pairs of stations
            origin_id, destination_id = pair
            stations_in_pair = set(range(stations_ids.index(origin_id), stations_ids.index(destination_id)))
            # TODO: Check test_supply.py
            if service_route.intersection(stations_in_pair):
                if self._pair_capacity[pair][seat.hard_type] < self.rolling_stock.seats[seat.hard_type]:
                    self._pair_capacity[pair][seat.hard_type] += 1

        self.tickets_sold_pair_seats[(origin, destination)][seat] += 1
        self.tickets_sold_seats[seat] += 1
        self.tickets_sold_hard_types[seat.hard_type] += 1

        return True

    def tickets_available(self, origin: str, destination: str, seat: Seat, anticipation: int) -> bool:
        """
        Check if there are tickets available for the service.

        Args:
            self (Service): Service object.
            origin (str): Origin station ID.
            destination (str): Destination station ID.
            seat (Seat): Seat type.
            anticipation (int): Days of anticipation in the purchase of the ticket.

        Returns:
            bool: True if there are tickets available, False otherwise.
        """
        occupied_seats = self._pair_capacity[(origin, destination)][seat.hard_type]

        if self.capacity_constraints and anticipation > self.lift_constraints:
            if (origin, destination) in self.capacity_constraints:
                constrained_capacity = self.capacity_constraints[(origin, destination)][seat.hard_type]
                if occupied_seats < constrained_capacity:
                    return True
        else:
            max_capacity = self.rolling_stock.seats[seat.hard_type]
            if occupied_seats < max_capacity:
                return True

        return False

    def __str__(self) -> str:
        """
        String representation of the service.

        Returns:
            str: String representation of the service.
        """
        new_line = '\n\t\t'
        return (
            f'Service id: {self.id} \n'
            f'\tDate of service: {self.date} \n'
            f'\tStops: {[sta.id for sta in self.line.stations]} \n'
            f'\tLine times (relative): {list(self.line.timetable.values())} \n'
            f'\tLine times (absolute): {[(format_td(at), format_td(dt)) for at, dt in self.schedule]} \n'
            f'\tTrain Service Provider: {self.tsp} \n'
            f'\tTime Slot: {self.time_slot} \n'
            f'\tRolling Stock: {self.rolling_stock} \n'
            f'\tPrices: \n'
            f'\t\t{new_line.join(f"{key}: {value}" for key, value in self.prices.items())} \n'
            f'\tTickets sold (seats): {self.tickets_sold_seats} \n'
            f'\tTickets sold (hard type): {self.tickets_sold_hard_types} \n'
            f'\tTickets sold per each pair (seats): {self.tickets_sold_pair_seats} \n'
            f'\tTickets sold per each pair (hard type): {self.tickets_sold_pair_hard_types} \n'
            f'\tCapacity constraints: {self.capacity_constraints} \n'
        )


class Supply:
    """
    List of Service's available in the system.

    Attributes:
        services List[Service]: List of services available in the system.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initialize a Supply object.

        Args:
            services (List[Service]): List of services available in the system.
        """
        self.services = services

    @classmethod
    def from_yaml(cls, path: str) -> 'Supply':
        """
        Class method to create a Supply object (List[Service]) from a yaml file.

        Args:
            path (str): Path to the yaml file.

        Returns:
            Supply: Supply object.
        """
        with open(path, 'r') as file:
            data = yaml.safe_load(file)

        stations = Supply._get_stations(data, key='stations')
        time_slots = Supply._get_time_slots(data, key='timeSlot')
        corridors = Supply._get_corridors(data, stations, key='corridor')
        lines = Supply._get_lines(data, corridors, key='line')
        seats = Supply._get_seats(data, key='seat')
        rolling_stock = Supply._get_rolling_stock(data, seats, key='rollingStock')
        tsps = Supply._get_tsps(data, rolling_stock, key='trainServiceProvider')
        services = Supply._get_services(data, lines, tsps, time_slots, seats, rolling_stock, key='service')

        return cls(list(services.values()))

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
        for s in self.services:
            if s.date == date and (origin, destination) in s.line.pairs.keys():
                filtered_services.append(s)
        return filtered_services

    @classmethod
    def _get_stations(cls, data: Dict[Any, Any], key: str = 'stations') -> Dict[str, Station]:
        """
        Private method to build a dict of Station objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'stations'.

        Returns:
            Dict[str, Station]: Dict of Station objects.
        """
        stations = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'short_name', 'city')), "Incomplete Station data"
            lat, lon = tuple(s.get('coordinates', {'lat': None, 'lon': None}).values())
            if not lat or not lon:
                station_id = str(s['id'])
                stations[station_id] = Station(station_id, s['name'], s['city'], s['short_name'])
            else:
                coords = (float(lat), float(lon))
                stations[str(s['id'])] = Station(str(s['id']), s['name'], s['city'], s['short_name'], coords)
        return stations

    @classmethod
    def _get_time_slots(cls, data: Dict[Any, Any], key: str = 'timeSlot') -> Dict[str, TimeSlot]:
        """
        Private method to build a dict of TimeSlot objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data as nested dict.
            key (str): Key to access the data in the YAML file. Default: 'timeSlot'.

        Returns:
            Dict[str, TimeSlot]: Dict of TimeSlot objects.
        """
        time_slots = {}
        for time_slot in data[key]:
            assert all(k in time_slot.keys() for k in ('id', 'start', 'end')), "Incomplete TimeSlot data"
            time_slot_id = str(time_slot['id'])
            time_slots[time_slot_id] = TimeSlot(time_slot_id, get_time(time_slot['start']), get_time(time_slot['end']))
        return time_slots

    @classmethod
    def _get_corridors(
            cls,
            data: Dict[Any, Any],
            stations: (Dict[str, Station]),
            key: str = 'corridor'
    ) -> Dict[str, Corridor]:
        """
        Private method to build a dict of Corridor objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data as nested dict.
            stations (Dict[str, Station]): Dict of Station objects.
            key (str): Key to access the data in the YAML file. Default: 'corridor'.

        Returns:
            Dict[str, Corridor]: Dict of Corridor objects.
        """

        def to_station(tree: Dict, sta_dict: Dict[str, Station]) -> Dict[Station, Dict]:
            """
            Recursive function to build a tree of Station objects from a tree of station IDs.

            Args:
                tree (Dict): Tree of station IDs.
                sta_dict (Dict[str, Station]): Dict of Station objects {station_id: Station object}

            Returns:
                Dict[Station, Dict]: Tree of Station objects.
            """
            if not tree:
                return {}
            return {sta_dict[node]: to_station(tree[node], sta_dict) for node in tree}

        corridors = {}
        for c in data[key]:
            assert all(k in c.keys() for k in ('id', 'name', 'stations')), "Incomplete Corridor data"

            tree_dictionary = convert_tree_to_dict(c['stations'])
            corr_stations_ids = set_stations_ids(tree_dictionary)
            assert all(s in stations.keys() for s in corr_stations_ids), "Station not found in Station list"

            stations_tree = to_station(deepcopy(tree_dictionary), stations)
            corridor_id = str(c['id'])
            corridors[corridor_id] = Corridor(corridor_id, c['name'], stations_tree)

        return corridors

    @classmethod
    def _get_lines(cls, data: Dict[Any, Any], corridors: Dict[str, Corridor], key='line') -> Dict[str, Line]:
        """
        Private method to build a dict of Line objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data
            corridors (Dict[str, Corridor]): Dict of Corridor objects.
            key (str): Key to access the data in the YAML file. Default: 'line'.

        Returns:
            Dict[str, Line]: Dict of Line objects.
        """
        lines = {}
        for ln in data[key]:
            assert all(k in ln.keys() for k in ('id', 'name', 'corridor', 'stops')), 'Incomplete Line data'

            corr_id = str(ln['corridor'])
            assert corr_id in corridors.keys(), 'Corridor not found in Corridor list'
            corr = corridors[corr_id]

            for stn in ln['stops']:
                assert all(k in stn for k in ('station', 'arrival_time', 'departure_time')), 'Incomplete Stops data'

            corr_stations_ids = [s.id for s in corr.stations.values()]
            assert all(s['station'] in corr_stations_ids for s in ln['stops']), 'Station not found in Corridor list'

            timetable = {s['station']: (float(s['arrival_time']), float(s['departure_time']))
                         for s in ln['stops']}
            line_id = str(ln['id'])
            lines[line_id] = Line(line_id, ln['name'], corr, timetable)

        return lines

    @classmethod
    def _get_seats(cls, data: Dict[Any, Any], key: str = 'seat') -> Dict[str, Seat]:
        """
        Private method to build a dict of Seat objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data.
            key (str): Key to access the data in the YAML file. Default: 'seat'.

        Returns:
            Dict[str, Seat]: Dict of Seat objects.
        """
        seats = {}
        for s in data[key]:
            assert all(k in s.keys() for k in ('id', 'name', 'hard_type', 'soft_type')), 'Incomplete Seat data'
            seat_id = str(s['id'])
            seats[seat_id] = Seat(seat_id, s['name'], s['hard_type'], s['soft_type'])
        return seats

    @classmethod
    def _get_rolling_stock(
            cls,
            data: Dict[Any, Any],
            seats: Dict[str, Seat],
            key: str = 'rollingStock'
    ) -> Dict[str, RollingStock]:
        """
        Private method to build a dict of RollingStock objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data.
            seats (Dict[str, Seat]): Dict of Seat objects.
            key (str): Key to access the data in the YAML file. Default: 'rollingStock'.

        Returns:
            Dict[str, RollingStock]: Dict of RollingStock objects.
        """
        rolling_stocks = {}
        for rolling_stock in data[key]:
            assert all(k in rolling_stock.keys() for k in ('id', 'name', 'seats')), 'Incomplete RollingStock data'

            for seat in rolling_stock['seats']:
                assert all(key in seat for key in ('hard_type', 'quantity')), 'Incomplete seats data for RS'

            assert all(seat['hard_type'] in [seat.hard_type for seat in seats.values()] for seat in
                       rolling_stock['seats']), 'Invalid hard_type for RS'

            rolling_stock_seats = {int(seat['hard_type']): int(seat['quantity']) for seat in rolling_stock['seats']}
            rolling_stock_id = str(rolling_stock['id'])
            rolling_stocks[rolling_stock_id] = RollingStock(rolling_stock_id,
                                                            rolling_stock['name'],
                                                            rolling_stock_seats)

        return rolling_stocks

    @classmethod
    def _get_tsps(
            cls,
            data: Dict[Any, Any],
            rolling_stock: Dict[str, RollingStock],
            key: str = 'trainServiceProvider'
    ) -> Dict[str, TSP]:
        """
        Private method to build a dict of TSP objects from YAML data.

        Args:
            data (Dict[Any, Any])): YAML data
            rolling_stock (Dict[str, RollingStock]): Dict of RollingStock objects.
            key (str): Key to access the data in the YAML file. Default: 'trainServiceProvider'.

        Returns:
            Dict[str, TSP]: Dict of TSP objects.
        """
        tsps = {}
        for tsp in data[key]:
            assert all(k in tsp.keys() for k in ('id', 'name', 'rolling_stock')), 'Incomplete TSP data'
            assert all(str(i) in rolling_stock.keys() for i in tsp['rolling_stock']), 'Unknown RollingStock ID'
            tsp_id = str(tsp['id'])
            tsps[tsp_id] = TSP(tsp_id, tsp['name'], [rolling_stock[str(rs_id)] for rs_id in tsp['rolling_stock']])
        return tsps

    @classmethod
    def _get_capacity_constraints(
            cls,
            service_line: Line,
            service_rolling_stock: RollingStock,
            yaml_capacity_constraints: Dict
    ) -> Union[Dict, None]:
        """
        Private method to build a dict of capacity constraints from YAML data.

        Args:
            service_line (Line): Line object.
            yaml_capacity_constraints (Dict[str, Any]): Dict of capacity constraints from YAML data.

        Returns:
            Union[Dict, None]: Dict of capacity constraints.
        """
        if yaml_capacity_constraints:
            capacity_constraints = {}
            for capacity_constraint in yaml_capacity_constraints:
                assert all(k in capacity_constraint for k in
                           ('origin', 'destination', 'seats')), 'Incomplete capacity constraints data for Service'
                assert all(s in service_line.corridor.stations.keys() for s in
                           (capacity_constraint['origin'],
                            capacity_constraint['destination'])), 'Invalid station in capacity constraints'

                for seat in capacity_constraint['seats']:
                    assert all(k in seat for k in ('hard_type', 'quantity')), 'Incomplete seats data for Service'
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
            seats: Dict[str, Seat],
            yaml_service_prices: Dict
    ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Private method to build a dict of service prices from YAML data.

        Args:
            service_line (Line): Line object.
            seats (Dict[str, Seat]): Dict of Seat objects.
            yaml_service_prices (Dict[str, Any]): Dict of service prices from YAML data.

        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Dict of service prices.
        """
        service_prices = {}
        for pair in yaml_service_prices:
            assert all(k in pair.keys() for k in ('origin', 'destination', 'seats')), 'Incomplete Service prices'

            origin = pair['origin']
            destination = pair['destination']
            assert all(s in service_line.corridor.stations.keys() for s in (origin, destination)), 'Invalid station in Service'
            for seat in pair['seats']:
                assert all(key in seat for key in ('seat', 'price')), 'Incomplete seats data for Service'
                assert str(seat['seat']) in seats, 'Invalid seat in Service prices'

            prices = {seats[str(seat['seat'])]: seat['price'] for seat in pair['seats']}
            service_prices[(origin, destination)] = prices

        return service_prices

    @classmethod
    def _get_services(
            cls,
            data: Dict[Any, Any],
            lines: Dict[str, Line],
            tsps: Dict[str, TSP],
            time_slots: Dict[str, TimeSlot],
            seats: Dict[str, Seat],
            rolling_stock: Dict[str, RollingStock],
            key: str = 'service'
    ) -> Dict[str, Service]:
        """
        Private method to build a dict of Service objects from YAML data.

        Args:
            data (Dict[Any, Any]): YAML data
            lines (Dict[str, Line]): Dict of Line objects.
            tsps (Dict[str, TSP]): Dict of TSP objects.
            time_slots (Dict[str, TimeSlot]): Dict of TimeSlot objects.
            seats (Dict[str, Seat]): Dict of Seat objects.
            rolling_stock (Dict[str, RollingStock]): Dict of RollingStock objects.
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
            assert all(k in service.keys() for k in service_keys), 'Incomplete Service data'

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
            service_prices = cls._get_service_prices(service_line=service_line,
                                                     seats=seats,
                                                     yaml_service_prices=yaml_service_prices)

            yaml_capacity_constraints = service['capacity_constraints']
            capacity_constraints = cls._get_capacity_constraints(service_line=service_line,
                                                                 service_rolling_stock=service_rolling_stock,
                                                                 yaml_capacity_constraints=yaml_capacity_constraints)

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
