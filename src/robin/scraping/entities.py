"""Entities for the scraping module."""

import datetime
import numpy as np
import pandas as pd
import yaml

from robin.scraping.exceptions import InvalidHardTypesException
from robin.scraping.constants import (
    OUTPUT_SUPPLY_PATH, PRICES_COLUMNS, RENFE_STATIONS_PATH, SPANISH_CORRIDOR_PATH, TIME_SLOT_SIZE
)
from robin.scraping.utils import timedelta_to_str
from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service, Supply
from robin.supply.utils import get_time

from copy import deepcopy
from functools import cached_property
from typing import Any, Dict, List, Mapping, Tuple


class DataLoader:
    """
    A DataLoader is a class that loads scraping data from the generated CSV files.

    Attributes:
        stops (pd.Dataframe): DataFrame containing the stops data.
        prices (pd.Dataframe): DataFrame containing the prices data.
        renfe_stations (pd.Dataframe): DataFrame containing the Renfe stations data.
        seat_components (Mapping[int, int]): Dictionary with seat components.
        seat_quantity (Mapping[int, int]): Dictionary with seat quantity.
        seat_names (List[str]): List of seat names.
        spanish_corridor (Dict[str, List[str]]): Dictionary with Spanish corridor data.
        time_slot_size (int): Time slot size in minutes.
    """

    def __init__(
        self,
        stops_path: str,
        prices_path: str,
        seat_components: Mapping[int, int],
        seat_quantity: Mapping[int, int],
        renfe_stations_path: str = RENFE_STATIONS_PATH,
        spanish_corridor_path: str = SPANISH_CORRIDOR_PATH,
        time_slot_size: int = TIME_SLOT_SIZE
    ) -> None:
        """
        Initialize a DataLoader with the given paths to the CSV files.

        Args:
            stops_path (str): Path to the stops CSV file.
            prices_path (str): Path to the prices CSV file.
            seat_components (Mapping[int, int]): Dictionary with seat components.
            seat_quantity (Mapping[int, int]): Dictionary with seat quantity.
            renfe_stations_path (str, optional): Path to the Renfe stations CSV file.
            spanish_corridor_path (str, optional): Path to the Spanish corridor YAML file.
            time_slot_size (int, optional): Time slot size in minutes.
        """
        self.stops = pd.read_csv(stops_path, dtype={'stop_id': str})
        self.prices = pd.read_csv(prices_path, dtype={'origin': str, 'destination': str})
        self.renfe_stations = pd.read_csv(renfe_stations_path, delimiter=';', dtype={'ADIF_ID': str, 'RENFE_ID': str})
        self.seat_components, self.seat_quantity = self._check_hard_types(seat_components, seat_quantity)
        self.seat_names = self.prices.columns[PRICES_COLUMNS:]
        self.spanish_corridor = self._read_yaml(path=spanish_corridor_path)
        self.time_slot_size = time_slot_size

    def _check_hard_types(
        self,
        seat_components: Mapping[int, int],
        seat_quantity: Mapping[int, int]
    ) -> Tuple[Mapping[int, int], Mapping[int, int]]:
        """
        Check if hard types in seat components are present in seat quantity.

        Returns:
            Tuple[Mapping[int, int], Mapping[int, int]]: Tuple of seat components and seat quantity.
        
        Raises:
            InvalidHardTypesException: If hard types in seat components are not present in seat quantity
        """
        hard_types = {hard_type for hard_type, _ in seat_components.values()}
        missing_types = hard_types - set(seat_quantity.keys())
        if missing_types:
            raise InvalidHardTypesException(list(missing_types), list(seat_quantity.keys()))
        return seat_components, seat_quantity

    def _get_line(self, stops: pd.DataFrame, corridor: Corridor) -> Line:
        """
        Get a line from stops data in a corridor.

        Args:
            stops (pd.DataFrame): DataFrame containing the stops data.
            corridor (Corridor): Corridor to which the line belongs.
        
        Returns:
            Line: Line created from the stops data.
        """
        line_data = {}
        for id_, arrival, departure in zip(stops['stop_id'], stops['arrival'], stops['departure']):
            line_data[id_] = (arrival, departure)
        line_id = stops['service_id'].values[0].split('_')[0]
        return Line(line_id, f'Line {line_id}', corridor, line_data)

    def _get_trip_prices(
        self,
        service_id: str,
        line: Line,
        start_time: datetime.timedelta
    ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Get trip prices for a given service ID, line and start time.

        Args:
            service_id (str): Service ID.
            line (Line): Line to which the trip belongs.
            start_time (datetime.timedelta): Start time of the trip.
        
        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: Dictionary with trip prices.    
        """
        total_prices = {}
        for pair in line.pairs:
            origin, destination = pair
            trip_id = service_id.split('_')[0]
            date = '-'.join(service_id.split('_')[1].split('-')[:-1])
            departure_time = start_time + datetime.timedelta(minutes=line.timetable[origin][1])
            departure_time = timedelta_to_str(departure_time)
            sub_service_id = f'{trip_id}_{date}-{departure_time}'
            match_service = self.prices['service_id'] == sub_service_id
            match_origin = self.prices['origin'] == origin
            match_destination = self.prices['destination'] == destination
            condition = match_service & match_origin & match_destination
            price_cols = [seat.name for seat in self.seats.values()]
            # If prices not found for a pair of stations, skip this pair and continue with the next one
            # It could happen if the pair is in the stops df but not in prices df
            try:
                prices = self.prices[condition][price_cols].values[0].tolist()
            except IndexError:
                continue
            total_prices[pair] = {seat: price for seat, price in zip(self.seats.values(), prices)}
        filtered_prices = {pair: {seat: price for seat, price in total_prices[pair].items() if not np.isnan(price)} for pair in total_prices}
        return filtered_prices

    def _read_yaml(self, path: str) -> Dict[str, List[str]]:
        """
        Read a YAML file and return its content.

        Args:
            path (str): Path to the YAML file.
        
        Returns:
            Dict[str, List[str]]: Dictionary with the content of the YAML file.
        """
        with open(path, 'r') as file:
            data = yaml.load(file, Loader=yaml.CSafeLoader)
        return data

    @cached_property
    def stations(self) -> Dict[str, Station]:
        """
        Returns a dictionary of stations with their IDs as keys.

        Returns:
            Dict[str, Station]: Dictionary of stations with their IDs as keys.
        """
        return Supply._get_stations(self.spanish_corridor, key='stations')

    @cached_property
    def time_slots(self) -> Dict[str, TimeSlot]:
        """
        Returns a dictionary of time slots with their IDs as keys.

        Returns:
            Dict[str, TimeSlot]: Dictionary of time slots with their IDs as keys.
        """
        time_slots_dict = {}
        for service_id in self.prices['service_id']:
            departure_str = service_id.split('-')[-1].replace('.', ':') + ':00'
            time_slot_start = get_time(departure_str)
            delta = datetime.timedelta(minutes=self.time_slot_size)
            time_slot_end = time_slot_start + delta
            time_slot_id = str(time_slot_start.seconds // 60) + str(delta.seconds // 60)
            time_slots_dict[time_slot_id] = TimeSlot(time_slot_id, time_slot_start, time_slot_end)
        return time_slots_dict

    @cached_property
    def corridors(self) -> Dict[str, Corridor]:
        """
        Returns a dictionary of corridors with their IDs as keys.

        Returns:
            Dict[str, Corridor]: Dictionary of corridors with their IDs as keys.
        """
        return Supply._get_corridors(self.spanish_corridor, self.stations, key='corridor')

    @cached_property
    def lines(self) -> Dict[str, Line]:
        """
        Returns a dictionary of lines with their IDs as keys.

        Returns:
            Dict[str, Line]: Dictionary of lines with their IDs as keys.
        """
        stops_by_service = self.stops.groupby('service_id')
        default_corridor = list(self.corridors.values())[0]
        routes_lines: pd.DataFrame = stops_by_service.apply(
            lambda service_stops: self._get_line(service_stops, default_corridor)
        )
        lines_dict = {}
        lines: List[Line] = list(set(routes_lines.values.tolist()))
        for line in lines:
            lines_dict[line.id] = line
        return lines_dict

    @cached_property
    def rolling_stocks(self) -> Dict[str, RollingStock]:
        """
        Returns a dictionary of rolling stocks with their IDs as keys.

        Returns:
            Dict[str, RollingStock]: Dictionary of rolling stocks with their IDs as keys.
        """
        return {'1': RollingStock('1', 'S-114', self.seat_quantity)}

    @cached_property
    def seats(self) -> Dict[str, Seat]:
        """
        Returns a dictionary of seats with their IDs as keys.

        Returns:
            Dict[str, Seat]: Dictionary of seats with their IDs as keys.
        """
        seats_dict = {}
        for i, seat_name in enumerate(self.seat_names, start=1):
            hard_type, soft_type = self.seat_components[seat_name]
            seats_dict[str(i)] = Seat(str(i), seat_name, hard_type, soft_type)
        return seats_dict

    @cached_property
    def tsps(self) -> Dict[str, TSP]:
        """
        Returns a dictionary of train service providers with their IDs as keys.

        Returns:
            Dict[str, TSP]: Dictionary of train service providers with their IDs as keys.
        """
        unique_tsps = self.prices['tsp'].unique()
        tsps_dict = {}
        for i, tsp_name in enumerate(unique_tsps, start=1):
            tsp_id = str(i)
            tsps_dict[tsp_id] = TSP(tsp_id, tsp_name, [rolling_stock for rolling_stock in self.rolling_stocks.values()])
        return tsps_dict

    @cached_property
    def services(self) -> List[Service]:
        """
        Returns a list of services.

        Returns:
            List[Service]: List of services.
        """
        trips = deepcopy(self.prices)
        trips['date'] = trips['service_id'].apply(lambda service_id: datetime.datetime.strptime(service_id.split('_')[1], '%d-%m-%Y-%H.%M').date())
        trips['line'] = trips['service_id'].apply(lambda service_id: self.lines[service_id.split('_')[0]])
        trips['train_service_provider'] = trips['tsp'].apply(
            lambda tsp_name: next(tsp for tsp in self.tsps.values() if tsp.name == tsp_name)
        )
        trips['time_slot'] = trips['service_id'].apply(
            lambda service_id: self.time_slots[
                f'{get_time(service_id.split("-")[-1].replace(".", ":") + ":00").seconds // 60}'
                f'{datetime.timedelta(minutes=self.time_slot_size).seconds // 60}'
            ]
        )
        trips['rolling_stock'] = trips['service_id'].apply(lambda _: self.rolling_stocks['1'])
        trips['prices'] = trips['service_id'].apply(
            lambda service_id: self._get_trip_prices(
                service_id=service_id, line=trips['line'].values[0], start_time=get_time(service_id.split('-')[-1].replace('.', ':') + ':00')
            )
        )
        trips['service'] = trips.apply(
            lambda service: Service(
            service['service_id'],
            service['date'],
            service['line'],
            service['train_service_provider'],
            service['time_slot'],
            service['rolling_stock'],
            service['prices']
            ),
            axis=1
        )
        return trips['service'].values.tolist()


class SupplySaver(Supply):
    """
    A SupplySaver is a class that saves supply entities to a YAML file.

    Attributes:
        services (List[Service]): List of services.
        stations (List[Station]): List of stations.
    """

    def __init__(self, services: List[Service], stations: List[Station]) -> None:
        """
        Initialize a SupplySaver with the given services and stations.

        The stations list is needed to create the corridor as not all stations may be present in the services.

        Args:
            services (List[Service]): List of services.
        """
        Supply.__init__(self, services)
        self.stations = stations

    def station_to_dict(self, station: Station) -> Dict[str, Any]:
        """
        Convert a Station object to a dictionary.

        Args:
            station (Station): Station object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Station object.
        """
        station_dict = {
            'id': station.id, 'name': station.name, 'city': station.city, 'short_name': station.short_name,
            'coordinates': {'latitude': float(station.coordinates[0]), 'longitude': float(station.coordinates[1])}
        }
        return station_dict

    def time_slot_to_dict(self, time_slot: TimeSlot) -> Dict[str, Any]:
        """
        Convert a TimeSlot object to a dictionary.

        Args:
            time_slot (TimeSlot): TimeSlot object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the TimeSlot object.
        """
        time_slot_dict = {'id': time_slot.id, 'start': str(time_slot.start), 'end': str(time_slot.end)}
        return time_slot_dict

    def corridor_to_dict(self, corridor: Corridor) -> Dict[str, Any]:
        """
        Convert a Corridor object to a dictionary.

        Args:
            corridor (Corridor): Corridor object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Corridor object.
        """
        def yaml_tree(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
            """
            Convert a tree structure to a YAML-compatible format.

            Args:
                tree (Dict[str, Any]): Tree structure to convert.
            
            Returns:
                List[Dict[str, Any]]: YAML-compatible format of the tree structure.
            """
            if not tree:
                return []
            else:
                node = [{'org': k.id, 'des': yaml_tree(v)} for k, v in tree.items()]
                return node

        tree_ids = yaml_tree(deepcopy(corridor.tree))
        corridor_dict = {'id': corridor.id, 'name': corridor.name, 'stations': tree_ids}
        return corridor_dict

    def line_to_dict(self, line: Line) -> Dict[str, Any]:
        """
        Convert a Line object to a dictionary.

        Args:
            line (Line): Line object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Line object.
        """
        stops = []
        for station in line.timetable:
            arrival, departure = line.timetable[station]
            stops.append({'station': station, 'arrival_time': arrival, 'departure_time': departure})
        line_dict = {'id': line.id, 'name': line.name, 'corridor': line.corridor.id, 'stops': stops}
        return line_dict

    def seat_to_dict(self, seat: Seat) -> Dict[str, Any]:
        """
        Convert a Seat object to a dictionary.

        Args:
            seat (Seat): Seat object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Seat object.
        """
        seat_dict = {'id': seat.id, 'name': seat.name, 'hard_type': seat.hard_type, 'soft_type': seat.soft_type}
        return seat_dict

    def rolling_stock_to_dict(self, rolling_stock: RollingStock) -> Dict[str, Any]:
        """
        Convert a RollingStock object to a dictionary.

        Args:
            rolling_stock (RollingStock): RollingStock object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the RollingStock object.
        """
        rolling_stock_dict = {
            'id': rolling_stock.id, 'name': rolling_stock.name,
            'seats': [{'hard_type': seat, 'quantity': rolling_stock.seats[seat]} for seat in rolling_stock.seats]
        }
        return rolling_stock_dict

    def tsp_to_dict(self, tsp: TSP) -> Dict[str, Any]:
        """
        Convert a TSP object to a dictionary.

        Args:
            tsp (TSP): TSP object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the TSP object.
        """
        tsp_dict = {
            'id': tsp.id, 'name': tsp.name, 'rolling_stock': [rolling_stock.id for rolling_stock in tsp.rolling_stock]
        }
        return tsp_dict

    def service_to_dict(self, service: Service) -> Dict[str, Any]:
        """
        Convert a Service object to a dictionary.

        Args:
            service (Service): Service object to convert.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the Service object.
        """
        prices = []
        for pair, seats in service.prices.items():
            prices.append({
                'origin': pair[0], 'destination': pair[1],
                'seats': [{'seat': seat.id, 'price': price} for seat, price in seats.items()]
            })
        service_dict = {
            'id': service.id, 'date': str(service.date), 'line': service.line.id,
            'train_service_provider': service.tsp.id, 'time_slot': service.time_slot.id,
            'rolling_stock': service.rolling_stock.id, 'origin_destination_tuples': prices,
            'capacity_constraints': service.capacity_constraints
        }
        return service_dict

    def to_yaml(self, output_path: str = OUTPUT_SUPPLY_PATH) -> None:
        """
        Save the supply entities to a YAML file.

        Args:
            output_path (str, optional): Path to the output YAML file. Defaults to 'supply_data.yaml'.
        """
        data = {
            'stations': [self.station_to_dict(station) for station in self.stations],
            'timeSlot': [self.time_slot_to_dict(tsp) for tsp in self.time_slots],
            'corridor': [self.corridor_to_dict(corridor) for corridor in self.corridors],
            'line': [self.line_to_dict(line) for line in self.lines],
            'seat': [self.seat_to_dict(seat) for seat in self.seats],
            'rollingStock': [self.rolling_stock_to_dict(rolling_stock) for rolling_stock in self.rolling_stocks],
            'trainServiceProvider': [self.tsp_to_dict(tsp) for tsp in self.tsps],
            'service': [self.service_to_dict(s) for s in self.services]
        }
        with open(output_path, 'w') as file:
            yaml.dump(data, file, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True)
