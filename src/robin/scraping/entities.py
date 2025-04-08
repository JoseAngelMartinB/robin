"""Entities for the scraping module."""

import datetime
import numpy as np
import pandas as pd
import yaml

from robin.scraping.exceptions import InvalidHardTypesException
from robin.scraping.constants import (
    OUTPUT_SUPPLY_PATH, PRICES_COLUMNS, RENFE_STATIONS_PATH, TIME_SLOT_SIZE, SPANISH_CORRIDOR_PATH
)
from robin.scraping.utils import (
    station_to_dict, time_slot_to_dict, corridor_to_dict, line_to_dict, seat_to_dict,
    rolling_stock_to_dict, tsp_to_dict, service_to_dict, timedelta_to_str
)
from robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service, Supply
from robin.supply.utils import get_time

from functools import cached_property
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


class DataLoader:
    """
    A DataLoader is a class that loads scraping data from the generated CSV files.

    Attributes:
        stops (pd.Dataframe): DataFrame containing the stops data.
        prices (pd.Dataframe): DataFrame containing the prices data.
        renfe_stations (pd.Dataframe): DataFrame containing the Renfe stations data.
        trips (pd.Dataframe): DataFrame containing the trips data.
        seat_components (Mapping[int, int]): Dictionary with seat components.
        seat_quantity (Mapping[int, int]): Dictionary with seat quantity.
        seat_names (List[str]): List of seat names.
        spanish_corridor (Dict[str, List[str]]): Dictionary with Spanish corridor data.
        ts_size (int): Time slot size in minutes.
    """

    def __init__(
        self,
        stops_path: Path,
        prices_path: Path,
        seat_components: Mapping[int, int],
        seat_quantity: Mapping[int, int],
        renfe_stations_path: str = RENFE_STATIONS_PATH,
        spanish_corridor_path: str = SPANISH_CORRIDOR_PATH,
        ts_size: int = TIME_SLOT_SIZE
    ) -> None:
        """
        Initialize a DataLoader with the given paths to the CSV files.

        Args:
            stops_path (Path): Path to the stops CSV file.
            prices_path (Path): Path to the prices CSV file.
            seat_components (Mapping[int, int]): Dictionary with seat components.
            seat_quantity (Mapping[int, int]): Dictionary with seat quantity.
            renfe_stations_path (str, optional): Path to the Renfe stations CSV file.
            spanish_corridor_path (str, optional): Path to the Spanish corridor YAML file.
            ts_size (int, optional): Time slot size in minutes.
        """
        self.stops = pd.read_csv(stops_path, dtype={'stop_id': str})
        self.prices = pd.read_csv(prices_path, dtype={'origin': str, 'destination': str})
        self.renfe_stations = pd.read_csv(renfe_stations_path, delimiter=';', dtype={'ADIF_ID': str, 'RENFE_ID': str})
        self.trips = pd.DataFrame({'service_id': list(dict.fromkeys(self.stops['service_id']))})
        self.seat_components, self.seat_quantity = self._check_hard_types(seat_components, seat_quantity)
        self.seat_names = self.prices.columns[PRICES_COLUMNS:]
        self.spanish_corridor = self._read_yaml(path=spanish_corridor_path)
        self.ts_size = ts_size

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
            total_prices[pair] = {st: p for st, p in zip(self.seats.values(), prices)}
        filtered_prices = {pair: {st: p for st, p in total_prices[pair].items() if not np.isnan(p)} for pair in total_prices}
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
        Returns a dictionary of time slots with the service IDs as keys.

        Returns:
            Dict[str, TimeSlot]: Dictionary of time slots with the service IDs as keys.
        """
        time_slots_dict = {}
        for service_id in self.trips['service_id']:
            departure_str = service_id.split('-')[-1].replace('.', ':') + ':00'
            ts_start = get_time(departure_str)
            delta = datetime.timedelta(minutes=self.ts_size)
            ts_end = ts_start + delta
            ts_id = str(ts_start.seconds // 60) + str(delta.seconds // 60)
            # NOTE: Use the service_id as the key for the time slot as then it is possible to index by it
            time_slots_dict[service_id] = TimeSlot(ts_id, ts_start, ts_end)
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
        return {1: RollingStock('1', 'S-114', self.seat_quantity)}

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
        return {1: TSP('1', 'Renfe', [rs for rs in self.rolling_stocks.values()])}

    @cached_property
    def services(self) -> List[Service]:
        """
        Returns a list of services.

        Returns:
            List[Service]: List of services.
        """
        self.trips['date'] = self.trips['service_id'].apply(lambda service_id: datetime.datetime.strptime(service_id.split('_')[1], '%d-%m-%Y-%H.%M').date())
        self.trips['line'] = self.trips['service_id'].apply(lambda service_id: self.lines[service_id.split('_')[0]])
        self.trips['train_service_provider'] = self.trips['service_id'].apply(lambda _: self.tsps[1])
        self.trips['time_slot'] = self.trips['service_id'].apply(lambda service_id: self.time_slots[service_id])
        self.trips['rolling_stock'] = self.trips['service_id'].apply(lambda _: self.rolling_stocks[1])
        self.trips['prices'] = self.trips['service_id'].apply(
            lambda service_id: self._get_trip_prices(
                service_id=service_id, line=self.trips['line'].values[0], start_time=get_time(service_id.split('-')[-1].replace('.', ':') + ':00')
            )
        )
        self.trips['service'] = self.trips.apply(
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
        return self.trips['service'].values.tolist()


class SupplySaver(Supply):
    """
    A SupplySaver is a class that saves supply entities to a YAML file.

    Attributes:
        services (List[Service]): List of services.
    """

    def __init__(self, services: List[Service]) -> None:
        """
        Initialize a SupplySaver with the given services.

        Args:
            services (List[Service]): List of services.
        """
        Supply.__init__(self, services)

    def to_yaml(self, output_path: Path = OUTPUT_SUPPLY_PATH) -> None:
        """
        Save the supply entities to a YAML file.

        Args:
            output_path (Path, optional): Path to the output YAML file. Defaults to 'supply_data.yaml'.
        """
        data = {
            'stations': [station_to_dict(stn) for stn in self.stations],
            'timeSlot': [time_slot_to_dict(s) for s in self.time_slots],
            'corridor': [corridor_to_dict(corr) for corr in self.corridors],
            'line': [line_to_dict(ln) for ln in self.lines],
            'seat': [seat_to_dict(s) for s in self.seats],
            'rollingStock': [rolling_stock_to_dict(rs) for rs in self.rolling_stocks],
            'trainServiceProvider': [tsp_to_dict(tsp) for tsp in self.tsps],
            'service': [service_to_dict(s) for s in self.services]
        }
        with open(output_path, 'w') as file:
            yaml.dump(data, file, Dumper=yaml.CSafeDumper, sort_keys=False, allow_unicode=True)
