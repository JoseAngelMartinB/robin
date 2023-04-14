"""Entities to be used in the data loader module."""

import datetime
import numpy as np
import os
import pandas as pd

from src.robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from src.robin.supply.utils import get_time
from src.robin.scraping.utils import *
from typing import Dict, List, Tuple

RENFE_STATIONS_PATH = f'data/renfe/renfe_stations.csv'
DEFAULT_SEAT_QUANTITY = {1: 250, 2: 50}


class DataLoader:
    """
    Class to load data retrieved with the scraping from csv files

    Attributes:
        trips (pd.DataFrame): Trips dataframe
        prices (pd.DataFrame): Prices dataframe
        stops (pd.DataFrame): Stops dataframe
        renfe_stations (pd.DataFrame): Renfe stations dataframe
        seats (Dict[str, Seat]): Dictionary with seat types
        stations (Dict[str, Station]): Dictionary with stations
        corridors (Dict[str, Corridor]): Dictionary with corridors
        lines (Dict[str, Line]): Dictionary with lines
        rolling_stock (Dict[str, RollingStock]): Dictionary with rolling stock
        tsps (Dict[str, TSP]): Dictionary with TSPs
        time_slots (Dict[str, TimeSlot]): Dictionary with time slots
        services (Dict[str, Service]): Dictionary with services
    """

    def __init__(self, trips_path: str, renfe_stations_path: str = RENFE_STATIONS_PATH):
        """
        Constructor of the class

        Args:
            trips_path (str): Path to the trips csv file
            renfe_stations_path (str, optional): Path to the renfe stations csv file.
        """
        self._trips_path = trips_path
        self._path_root = os.path.dirname(os.path.dirname(self._trips_path))
        self._scraping_id = self._get_scraping_id()
        self._prices_path = f"{self._path_root}/prices/prices_{self._scraping_id}.csv"
        self._stops_path = f"{self._path_root}/stop_times/stopTimes_{self._scraping_id}.csv"
        self.origin_id, self.destination_id, self.start_date, self.end_date = self._scraping_id.split('_')

        self.trips = self._load_dataframe(path=self._trips_path, data_type={'trip_id': str})
        self.prices = self._load_dataframe(path=self._prices_path, data_type={'origin': str, 'destination': str})
        self.stops = self._load_dataframe(path=self._stops_path, data_type={'stop_id': str})
        self.renfe_stations = pd.read_csv(filepath_or_buffer=renfe_stations_path,
                                          delimiter=',',
                                          dtype={'stop_id': str})

        self._seat_names = self.prices.columns[-3:]
        self.seats = {}
        self.stations = {}
        self.corridors = {}
        self.lines = {}
        self.rolling_stock = {}
        self.tsps = {}
        self.time_slots = {}
        self.services = {}

    def build_supply_entities(self, seat_quantity: Mapping[int, int] = DEFAULT_SEAT_QUANTITY) -> None:
        """
        Build supply entities from scraping data
        """
        self._build_seat_types()
        self._build_corridor()
        self._build_lines()
        self._build_rolling_stock(seat_quantity=seat_quantity)
        self._build_tsp()
        self._build_services()

    def save_yaml(self, save_path: str) -> None:
        """
        Save supply entities to yaml file

        Args:
            filename (str): File name for the yaml file
            path (str): Path to save yaml file
        """
        data = [
            ('stations', [station_to_dict(stn) for stn in self.stations.values()]),
            ('seat', [seat_to_dict(s) for s in self.seats.values()]),
            ('corridor', [corridor_to_dict(corr) for corr in self.corridors.values()]),
            ('line', [line_to_dict(ln) for ln in self.lines.values()]),
            ('rollingStock', [rolling_stock_to_dict(rs) for rs in self.rolling_stock.values()]),
            ('trainServiceProvider', [tsp_to_dict(tsp) for tsp in self.tsps.values()]),
            ('timeSlot', [time_slot_to_dict(s) for s in self.time_slots.values()]),
            ('service', [service_to_dict(s) for s in self.services.values()])
        ]

        for key, value in data:
            write_to_yaml(save_path, {key: value})

    def show_metadata(self) -> None:
        """
        Print metadata of the retrieved scraping files
        """
        print(f"Origin:{self.origin_id} - Destination:{self.destination_id}")
        print(f"Since: {self.start_date}, Until: {self.end_date}")

    def _get_scraping_id(self) -> str:
        """
        Get scraping id from trips path specified by user

        Returns:
            string with scraping id
        """
        # E.g file_name = 'trips_MADRI_BARCE_2022-12-30_2023-01-03'
        file_name = self._trips_path.split('/')[-1].split(".")[0]

        # E.g. 'MADRI_BARCE_2022-12-30_2023-01-03'
        return "_".join(file_name.split("_")[1:])

    def _build_lines(self) -> None:
        """
        Build Line objects from stops dataframe
        """
        grouped_df = self.stops.groupby('service_id')  # Get all stops for each service_id
        routes_lines = grouped_df.apply(lambda row: self._get_line(row, list(self.corridors.values())[0]))

        for line in list(set(routes_lines.values.tolist())):
            self.lines[line.id] = line

        self.trips['lines'] = self.trips['service_id'].apply(lambda x: self._get_trip_line(x))

    def _build_seat_types(self) -> Dict[str, Seat]:
        """
        Build seat types from prices dataframe

        Returns:
            seats: tuple of Seat() objects
        """
        hard_type, soft_type = 1, 1  # Initialize seat types
        for i, seat_name in enumerate(self._seat_names, start=1):
            self.seats[str(i)] = Seat(str(i), seat_name, hard_type, soft_type)
            if i % 2 == 0:
                soft_type += 1
            else:
                hard_type += 1

    def _build_station_objects(self, corridor_stations: List[str]) -> None:
        """
        Build Station() objects from corridor stations list of station ids

        Args:
            corridor_stations (List[str]): list of strings with the station ids
        """
        for station in corridor_stations:
            # Retrieve station info from dataframe using the station id
            station_row = self.renfe_stations[self.renfe_stations['stop_id'] == station]
            name = station_row['stop_name'].values[0]
            city = station_row['stop_name'].values[0].replace('-', ' ').split(' ')[0]
            shortname = str(station_row['stop_name'].values[0])[:3].upper()
            coords = tuple(station_row[['stop_lat', 'stop_lon']].values[0])
            self.stations[station] = Station(station, name, city, shortname, coords)

    def _get_corridor_stations(self) -> List[str]:
        """
        Get list of stations that are part of the corridor

        Returns:
            corridor_stations (List[str]): list of strings with the station ids
        """
        grouped_df = self.stops.groupby('service_id')

        # Get nested list with stops for each trip
        list_stations = grouped_df.apply(lambda d: list(d['stop_id'])).values.tolist()

        # Initialize corridor with max length trip
        corridor_stations = list_stations.pop(list_stations.index(max(list_stations, key=len)))

        # Complete corridor with other stops that are not in the initial defined corridor
        for trip in list_stations:
            for i, s in enumerate(trip):
                if s not in corridor_stations:
                    corridor_stations.insert(corridor_stations.index(trip[i + 1]), s)

        return corridor_stations

    def _get_trip_line(self, service_id: str) -> Line:
        """
        Get trip line from set_lines dictionary

        Args:
            service_id: string.
            lines: dictionary with lines.

        Returns:
            line (Line): Line object for the specified service_id
        """
        try:
            line = self.lines[service_id.split("_")[0]]
        except KeyError:
            raise KeyError(f"Line not found for service_id: {service_id}")

        return line

    def _build_corridor(self) -> None:
        """
        Get corridor from stops dataframe

        Returns:
            corridor: list of Station() objects
        """
        # Get list of stations in corridor
        corridor_stations = self._get_corridor_stations()
        self._build_station_objects(corridor_stations)  # Station objects get stored in self.stations dictionary

        # Build corridor name using first and last station names
        first_station, last_station = tuple(self.stations.values())[::len(self.stations) - 1]
        corridor_name = first_station.shortname + "-" + last_station.shortname

        def corridor_tree(station: List[Station]) -> Dict[Station, Dict[Station, Dict]]:
            """
            Build corridor tree from list of stations

            Args:
                station: list of Station() objects

            Returns:
                corridor_tree: dictionary with Station() objects as keys and dictionaries as values
            """
            if len(station) == 1:
                return {station[0]: {}}
            return {station[0]: corridor_tree(station[1:])}

        corridor_tree = corridor_tree(list(self.stations.values()))
        self.corridors[1] = Corridor("1", corridor_name, corridor_tree)

    def _get_line(self, stops: pd.DataFrame, corridor: Corridor) -> Line:
        """
        Get line from stops dataframe
        Args:
            stops: dataframe with stops
            corr: Corridor() object
        Returns:
             Line() object
        """
        line_data = {}
        for id_, arrival, departure in zip(stops['stop_id'], stops['arrival'], stops['departure']):
            line_data[id_] = (arrival, departure)
        line_id = stops['service_id'].values[0].split("_")[0]
        return Line(line_id, f"Line {line_id}", corridor, line_data)

    def _build_rolling_stock(self, seat_quantity: Mapping[int, int]) -> None:
        """
        Build RollingStock objects
        """
        self.rolling_stock[1] = RollingStock("1", "S-114", seat_quantity)

    def _build_tsp(self) -> None:
        """
        Build TSP objects
        """
        self.tsps[1] = TSP("1", "Renfe", [rs for rs in self.rolling_stock.values()])

    def _build_services(self) -> None:
        """
        Build Service objects
        """
        self.trips['service'] = self.trips.apply(lambda x: self._get_service(x['service_id'],
                                                                             x['departure'],
                                                                             x['lines'],
                                                                             tuple(self.tsps.values())[0],
                                                                             tuple(self.rolling_stock.values())[0]),
                                                 axis=1)

        my_services = self.trips['service'].values.tolist()
        for s in my_services:
            self.services[s.id] = s

    def _build_time_slot(self, start_time: datetime.timedelta, ts_size: int = 10) -> TimeSlot:
        """
        Build TimeSlot objects from start time

        Args:
            start_time: string with start time
            ts_size: int with time slot size in minutes
        """
        ts_end = start_time + datetime.timedelta(minutes=ts_size)
        delta = datetime.timedelta(minutes=10)
        ts_id = str(start_time.seconds // 60) + str(delta.seconds // 60)
        time_slot = TimeSlot(ts_id, start_time, ts_end)
        self.time_slots[time_slot.id] = time_slot

        return time_slot

    def _get_trip_prices(
            self,
            service_id: str,
            line: Line,
            start_time: datetime.timedelta
    ) -> Dict[Tuple[str, str], Dict[Seat, float]]:
        """
        Get trip prices from prices dataframe

        Args:
            service_id: string
            line: Line() object
            start_time: string with start time

        Returns:
            Dict[Tuple[str, str], Dict[Seat, float]]: dictionary with pairs of stations as keys and dictionaries with
            Seat() objects as keys and prices as values.
        """
        total_prices = {}
        for pair in line.pairs:
            origin, destination = pair
            trip_id = service_id.split("_")[0]
            date = "-".join(service_id.split("_")[1].split("-")[:-1])
            departure_time = start_time + datetime.timedelta(minutes=line.timetable[origin][1])
            departure_time = time_delta_to_time_string(departure_time)
            sub_service_id = f"{trip_id}_{date}-{departure_time}"
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

    def _get_service(self,
                     service_id: str,
                     departure: str,
                     line: Line,
                     tsp: TSP,
                     rs: RollingStock
                     ) -> Service:
        """
        Get Service object from Renfe data

        Args:
            service_id: string
            departure: string
            price: tuple of floats
            line: Line() object
            tsp: TSP() object
            rs: RollingStock() object

        Returns:
            Service() object
        """
        id_ = service_id
        date = departure.split(" ")[0]
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        departure = departure.split(" ")[1]
        start_time = get_time(departure)
        time_slot = self._build_time_slot(start_time=start_time)
        total_prices = self._get_trip_prices(line=line, service_id=service_id, start_time=start_time)

        return Service(id_=id_,
                       date=date,
                       line=line,
                       tsp=tsp,
                       time_slot=time_slot,
                       rolling_stock=rs,
                       prices=total_prices)

    def _load_dataframe(self, path: str, data_type: Dict = None) -> pd.DataFrame:
        """
        Load dataframe from csv file

        Args:
            path (str): String with path to csv file
            data_type (Dict): Dictionary with column names and data types

        Returns:
            pd.dataframe
        """
        return pd.read_csv(path, delimiter=',', dtype=data_type)
