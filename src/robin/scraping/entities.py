"""Entities to be used in the data loader module."""

import datetime
import numpy as np
import os
import pandas as pd

from src.robin.supply.entities import Station, TimeSlot, Corridor, Line, Seat, RollingStock, TSP, Service
from src.robin.supply.utils import get_time
from src.scraping.utils import *
from typing import Dict, List

RENFE_STATIONS_PATH = "../../data/renfe/renfe_stations.csv"


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
        self.prices = self._load_dataframe(path=self._prices_path)
        self.stops = self._load_dataframe(path=self._stops_path, data_type={'stop_id': str})
        self.renfe_stations = pd.read_csv(filepath_or_buffer=renfe_stations_path,
                                          delimiter=',',
                                          dtype={'stop_id': str})

        self._seat_names = self.prices.columns[1:]
        self.seats = self._build_seat_types()
        self.stations = {}
        self.corridors = {}
        self.lines = {}
        self.rolling_stock = {}
        self.tsps = {}
        self.time_slots = {}
        self.services = {}

    def build_supply_entities(self) -> None:
        """
        Build supply entities from scraping data
        """
        self._build_corridor()
        self._build_lines()
        self._build_rolling_stock()
        self._build_tsp()
        self._build_services()

    def save_yaml(self, filename: str, path: str = "../../data/") -> None:
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
            write_to_yaml(path + filename, {key: value})

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
        grouped_df = self.stops.groupby('service_id')
        routes_lines = grouped_df.apply(lambda x: self._get_line(x, list(self.corridors.values())[0]))

        for line in list(set(routes_lines.values.tolist())):
            self.lines[line.id] = line

        self.trips['lines'] = self.trips['service_id'].apply(lambda x: self._get_trip_line(x))

    def _build_seat_types(self) -> Dict[str, Seat]:
        """
        Build seat types from prices dataframe

        Returns:
            seats: tuple of Seat() objects
        """
        hard_type, soft_type = 1, 1
        seats = {}
        for i, sn in enumerate(self._seat_names, start=1):
            seats[str(i)] = Seat(str(i), sn, hard_type, soft_type)
            if i % 2 == 0:
                soft_type += 1
            else:
                hard_type += 1

        return seats

    def _build_station_objects(self, corridor_stations: List[str]) -> None:
        """
        Build Station() objects from corridor stations list retrieved from stops dataframe

        Args:
            corridor_stations (List[str]): list of strings with the station ids
        """
        # Parse stations. Use Adif stop_id retrieve station info (name, lat, lon)
        # Build dictionary of stations with stop_id as key and Station() object as value
        for s in corridor_stations:
            name = self.renfe_stations[self.renfe_stations['stop_id'] == s]['stop_name'].values[0]
            city = self.renfe_stations[self.renfe_stations['stop_id'] == s]['stop_name'].values[0]
            shortname = str(self.renfe_stations[self.renfe_stations['stop_id'] == s]['stop_name'].values[0])[:3].upper()
            coords = tuple(self.renfe_stations[self.renfe_stations['stop_id'] == s][['stop_lat', 'stop_lon']].values[0])
            self.stations[s] = Station(s, name, city, shortname, coords)

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

    def _get_trip_price(self, service_id: str) -> None:
        """
        Get trip price from prices dataframe

        Args:
            service_id: string
            seats: tuple of Seat() objects

        Returns:
            price: tuple of floats (three types of seats for Renfe AVE)
        """
        try:
            prices = self.prices[self.prices['service_id'] == service_id][self._seat_names].values[0]
        except IndexError:
            prices = tuple([float("NaN") for _ in range(3)])

        return {s: p for s, p in zip(self.seats.values(), prices)}

    def _build_corridor(self) -> None:
        """
        Get corridor from stops dataframe

        Returns:
            corridor: list of Station() objects
        """

        corridor_stations = self._get_corridor_stations()
        self._build_station_objects(corridor_stations)

        first_station, last_station = tuple(self.stations.values())[::len(self.stations) - 1]
        corr_name = first_station.shortname + "-" + last_station.shortname

        def corridor_tree(sta):
            if len(sta) == 1:
                return {sta[0]: {}}
            return {sta[0]: corridor_tree(sta[1:])}

        corr_tree = corridor_tree(list(self.stations.values()))
        self.corridors[1] = Corridor("1", corr_name, corr_tree)

    def _get_line(self, stops: pd.DataFrame, corr: Corridor) -> Line:
        """
        Get line from stops dataframe
        Args:
            stops: dataframe with stops
            corr: Corridor() object
        Returns:
             Line() object
        """
        line_data = {s: (a, d) for s, a, d in zip(stops['stop_id'], stops['arrival'], stops['departure'])}
        idx = stops['service_id'].values[0].split("_")[0]
        return Line(idx, f"Line {idx}", corr, line_data)

    def _build_rolling_stock(self) -> None:
        """
        Build RollingStock objects
        """
        self.rolling_stock[1] = RollingStock("1", "S-114", {1: 250, 2: 50})

    def _build_tsp(self) -> None:
        """
        Build TSP objects
        """
        self.tsps[1] = TSP("1", "Renfe", [rs for rs in self.rolling_stock.values()])

    def _build_services(self) -> None:
        """
        Build Service objects
        """
        for i, row in self.trips.iterrows():
            service = self._get_service(
                row['service_id'],
                row['departure'],
                row['lines'],
                tuple(self.tsps.values())[0],
                tuple(self.rolling_stock.values())[0]
            )

            print(service)
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
        """

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
        ts_init = get_time(departure)
        ts_end = get_time(str(get_time(departure) + datetime.timedelta(minutes=10)))
        delta = datetime.timedelta(minutes=10)
        ts_id = str(get_time(departure).seconds // 60) + str(delta.seconds // 60)
        time_slot = TimeSlot(ts_id, ts_init, ts_end)
        self.time_slots[time_slot.id] = time_slot

        def time_delta_to_time_string(td: datetime.timedelta) -> str:
            """
            Convert time delta to string HH.MM

            Args:
                td: datetime.timedelta object

            Returns:
                string with time delta in format HH.MM
            """
            hours = td.total_seconds() // 3600
            minutes = (td.total_seconds() % 3600) / 60
            return f"{int(hours):02d}.{int(minutes):02d}"

        total_prices = {}
        for pair in line.pairs:
            origin, destination = pair
            trip_id = service_id.split("_")[0]
            date = "-".join(service_id.split("_")[1].split("-")[:-1])
            departure_time = ts_init + datetime.timedelta(minutes=line.timetable[origin][1])
            departure_time = time_delta_to_time_string(departure_time)
            sub_service_id = f"{trip_id}_{date}-{departure_time}"
            print(sub_service_id)
            prices = self.prices[(self.prices['service_id'] == sub_service_id) & (self.prices['origin'] == origin) & (self.prices['destination'] == destination)]
            print(prices)
            # prices = [float("NaN") for _ in range(len(self.seats))]
            total_prices[pair] = {st: p for st, p in zip(self.seats.values(), prices)}
        service = Service(id_=id_,
                          date=date,
                          line=line,
                          tsp=tsp,
                          time_slot=time_slot,
                          rolling_stock=rs,
                          prices=total_prices)
        #print(service)
        return service

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


if __name__ == '__main__':
    trips_path = '../../data/renfe/trips/trips_MADRI_BARCE_2023-04-11_2023-04-12.csv'

    data_loader = DataLoader(trips_path)
    data_loader.show_metadata()

    data_loader.build_supply_entities()
    print(list(data_loader.services.values())[0])

    data_loader.save_yaml(filename="supply_data_ref.yml")
