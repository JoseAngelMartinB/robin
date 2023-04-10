"""Entities for Renfe scraping."""

import bs4
import datetime
import os
import pandas as pd
import re
import requests
import unicodedata

from src.scraping.renfe.utils import ChromeDriverManager, format_duration, is_number, remove_blanks, time_to_minutes

from bs4 import BeautifulSoup
from typing import List, Tuple, Dict

# Renfe URL's
MAIN_MENU_URL = "https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html"
SCHEDULE_URL = "https://horarios.renfe.com/HIRRenfeWeb/"

RENFE_STATIONS_CSV = "../../../data/renfe/renfe_stations.csv"
SAVE_PATH = "../../../data/renfe"


class RenfeScraper:
    """
    Renfe Scraping class.

    Attributes:
        stations_df (pd.DataFrame): A pandas DataFrame to parse stations data.
        available_stations (List[str]): A list with the available stations in the Renfe website.
        allowed_train_types (List[str]): A list with the allowed train types to scrape.
        chromeDriver (ChromeDriverManager): A ChromeDriverManager object to manage the ChromeDriver.
        line_stations (List[str]): A list with the stations of the line.
    """

    def __init__(
            self,
            stations_csv_path: str = RENFE_STATIONS_CSV,
            menu_url: str = MAIN_MENU_URL
    ) -> None:
        """
        Initialize a RenfeScraper object.

        Args:
            stations_csv_path (str): Path to the csv file with the available stations.
            menu_url (str): URL of the Renfe main menu.
        """
        self.stations_df = RenfeScraper._load_stations_df(stations_csv_path)
        self.available_stations = RenfeScraper._scrape_stations(menu_url)
        self.allowed_train_types = ["AVE", "AVLO"]
        self.chromeDriver = ChromeDriverManager()
        self.line_stations = []

    def scrape(
            self,
            origin: str,
            destination: str,
            init_date: datetime.date = None,
            range_days: int = 1,
            save_path: str = SAVE_PATH
    ) -> None:
        """
        Scrapes the Renfe website for the trips and prices of services between two stations, a given date and for a
        range of days

        Args:
            origin (str): Adif station id of the origin station.
            destination (str): Adif station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
        """
        origin_id = self._get_renfe_station_id(origin)
        destination_id = self._get_renfe_station_id(destination)

        # Assert that the origin and destination stations are in the list of stations operated by Renfe
        pair_of_stations_in_csv = all(s in self.available_stations.keys() for s in (origin_id, destination_id))
        assert pair_of_stations_in_csv, "Invalid origin or destination"

        if not init_date:
            init_date = datetime.date.today()

        date = init_date
        end_date = init_date + datetime.timedelta(days=range_days)
        trips_df = pd.DataFrame()
        print(f"Scraping prices from {init_date} to {end_date}...")
        for i in range(range_days):
            trips_df = pd.concat([trips_df,
                                  self.scrape_trips(origin_id=origin_id,
                                                    destination_id=destination_id,
                                                    date=date)
                                  ], ignore_index=True)
            date += datetime.timedelta(days=1)

        RenfeScraper._save_trips_df(df=trips_df,
                                    origin_id=origin_id,
                                    destination_id=destination_id,
                                    init_date=init_date,
                                    end_date=end_date,
                                    save_path=save_path)

        RenfeScraper._save_stops_df(df=trips_df,
                                    origin_id=origin_id,
                                    destination_id=destination_id,
                                    init_date=init_date,
                                    end_date=end_date,
                                    save_path=save_path)

        schedules = trips_df['schedule'].values.tolist()
        print("Schedules: ", schedules)
        # Initialize corridor with max length trip
        corridor_stations = list(schedules.pop(schedules.index(max(schedules, key=len))))

        # Complete corridor with other stops that are not in the initial defined corridor
        for trip in schedules:
            for i, s in enumerate(trip):
                if s not in corridor_stations:
                    if i == len(trip) - 1:
                        corridor_stations.append(s)
                    else:
                        index = corridor_stations.index(trip[i + 1])
                        corridor_stations[index:index] = [s]

        print("Corridor: ", corridor_stations)
        date = init_date
        end_date = init_date + datetime.timedelta(days=range_days)
        prices_df = pd.DataFrame()
        for org in corridor_stations:
            for des in corridor_stations[corridor_stations.index(org) + 1:]:
                for i in range(range_days):
                    prices_df = pd.concat([prices_df,
                                           self.scrape_prices(origin_id=org,
                                                              destination_id=des,
                                                              date=date)
                                           ], ignore_index=True)
                    date += datetime.timedelta(days=1)

        print(prices_df.head())
        os.makedirs(f'{save_path}/prices/', exist_ok=True)
        prices_df.to_csv(f"{save_path}/prices/prices_{origin_id}_{destination_id}_{init_date}_{end_date}.csv",
                         index=False)

    def scrape_prices(
            self,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> pd.DataFrame:
        """
        Scrapes prices from Renfe website using selenium and saves retrieved data to a csv file.

        Args:
            origin_id (str): Renfe id of the origin station
            destination_id (str): Renfe id of the destination station
            date (datetime.date): date of the trip
            range_days (int): number of days to search for trips
            save_path (str): path to save the csv file

        Returns:
            pd.DataFrame: DataFrame with the scraped data
        """
        records = []
        date_str = date.strftime("%d-%m-%Y")  # Format date to match Renfe website format
        root = "https://venta.renfe.com/vol/"
        query = f"buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional="

        url = root + query
        print("Date: ", date)
        print("Search url: ", url)
        while True:
            html_str = self.chromeDriver.request_price(url)
            if not html_str:
                user_input = input("Error retrieving prices. Retry? (y/n): ").lower()
                if user_input == "n":
                    break
            else:
                break

        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find("div", {"class": "tab-content"})
        header = soup.find("thead")
        seat_types = self._get_seat_types(header)
        content_rows = table.find_all("tr", attrs={"cdgotren": True})
        for row in content_rows:
            data = self._get_service_prices(row, date, seat_types)
            data[1:1] = [origin_id, destination_id]
            records.append(data)

        col_names = ['trip_id', 'origin', 'destination', 'train_type', 'departure', 'arrival', 'duration', 'prices']
        return self._get_prices_dataframe(records=records, col_names=col_names, seat_types=seat_types)

    def scrape_trips(
            self,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> pd.DataFrame:
        """
        Obtains two pandas dataframes from Renfe website, one with the trips information and another with the stops,
            which are saved to csv files.

        Args:
            origin_id (str): Origin station id
            destination_id (str): Destination station id
            date (datetime.date): Initial date to search for trips
            range_days (int): Number of days to search for trips from the initial date
            save_path (str): Path to save the csv files

        Returns:
            pd.DataFrame: DataFrame with the scraped trips data
        """
        rows = []
        url = self._get_renfe_schedules_url(origin_id, destination_id, date)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        main_table = soup.select_one('.irf-travellers-table__container-table')
        for tr in main_table.select('tr.odd.irf-travellers-table__tr'):
            row = tr.select('td.txt_borde1.irf-travellers-table__td')
            if not self._is_content_row(row):
                continue
            service_data = self._get_trip_data(row, date)
            if service_data[1] not in self.allowed_train_types:
                continue
            rows.append(service_data)
        date += datetime.timedelta(days=1)

        col_names = ['trip_id', 'train_type', 'schedule', 'departure', 'duration', 'price']
        return self._dataframe_from_records(records=rows, col_names=col_names)

    def _dataframe_from_records(self,
                                records: List,
                                col_names: List,
                                ) -> pd.DataFrame:
        """
        Returns a dataframe with the information retrieved from the scraping encoded in a list of lists.
        Each list in the list of lists represents the data of a service, and it becomes a row in the dataframe.

        Args:
            records (List): List of lists with the information retrieved from the scraping
            col_names (List): List of column names for the dataframe

        Returns:
            pd.DataFrame: dataframe with the information retrieved from the scraping
        """
        df = pd.DataFrame(records, columns=col_names)

        # Filter only AVE or AVLO trains
        train_type_filter = df["train_type"].str.contains("|".join(self.allowed_train_types))
        df = df[train_type_filter].reset_index(drop=True)
        df["service_id"] = df.apply(lambda x: x["trip_id"] + "_" + x["departure"].strftime("%d-%m-%Y-%H.%M"), axis=1)
        return df

    def _get_name_best_match(self, raw_text: str) -> str:
        """
        Returns the best match between the name of the station and the name of the station in the stations csv file

        Args:
            raw_text (str): Name of the station as it appears in the Renfe website

        Returns:
            str: Best match between the name of the station and the name of the station in the stations csv file
        """
        # Get Adif station names from df, and convert them to lowercase, without spaces or dashes
        adif_names = [re.sub(r'[-/]', ' ', name.lower()) for name in self.stations_df['stop_name'].tolist()]

        raw_name = re.sub(r'[^a-zA-Z0-9 -]', '', raw_text)  # Remove non-alphanumeric characters
        name = " ".join(filter(None, re.split(r'\W+', raw_name))).lower()

        best_match = max(adif_names,
                         key=lambda gn: sum(w in gn.split(" ") for w in name.split(" ")),
                         default="Unknown")

        if best_match != "Unknown":
            return self.stations_df.at[adif_names.index(best_match), 'stop_id']
        return "00000"

    def _get_prices_dataframe(self, records: List, col_names: List, seat_types: List) -> pd.DataFrame:
        """
        Returns a dataframe with the information retrieved from the scraping encoded in a list of lists.

        Args:
            records (List): List of lists with the information retrieved from the scraping

        Returns:
            pd.DataFrame: Dataframe with the information retrieved from the scraping
        """
        df = self._dataframe_from_records(records, col_names)
        prices = dict(zip(df.service_id, df.prices))
        list_prices = []
        for k, v in prices.items():
            list_prices.append([k, *list(v.values())])

        df_prices = pd.DataFrame(list_prices, columns=['service_id'] + seat_types)
        return df_prices

    def _get_renfe_schedules_url(self, origin_id: str, destination_id: str, date: datetime.date) -> str:
        """
        Returns the URL of the Renfe schedules website for a given origin-destination pair of stations and a given date

        Args:
            origin_id (str): Renfe id of the origin station
            destination_id (str): Renfe id of the destination station
            date (datetime.date): date of the trip

        Returns:
            str: URL of the Renfe schedules website for the given origin-destination pair of stations and a
        """
        year, month, day = date.strftime('%Y-%m-%d').split("-")
        weekday = date.weekday() + 1
        url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
        print("Date: ", date)
        print("Search url: ", url)
        return url

    def _get_renfe_station_id(self, adif_id: str) -> str:
        """
        Returns the Renfe station id given the ADIF station id.

        Args:
            adif_id (str): Adif station id.

        Returns:
            str: Renfe station id.
        """
        return self.stations_df[self.stations_df['stop_id'] == adif_id]['renfe_id'].values[0]

    def _get_service_prices(self,
                            row: bs4.element.Tag,
                            date: datetime.date,
                            seat_types: List
    ) -> List:
        """
        Get the prices of the different seat types for a given row (service in Renfe website).

        Args:
            row (bs4.element.Tag): row of the table with the services
            date (datetime.date): date of the service
            seat_types (List): list of the seat types to get the prices for

        Returns:
            List: A list with the different service prices for each seat type available in the service.
        """
        train_id = row['cdgotren']
        cols = row.find_all("td")
        first_col = cols[1].find_all("div")
        departure, duration = first_col[0].text, first_col[1].text
        departure_time = time_to_minutes(remove_blanks(s=departure, replace_by=''))
        departure = datetime.datetime.combine(date, datetime.time(hour=0, minute=0))
        departure = departure + datetime.timedelta(minutes=departure_time)

        duration = remove_blanks(s=duration, replace_by=' ')
        duration = format_duration(duration)
        duration = datetime.timedelta(minutes=duration)

        arrival = departure + duration
        train_type = self._map_train_type(trip_id=train_id)

        arrival = arrival + datetime.timedelta(days=1)
        train_type = remove_blanks(s=train_type, replace_by=' ')
        price_cols = cols[4:-1]
        prices = RenfeScraper._get_prices(cols=price_cols, seat_types=seat_types)
        train = [train_id, train_type, departure, arrival, duration, prices]
        assert all(train), f"Error parsing service prices: {train}"
        return train

    def _map_train_type(self, trip_id: str) -> str:
        """
        Returns the train type given the trip id.

        Args:
            trip_id (str): trip id.

        Returns:
            str: train type.
        """
        if trip_id[:2] == '00':
            return 'ALVIA'
        if trip_id[:2] in ('08', '34'):
            return 'AVANT'
        if trip_id[:2] in ('03', '19'):
            return 'AVE'
        if trip_id[:2] == '06':
            return 'AVLO'
        if trip_id[:2] == '17':
            return 'REG.EXP'
        return 'UNK'

    def _get_seat_types(self, thead: bs4.element.Tag) -> List[str]:
        """
        Get a list of seat types from the table header.

        Args:
            thead (bs4.element.Tag): table header of the table with the services

        Returns:
            List[str]: List of seat types
        """
        table_head_text = [th.text.strip() for th in thead.find('tr').find_all('th')]
        seat_types_raw = table_head_text[4:-1]

        # Remove accents and other special characters from seat type names
        seat_types = []
        for s in seat_types_raw:
            s_normalized = unicodedata.normalize('NFD', s)
            s_clean = ''.join(c for c in s_normalized if unicodedata.category(c) != 'Mn')
            seat_types.append(s_clean)

        return seat_types

    def _get_trip_data(self,
                       row: bs4.element.ResultSet,
                       date: datetime.date
                       ) -> Tuple[str, str, Dict[str, Tuple[int, int]], datetime.datetime, int, Dict[str, float]]:
        """
        Returns the data of a trip retrieved from a row from the schedules table.

        Args:
            row (bs4.element.Tag): row of the table with the trips information
            date (datetime.date): date of the trip

        Returns:
            Tuple[str, str, str, datetime.datetime, float, Dict[str, float]]: data of the trip from the schedules table
        """
        trip_id = tuple(filter(None, re.split(r"\s+", row[0].text.strip())))[0]
        train_type = self._map_train_type(trip_id=trip_id)

        schedule_link = row[0].find('a')['href']
        trip_url = RenfeScraper._get_trip_url(schedule_link=schedule_link, schedule_url=SCHEDULE_URL)
        if train_type not in ['AVE', 'AVLO']:
            trip_schedule = 1
        else:
            trip_schedule = self._scrape_trip_schedule(url=trip_url)

        html_prices = re.sub(r"\s+", "", row[4].find("div").text)
        raw_prices = re.sub(r'PrecioInternet|:', '', html_prices).replace(",", ".")

        prices = {}
        for fare, price in re.findall(r'([a-zA-Z]+)[^\d]+([\d.]+)', raw_prices):
            prices[fare] = float(price)

        departure = row[1].text.strip()
        departure = datetime.datetime.strptime(str(date) + "-" + departure, '%Y-%m-%d-%H.%M')
        duration = format_duration(row[3].text.strip())
        train = (trip_id, train_type, trip_schedule, departure, duration, prices)
        assert all(train), f"Error parsing service trips: {train}"  # Assert non empty values

        return train

    def _is_content_row(self, row: bs4.element.ResultSet) -> bool:
        """
        Checks if row is a valid row with trip information

        Args:
            row (bs4.element.ResultSet): row of the table with the trips information
        """
        if not row or len(row) < 6:
            return False
        return True

    def _scrape_trip_schedule(
            self,
            url: str,
    ) -> Dict[str, Tuple[int, int]]:
        """
        Returns dictionary of stops from URL with stops information from Renfe

        Args:
            schedule_link (str): URL with stops information from Renfe
            schedule_url (str): URL of Renfe website

        Returns:
            Dict[str, Tuple[int, int]]: Dictionary of stops, where keys are each station and values are a tuple with
            (arrival, departure) times.
        """
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        table = soup.select_one('.irf-renfe-travel__table.cabecera_tabla')

        trip_schedule = {}
        for row in table.select('tr'):
            aux = row.select('td')
            if not aux:
                continue

            raw_table_name = aux[0].text.strip()
            station_id = self._get_name_best_match(raw_table_name)
            departure_time = time_to_minutes(aux[1].text.strip())
            arrival_time = time_to_minutes(aux[2].text.strip())
            trip_schedule[station_id] = (departure_time, arrival_time)

        # Get first and last stations and set their arrival and departure times to the same value
        first_station, *_, last_station = trip_schedule.keys()
        trip_schedule[first_station] = (trip_schedule[first_station][1], trip_schedule[first_station][1])
        trip_schedule[last_station] = (trip_schedule[last_station][0], trip_schedule[last_station][0])

        relative_schedule = RenfeScraper._absolute_to_relative(trip_schedule)
        return relative_schedule

    @staticmethod
    def _absolute_to_relative(schedule: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Converts absolute times to relative times.

        Args:
            schedule (Dict[str, Tuple[int, int]]): Dictionary of stops, where keys are each station and values are
            a tuple of floats with (arrival, departure) times in minutes from midnight.

        Returns:
            Dict[str, Tuple[int, int]]: Same dictionary of stops, where keys are each station and values are a tuple
            with (arrival, departure) relative time in minutes
        """
        start_time = schedule[tuple(schedule.keys())[0]][0]

        relative_schedule = {}
        for key, (start, end) in schedule.items():
            relative_start = start - start_time
            relative_end = end - start_time
            relative_schedule[key] = (relative_start, relative_end)

        return relative_schedule

    @staticmethod
    def _get_prices(cols, seat_types: List) -> Dict[str, float]:
        """
        Get the prices of the different seat types for a given row (service in Renfe website).

        Args:
            cols (bs4.element.ResultSet): Columns of the table with the services
            seat_types (List): List of the seat types

        Returns:
            Dict[str, float]: Dictionary with the prices of the different seat types
        """
        if len(cols) == 1:  # Means train is full, so no prices are available
            return {st: float("NaN") for st in seat_types}

        prices = {}
        # If any of the prices is available, number of price cols will be equal to number of seat types
        for seat, col in zip(seat_types, cols):
            button_col = col.find("button")  # Button columns contains the prices info
            price = float("NaN")
            price_div = button_col.find_all("div")
            if price_div:
                price = price_div[1].text.split(" ")[0].replace(",", ".")
            assert is_number(price), f"Error parsing price: {price}"
            prices[seat] = price
        return prices

    @staticmethod
    def _get_trip_url(schedule_link: str, schedule_url: str) -> str:
        """
        Returns the URL of the trip schedule given the schedule link and the schedule URL

        Args:
            schedule_link (str): URL with stops information from Renfe
            schedule_url (str): Root URL of Renfe website for trip schedules

        Returns:
            str: URL of the trip schedule
        """
        query = str(schedule_link).replace("\n", "").replace("\t", "").replace(" ", "%20")
        pattern = r'\("(.+)"\)'  # Pattern to capture everything between parenthesis in a raw string
        match = re.search(pattern, query)
        if not match:
            raise ValueError("Invalid schedule link")

        query = match.group(1)
        return schedule_url + query

    @staticmethod
    def _load_stations_df(csv_path: str) -> pd.DataFrame:
        """
        Loads the available stations from a csv file.

        Args:
            csv_path (str): Path to the csv file.

        Returns:
            pd.DataFrame: A pandas DataFrame with the available stations.
        """
        return pd.read_csv(csv_path, dtype={"stop_id": str, "renfe_id": str})

    @staticmethod
    def _save_stops_df(df,
                       origin_id: str,
                       destination_id: str,
                       init_date: datetime.date,
                       end_date: datetime.date,
                       save_path: str) -> None:
        """
        Saves the dataframe with the stops information to a csv file.

        Args:
            df (pd.DataFrame): dataframe with the stops information
            origin_id (str): Renfe id of the origin station
            destination_id (str): Renfe id of the destination station
            init_date (datetime.date): initial date of the trip
            end_date (datetime.date): end date of the trip
            save_path (str): Path to save the csv file
        """
        schedules_dict = dict(zip(df.service_id, df.schedule))

        rows = []
        for service_id, schedule in schedules_dict.items():
            for stop_id, (arrival, departure) in schedule.items():
                rows.append({'service_id': service_id, 'stop_id': stop_id, 'arrival': arrival, 'departure': departure})

        df_stops = pd.DataFrame(rows)
        print(df_stops.head())

        os.makedirs(f'{save_path}/stopTimes/', exist_ok=True)
        df_stops.to_csv(
            f'{save_path}/stop_times/stopTimes_{origin_id}_{destination_id}_{init_date}_{end_date}.csv',
            index=False, header=True)

    @staticmethod
    def _save_trips_df(df: pd.DataFrame,
                       origin_id: str,
                       destination_id: str,
                       init_date: datetime.date,
                       end_date: datetime.date,
                       save_path: str) -> None:
        """
        Saves the dataframe with the trips information to a csv file.

        Args:
            df (pd.DataFrame): dataframe with the trips information
            origin_id (str): Renfe id of the origin station
            destination_id (str): Renfe id of the destination station
            init_date (datetime.date): initial date of the trip
            end_date (datetime.date): end date of the trip
            save_path (str): Path to save the csv file
        """
        df = df.drop(['schedule', 'price'], axis=1)
        print(df.head())
        os.makedirs(f'{save_path}/trips/', exist_ok=True)
        df.to_csv(f'{save_path}/trips/trips_{origin_id}_{destination_id}_{init_date}_{end_date}.csv',
                  index=False)

    @staticmethod
    def _scrape_stations(url: str) -> Dict[str, str]:
        """
        Scrapes the stations from the Renfe main menu.

        Args:
            url (str): URL of the Renfe main menu.

        Returns:
            Dict[str, str]: A dictionary with the Renfe station ids (str) as keys and the station names (str) as values.
        """
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        menu = soup.find('div', {'class': 'irf-search-shedule__container-ipt'})
        options = menu.find_all('option')
        return {opt["value"]: " ".join(filter(lambda x: x != "", opt.text.split(" "))) for opt in options}


if __name__ == "__main__":
    scraper = RenfeScraper()

    origin = "60000"
    destination = "71801"
    # date = datetime.date(day=14, month=4, year=2023)
    scraper.scrape(origin=origin, destination=destination)
