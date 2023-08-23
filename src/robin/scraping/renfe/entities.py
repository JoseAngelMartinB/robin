"""Entities for Renfe scraping."""

import bs4
import datetime
import os
import pandas as pd
import re
import requests
import unicodedata

from src.robin.scraping.renfe.utils import format_duration, is_number, remove_blanks, time_to_minutes

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from typing import Dict, List, Tuple, Union

# Renfe URL's
MAIN_MENU_URL = 'https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html'
SCHEDULE_URL = 'https://horarios.renfe.com/HIRRenfeWeb/'

# Renfe stations CSV path
SAVE_PATH = 'data/renfe'
RENFE_STATIONS_CSV = f'{SAVE_PATH}/renfe_stations.csv'


class DriverManager:
    """
    Driver manager to handle the webdriver and the scraping of the Renfe website.

    Attributes:
        driver (selenium.webdriver): Webdriver to handle the browser.
        stations_df (pd.DataFrame): Dataframe with the stations information.
        allowed_train_types (List[str]): List of allowed train types.
    """

    def __init__(self, stations_df: pd.DataFrame, allowed_train_types: List[str] = ['AVE', 'AVLO']) -> None:
        """
        Initializes the DriverManager object.

        Args:
            stations_df (pd.DataFrame): Dataframe with the stations information.
            allowed_train_types (List[str]): List of allowed train types. Defaults to ['AVE', 'AVLO'].
        """
        driver_options = Options()
        driver_options.add_argument('--disable-extensions')
        # driver_options.add_argument('--disable-gpu')
        driver_options.add_argument('--headless')  # Don't open browser window
        self.driver = webdriver.Chrome(options=driver_options)

        self.stations_df = stations_df
        self.allowed_train_types = allowed_train_types

    def _get_dataframe_from_records(
            self,
            records: List,
            col_names: List,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with the information retrieved from the scraping encoded in a list of lists.

        Each list in the list of lists represents the data of a service, and it becomes a row in the DataFrame.

        Args:
            records (List): List of lists with the information retrieved from the scraping.
            col_names (List): List of column names for the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the information retrieved from the scraping.
        """
        df = pd.DataFrame(records, columns=col_names)

        # Filter only AVE or AVLO trains
        train_type_filter = df['train_type'].str.contains('|'.join(self.allowed_train_types))
        df = df[train_type_filter].reset_index(drop=True)
        df['service_id'] = df.apply(lambda x: x['trip_id'] + '_' + x['departure'].strftime('%d-%m-%Y-%H.%M'), axis=1)
        return df

    def _get_name_best_match(self, raw_text: str) -> str:
        """
        Returns the best match between the name of the station and the name of the station in the stations CSV file.

        Args:
            raw_text (str): Name of the station as it appears in the Renfe website

        Returns:
            str: Best match between the name of the station and the name of the station in the stations CSV file.
        """
        # Get Adif station names from df, and convert them to lowercase, without spaces or dashes
        adif_names = [re.sub(r'[-/]', ' ', name.lower()) for name in self.stations_df['stop_name'].tolist()]

        raw_name = re.sub(r'[^a-zA-Z0-9 -]', '', raw_text)  # Remove non-alphanumeric characters
        name = ' '.join(filter(None, re.split(r'\W+', raw_name))).lower()

        best_match = max(adif_names,
                         key=lambda gn: sum(w in gn.split(' ') for w in name.split(' ')),
                         default='Unknown')

        if best_match != 'Unknown':
            return self.stations_df.at[adif_names.index(best_match), 'stop_id']
        return '00000'

    def _get_trip_data(
            self,
            row: bs4.element.ResultSet,
            date: datetime.date
    ) -> Tuple[str, str, Dict[str, Tuple[int, int]], datetime.datetime, int, Dict[str, float]]:
        """
        Returns the data of a trip retrieved from a row from the schedules table.

        Args:
            row (bs4.element.Tag): Row of the table with the trips information.
            date (datetime.date): Date of the trip.

        Returns:
            Tuple[str, str, str, datetime.datetime, float, Dict[str, float]]: Data of the trip from the schedules table.
        """
        trip_id = tuple(filter(None, re.split(r'\s+', row[0].text.strip())))[0]
        train_type = self._map_train_type(trip_id=trip_id)

        schedule_link = row[0].find('a')['href']
        trip_url = DriverManager._get_trip_url(schedule_link=schedule_link, schedule_url=SCHEDULE_URL)
        if train_type not in ['AVE', 'AVLO']:
            trip_schedule = 1
        else:
            trip_schedule = self._scrape_trip_schedule(url=trip_url)

        html_prices = re.sub(r'\s+', '', row[4].find('div').text)
        raw_prices = re.sub(r'PrecioInternet|:', '', html_prices).replace(',', '.')

        prices = {}
        for fare, price in re.findall(r'([a-zA-Z]+)[^\d]+([\d.]+)', raw_prices):
            prices[fare] = float(price)

        departure = row[1].text.strip()
        departure = datetime.datetime.strptime(str(date) + '-' + departure, '%Y-%m-%d-%H.%M')
        duration = format_duration(row[3].text.strip())
        train = (trip_id, train_type, trip_schedule, departure, duration, prices)
        assert all(train), f'Error parsing service trips: {train}'  # Assert non empty values

        return train

    def _is_content_row(self, row: bs4.element.ResultSet) -> bool:
        """
        Checks if row is a valid row with trip information.

        Args:
            row (bs4.element.ResultSet): Row of the table with the trips information.

        Returns:
            bool: True if row is a valid row with trip information, False otherwise.
        """
        if not row or len(row) < 6:
            return False
        return True

    def _scrape_trip_schedule(self, url: str) -> Dict[str, Tuple[int, int]]:
        """
        Returns dictionary of stops from URL with stops information from Renfe.

        Args:
            schedule_link (str): URL with stops information from Renfe.
            schedule_url (str): URL of Renfe website.

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

        relative_schedule = DriverManager._absolute_to_relative(trip_schedule)
        return relative_schedule

    def _map_train_type(self, trip_id: str) -> str:
        """
        Returns the train type given the trip id.

        Args:
            trip_id (str): Trip id.

        Returns:
            str: Train type.
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

    def _get_adif_station_id(self, renfe_id: str) -> str:
        """
        Gets the adif id of a station given its renfe id.

        Args:
            renfe_id (str): Renfe id of the station.

        Returns:
            str: Adif id of the station.
        """
        return self.stations_df[self.stations_df['renfe_id'] == renfe_id]['stop_id'].values[0]

    def _get_prices_dataframe(self, records: List, col_names: List) -> pd.DataFrame:
        """
        Returns a DataFrame with the information retrieved from the scraping encoded in a list of lists.

        Args:
            records (List): List of lists with the information retrieved from the scraping.

        Returns:
            pd.DataFrame: DataFrame with the information retrieved from the scraping.
        """
        df_prices = self._get_dataframe_from_records(records, col_names)

        def extract_values(row: pd.Series) -> Dict:
            """
            Extracts the values of the prices dictionary and returns them as a dictionary.

            Args:
                row (pd.Series): Row of the dataframe.

            Returns:
                Dict: Dictionary with the values of the prices dictionary.
            """
            return {k: v for k, v in row['prices'].items()}

        # Extract the values of the prices dictionary and add them as new columns
        new_columns = df_prices.apply(extract_values, axis=1, result_type='expand')
        df_prices = pd.concat([df_prices, new_columns], axis=1)  # Concatenate the new columns to the dataframe
        df_prices = df_prices.drop('prices', axis=1)  # Drop the prices column
        return df_prices

    def _get_seat_types(self, thead: bs4.element.Tag) -> List[str]:
        """
        Get a list of seat types from the table header.

        Args:
            thead (bs4.element.Tag): Table header of the table with the services.

        Returns:
            List[str]: List of seat types.
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

    def _get_service_prices(
            self,
            row: bs4.element.Tag,
            date: datetime.date,
            seat_types: List
    ) -> List:
        """
        Get the prices of the different seat types for a given row (service in Renfe website).

        Args:
            row (bs4.element.Tag): Row of the table with the services.
            date (datetime.date): Date of the service.
            seat_types (List): List of the seat types to get the prices for.

        Returns:
            List: A list with the different service prices for each seat type available in the service.
        """
        train_id = row['cdgotren']
        cols = row.find_all('td')
        first_col = cols[1].find_all('div')
        departure, duration = first_col[0].text, first_col[1].text
        departure_time = time_to_minutes(remove_blanks(x=departure, replace_by=''))
        init_datetime = datetime.datetime.combine(date, datetime.time(hour=0, minute=0))
        departure = init_datetime + datetime.timedelta(minutes=departure_time)
        duration = remove_blanks(x=duration, replace_by=' ')
        duration = format_duration(duration)
        duration = datetime.timedelta(minutes=duration)
        arrival = departure + duration
        train_type = self._map_train_type(trip_id=train_id)
        train_type = remove_blanks(x=train_type, replace_by=' ')
        price_cols = cols[4:-1]
        prices = DriverManager._get_prices(cols=price_cols, seat_types=seat_types)
        train = [train_id, train_type, departure, arrival, duration, prices]
        assert all(train), f'Error parsing service prices: {train}'
        return train

    def _get_renfe_schedules_url(self, origin_id: str, destination_id: str, date: datetime.date) -> str:
        """
        Returns the URL of the Renfe schedules website for a given origin-destination pair of stations and a given date.

        Args:
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            date (datetime.date): Date of the trip.

        Returns:
            str: URL of the Renfe schedules website for the given origin-destination pair of stations and date.
        """
        year, month, day = date.strftime('%Y-%m-%d').split("-")
        weekday = date.weekday() + 1
        url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
        print('Date: ', date)
        print('Search url: ', url)
        return url

    def _request_price(self, url: str, patience: int = 25) -> Union[str, bool]:
        """
        Request a page and wait for the price to load.

        Args:
            url (str): URL to request.
            patience (int, optional): Patience in seconds. Defaults to 12.

        Return:
            str: HTML of the page if the price loaded, False otherwise.
        """
        self.driver.get(url)

        try:
            WebDriverWait(self.driver, patience).until(EC.presence_of_element_located((By.CLASS_NAME, 'trayectoRow')))
        except TimeoutException:
            return False

        return self.driver.page_source

    def scrape_prices(
            self,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> Union[None, pd.DataFrame]:
        """
        Scrapes prices from Renfe website using selenium and saves retrieved data to a CSV file.

        Args:
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            date (datetime.date): Date of the trip.
            range_days (int): Number of days to search for trips.
            save_path (str): Path to save the CSV file.

        Returns:
            pd.DataFrame: DataFrame with the scraped data.
        """
        records = []
        date_str = date.strftime('%d-%m-%Y')  # Format date to match Renfe website format
        root = 'https://venta.renfe.com/vol/'
        query = f'buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional='

        url = root + query
        print('Date: ', date)
        print('Search url: ', url)
        html_str = self._request_price(url)
        if not html_str:
            print('Error retrieving prices. Skipping...')
            return pd.DataFrame()

        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('div', {'class': 'tab-content'})
        if not table:
            return None
        header = soup.find('thead')
        seat_types = self._get_seat_types(header)
        content_rows = table.find_all('tr', attrs={'cdgotren': True})
        origin = self._get_adif_station_id(origin_id)
        destination = self._get_adif_station_id(destination_id)
        for row in content_rows:
            data = self._get_service_prices(row, date, seat_types)
            data[1:1] = [origin, destination]
            train_type = self._map_train_type(data[0])
            if train_type not in self.allowed_train_types:
                continue
            records.append(data)
        if not records:
            return pd.DataFrame()
        col_names = ['trip_id', 'origin', 'destination', 'train_type', 'departure', 'arrival', 'duration', 'prices']
        return self._get_prices_dataframe(records=records, col_names=col_names)

    def scrape_stations(self, url: str) -> Dict[str, str]:
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
        return {opt['value']: ' '.join(filter(lambda x: x != '', opt.text.split(' '))) for opt in options}
    
    def scrape_trips(
            self,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> Union[bool, pd.DataFrame]:
        """
        Obtains two pandas dataframes from Renfe website, one with the trips information and another with the stops,
            which are saved to CSV files.

        Args:
            origin_id (str): Origin station id.
            destination_id (str): Destination station id.
            date (datetime.date): Initial date to search for trips.
            range_days (int): Number of days to search for trips from the initial date.
            save_path (str): Path to save the CSV files.

        Returns:
            pd.DataFrame: DataFrame with the scraped trips data.
        """
        rows = []
        url = self._get_renfe_schedules_url(origin_id, destination_id, date)
        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')
        main_table = soup.select_one('.irf-travellers-table__container-table')

        if not main_table:
            return None

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
        return self._get_dataframe_from_records(records=rows, col_names=col_names)

    @staticmethod
    def _absolute_to_relative(schedule: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """
        Converts absolute times to relative times.

        Args:
            schedule (Dict[str, Tuple[int, int]]): Dictionary of stops, where keys are each station and values are
                a tuple of floats with (arrival, departure) times in minutes from midnight.

        Returns:
            Dict[str, Tuple[int, int]]: Same dictionary of stops, where keys are each station and values are a tuple
                with (arrival, departure) relative time in minutes.
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
            cols (bs4.element.ResultSet): Columns of the table with the services.
            seat_types (List): List of the seat types.

        Returns:
            Dict[str, float]: Dictionary with the prices of the different seat types.
        """
        if len(cols) == 1:  # Means train is full, so no prices are available
            return {st: float('NaN') for st in seat_types}

        prices = {}
        # If any of the prices is available, number of price cols will be equal to number of seat types
        for seat, col in zip(seat_types, cols):
            button_col = col.find('button')  # Button columns contains the prices info
            price = float('NaN')
            price_div = button_col.find('div', {'class': 'precio booking-list-element-big-font'})
            if price_div:
                price = price_div.text.split(' ')[0].replace(',', '.')
            assert is_number(price), f'Error parsing price: {price}'
            prices[seat] = price

        if len(prices) != len(seat_types):
            return {st: float('NaN') for st in seat_types}
        return prices

    @staticmethod
    def _get_trip_url(schedule_link: str, schedule_url: str) -> str:
        """
        Returns the URL of the trip schedule given the schedule link and the schedule URL.

        Args:
            schedule_link (str): URL with stops information from Renfe.
            schedule_url (str): Root URL of Renfe website for trip schedules.

        Returns:
            str: URL of the trip schedule.
        """
        query = str(schedule_link).replace("\n", "").replace("\t", "").replace(" ", "%20")
        pattern = r'\("(.+)"\)'  # Pattern to capture everything between parenthesis in a raw string
        match = re.search(pattern, query)
        if not match:
            raise ValueError('Invalid schedule link')

        query = match.group(1)
        return schedule_url + query


class RenfeScraper:
    """
    Renfe Scraping class.

    Attributes:
        stations_df (pd.DataFrame): A pandas DataFrame to parse stations data.
        available_stations (List[str]): A list with the available stations in the Renfe website.
        driver (DriverManager): A DriverManager object to manage the browser.
    """

    def __init__(
            self,
            stations_csv_path: str = RENFE_STATIONS_CSV,
            menu_url: str = MAIN_MENU_URL,
            allowed_train_types: List[str] = ['AVE', 'AVLO']
    ) -> None:
        """
        Initialize a RenfeScraper object.

        Args:
            stations_csv_path (str): Path to the CSV file with the available stations.
            menu_url (str): URL of the Renfe main menu.
            allowed_train_types (List[str]): List of allowed train types to scrape.
        """
        self.stations_df = pd.read_csv(stations_csv_path, dtype={'stop_id': str, 'renfe_id': str})
        self.driver = DriverManager(
            stations_df=self.stations_df,
            allowed_train_types=allowed_train_types
        )
        self.available_stations = self.driver.scrape_stations(menu_url)

    def _get_corridor_stations(self, trips_df: pd.DataFrame) -> List[str]:
        """
        Get the stations of the corridor from the trips DataFrame.

        Args:
            trips_df (pd.DataFrame): DataFrame with the trips information.

        Returns:
            List[str]: List with the stations of the corridor.
        """
        # Initialize corridor with max length trip
        schedules = trips_df['schedule'].values.tolist()
        corridor_stations = list(schedules.pop(schedules.index(max(schedules, key=len))))

        # Complete corridor with other stops that are not in the initial defined corridor
        for trip in schedules:
            for i, station in enumerate(trip):
                if station not in corridor_stations:
                    # If station is the last one, append it to the end of the corridor
                    if i == len(trip) - 1:
                        corridor_stations.append(station)
                    else:
                        # If station is not the last one, insert it in the corridor before the next station
                        index = corridor_stations.index(trip[i + 1])
                        corridor_stations.insert(index, station)

        return corridor_stations

    def _get_renfe_station_id(self, adif_id: str) -> str:
        """
        Returns the Renfe station id given the Adif station id.

        Args:
            adif_id (str): Adif station id.

        Returns:
            str: Renfe station id.
        """
        return self.stations_df[self.stations_df['stop_id'] == adif_id]['renfe_id'].values[0]

    def _save_df_stops(
            self,
            df_trips: pd.DataFrame,
            origin_id: str,
            destination_id: str,
            init_date: datetime.date,
            end_date: datetime.date,
            save_path: str
    ) -> None:
        """
        Saves the dataframe with the stops information to a CSV file.

        Args:
            df_trips (pd.DataFrame): Dataframe with the trips information.
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            init_date (datetime.date): initial date of the trip.
            end_date (datetime.date): end date of the trip.
            save_path (str): Path to save the CSV file.
        """
        # Create a dictionary with the service_id as key and the schedule as value
        schedules_dict = dict(zip(df_trips.service_id, df_trips.schedule))

        # Create a list of dictionaries with the stop information
        rows = []
        for service_id, schedule in schedules_dict.items():
            for stop_id, (arrival, departure) in schedule.items():
                rows.append({'service_id': service_id, 'stop_id': stop_id, 'arrival': arrival, 'departure': departure})

        # Create a stops DataFrame with the list of dictionaries and save it to a CSV file
        df_stops = pd.DataFrame(rows)
        os.makedirs(f'{save_path}/stop_times/', exist_ok=True)
        df_stops.to_csv(
            f'{save_path}/stop_times/stopTimes_{origin_id}_{destination_id}_{init_date}_{end_date}.csv',
            index=False
        )

    def scrape(
            self,
            origin: str,
            destination: str,
            init_date: datetime.date = None,
            range_days: int = 1,
            save_path: str = SAVE_PATH
    ) -> None:
        """
        Scrapes the Renfe website for the trips and prices of services between two stations.

        Args:
            origin (str): Adif station id of the origin station.
            destination (str): Adif station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            save_path (str): Path to save the csv files.
        """
        # Convert Adif station ids to Renfe station ids
        origin_id = self._get_renfe_station_id(origin)
        destination_id = self._get_renfe_station_id(destination)

        # Assert that the origin and destination stations are in the list of stations operated by Renfe
        pair_of_stations_in_csv = all(s in self.available_stations.keys() for s in (origin_id, destination_id))
        assert pair_of_stations_in_csv, 'Invalid origin or destination'

        # If no initial date is provided, use today's date
        if not init_date:
            init_date = datetime.date.today()

        # Scrape trips
        df_trips = self.scrape_trips(
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            range_days=range_days,
            save_path=save_path
        )
        end_date = init_date + datetime.timedelta(days=range_days)
        print(f'Scraped {len(df_trips)} trips between {origin_id} and {destination_id} from {init_date} to {end_date}')
        print(df_trips.head())

        # Scrape prices
        df_prices = self.scrape_prices(
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            range_days=range_days,
            df_trips=df_trips,
            save_path=save_path
        )
        print(f'Scraped prices between {origin_id} and {destination_id} from {init_date} to {end_date}')
        print(df_prices.head())

    def scrape_prices(
            self,
            origin_id: str,
            destination_id: str,
            init_date: datetime.date = None,
            range_days: int = 1,
            df_trips: pd.DataFrame = None,
            save_path: str = SAVE_PATH
    ) -> pd.DataFrame:
        """
        Scrapes the Renfe website for the prices of services between two stations.

        Args:
            origin_id (str): Renfe station id of the origin station.
            destination_id (str): Renfe station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            df_trips (pd.DataFrame): DataFrame containing the scraped trips.
            save_path (str): Path to save the scraped data.
        
        Returns:
            pd.DataFrame: DataFrame containing the scraped prices.
        """
        # Get corridor stations
        corridor_stations = self._get_corridor_stations(df_trips)

        # Scrape prices
        end_date = init_date + datetime.timedelta(days=range_days)
        df_prices = pd.DataFrame()
        for org in corridor_stations:
            # If the origin station is the same as the destination station, skip it
            for des in corridor_stations[corridor_stations.index(org) + 1:]:
                date = init_date
                for _ in range(range_days):
                    org_id = self._get_renfe_station_id(org)
                    des_id = self._get_renfe_station_id(des)
                    new_df_prices = self.driver.scrape_prices(origin_id=org_id, destination_id=des_id, date=date)
                    if new_df_prices is None:
                        print(f'No prices found for {org_id} - {des_id} on {date}. Exiting...')
                        break
                    df_prices = pd.concat([df_prices, new_df_prices], ignore_index=True)
                    date += datetime.timedelta(days=1)
                else:
                    continue
                break
            else:
                continue
            break

        # Save prices
        os.makedirs(f'{save_path}/prices/', exist_ok=True)
        df_prices.to_csv(
            f'{save_path}/prices/prices_{origin_id}_{destination_id}_{init_date}_{end_date}.csv',
            index=False
        )
        return df_prices

    def scrape_trips(
            self,
            origin_id: str,
            destination_id: str,
            init_date: datetime.date = None,
            range_days: int = 1,
            save_path: str = SAVE_PATH
    ) -> pd.DataFrame:
        """
        Scrapes the Renfe website for the trips between two stations.

        Args:
            origin_id (str): Renfe station id of the origin station.
            destination_id (str): Renfe station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            save_path (str): Path to save the scraped data.

        Returns:
            pd.DataFrame: DataFrame containing the scraped trips.
        """
        # Scrape trips from the schedules table
        date = init_date
        end_date = init_date + datetime.timedelta(days=range_days)
        df_trips = pd.DataFrame()
        for _ in range(range_days):
            new_df_trips = self.driver.scrape_trips(origin_id=origin_id, destination_id=destination_id, date=date)
            if new_df_trips is None:
                print(f'No trips found for {origin_id} - {destination_id} on {date}. Exiting...')
                break
            df_trips = pd.concat([df_trips, new_df_trips], ignore_index=True)
            date += datetime.timedelta(days=1)

        self._save_df_stops(
            df_trips=df_trips,
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            end_date=end_date,
            save_path=save_path
        )
        return df_trips
