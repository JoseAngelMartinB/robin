"""Entities for Renfe scraping."""

import bs4
import datetime
import os
import pandas as pd
import re
import requests
import unicodedata

from src.robin.scraping.exceptions import NotAvailableStationsException
from src.robin.scraping.renfe.utils import format_duration, is_number, time_to_minutes

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from typing import Dict, List, Tuple, Union

# Renfe URL's
MAIN_MENU_URL = 'https://www.renfe.com/content/renfe/es/es/viajar/informacion-util/horarios/app-horarios.html'
SCHEDULE_URL = 'https://horarios.renfe.com/HIRRenfeWeb/'

# Renfe stations CSV path
SAVE_PATH = 'data/renfe'
RENFE_STATIONS_CSV = f'{SAVE_PATH}/renfe_stations.csv'
LR_RENFE_SERVICES = ('AVE', 'AVLO', 'AVE INT', 'ALVIA', 'AVANT')

# Default values
DEFAULT_PATIENCE = 25


class DriverManager:
    """
    Driver manager to handle the webdriver and the scraping of the Renfe website.

    Attributes:
        driver (selenium.webdriver): Webdriver to handle the browser.
        stations_df (pd.DataFrame): Dataframe with the stations' information.
        allowed_train_types (List[str]): List of allowed train types.
    """

    def __init__(
            self,
            stations_df: pd.DataFrame,
            allowed_train_types: List[str] = LR_RENFE_SERVICES
    ) -> None:
        """
        Initializes the DriverManager object.

        Args:
            stations_df (pd.DataFrame): Dataframe with the stations' information.
            allowed_train_types (List[str]): List of allowed train types.
        """
        driver_options = Options()
        driver_options.add_argument('--disable-extensions')
        # driver_options.add_argument('--headless')
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

        train_type_filter = df['train_type'].str.contains('|'.join(self.allowed_train_types))
        df = df[train_type_filter].reset_index(drop=True)
        if df.empty:
            df['service_id'] = pd.Series(dtype='str')
            return df
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
        print(f'Unknown station: {name}')
        return '00000'

    # TODO: REMOVE
    def _get_trip_data(
            self,
            row: bs4.element.ResultSet,
            date: datetime.date,
            origin_id: str,
            destination_id: str
    ) -> Tuple[str, str, Dict[str, Tuple[int, int]], datetime.datetime, int]:
        """
        Returns the data of a trip retrieved from a row from the schedules table.

        Args:
            row (bs4.element.Tag): Row of the table with the trips information.
            date (datetime.date): Date of the trip.

        Returns:
            Tuple[str, str, str, datetime.datetime, float, Dict[str, float]]: Data of the trip from the schedules table.
        """
        trip_info = list(filter(None, re.split(r'\s+', row[0].text.strip())))
        trip_id, *train_type = trip_info
        train_type = ' '.join(train_type)

        schedule_link = row[0].find('a')['href']
        trip_url = DriverManager._get_trip_url(schedule_link=schedule_link, schedule_url=SCHEDULE_URL)
        trip_schedule = self._scrape_trip_schedule(url=trip_url)

        departure = row[1].text.strip()
        departure = datetime.datetime.strptime(str(date) + '-' + departure, '%Y-%m-%d-%H.%M')
        duration = format_duration(row[3].text.strip())
        train = (trip_id, train_type, trip_schedule, departure, duration)
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

    def _scrape_trip_schedule(
            self,
            url: str,
            origin_id: str,
            destination_id: str
        ) -> Dict[str, Tuple[int, int]]:
        """
        Returns dictionary of stops from URL with stops information from Renfe.

        Args:
            url (str): URL with stops information from Renfe.
            origin_id (str): Origin station id.
            destination_id (str): Destination station id.

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

        # Extract the values of the prices dictionary and add them as new columns
        new_columns = df_prices.apply(lambda row: {k: v for k, v in row['prices'].items()}, axis=1, result_type='expand')
        df_prices = pd.concat([df_prices, new_columns], axis=1)  # Concatenate the new columns to the dataframe
        df_prices = df_prices.drop('prices', axis=1)  # Drop the prices column
        return df_prices

    def _get_renfe_prices_url(self, origin_id: str, destination_id: str, date: datetime.date) -> str:
        """
        Returns the URL of the Renfe prices website for a given origin-destination pair of stations and a given date.

        Args:
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            date (datetime.date): Date of the trip.

        Returns:
            str: URL of the Renfe prices website for the given origin-destination pair of stations and date.
        """
        date_str = date.strftime('%d/%m/%Y')  # Format date to match Renfe website format
        root = 'https://venta.renfe.com/vol/'
        query = f'buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional='

        url = root + query
        print('Date: ', date)
        print('Search url: ', url)
        return url

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

    def _request_url(
            self,
            url: str,
            find_by: str,
            find_value: str,
            patience: int = DEFAULT_PATIENCE
    ) -> Union[WebElement, None]:
        """
        Request a page and wait for the price to load.

        Args:
            url (str): URL to request.
            find_by (str): By method to find the element.
            find_value (str): Value to find the element.
            patience (int, optional): Patience in seconds. Defaults to 25.

        Return:
            Union[WebElement, None]: WebElement with the prices or None if the prices are not loaded.
        """
        self.driver.get(url)
        div = self.driver.find_element(find_by, find_value)
        try:
            WebDriverWait(self.driver, patience).until(EC.visibility_of_element_located((find_by, find_value)))
        except TimeoutException:
            return None
        return div

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
        url = self._get_renfe_prices_url(origin_id, destination_id, date)
        prices = self._request_url(url=url, find_by=By.ID, find_value='listaTrenesTBodyIda')
        if not prices:
            print('Error retrieving prices. Skipping...')
            return pd.DataFrame()
        df_prices = self._get_df_prices(prices, origin_id, destination_id, date)
        return df_prices

    def _get_df_prices(
            self,
            prices: WebElement,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with the prices of the trains.

        Args:
            prices (WebElement): WebElement with the prices table.
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            date (datetime.date): Date of the trip.

        Returns:
            pd.DataFrame: DataFrame with the prices of the trains.
        """
        trains = prices.find_elements(By.CSS_SELECTOR, '.row.selectedTren')
        records = []
        for train in trains:
            trip_id = self._get_prices_trip_id(train)
            if not trip_id:
                continue
            origin = self._get_adif_station_id(origin_id)
            destination = self._get_adif_station_id(destination_id)
            train_type = self._get_prices_train_type(train)
            departure, arrival, duration = self._get_prices_train_schedule(train, date)
            train_prices = self._get_prices_train(train)
            if not self._is_allowed_train_type(train_type):
                continue
            train_record = [trip_id, origin, destination, train_type, departure, arrival, duration, train_prices]
            records.append(train_record)

        if not records:
            return pd.DataFrame()
        col_names = ['trip_id', 'origin', 'destination', 'train_type', 'departure', 'arrival', 'duration', 'prices']
        return self._get_prices_dataframe(records=records, col_names=col_names)

    def _is_allowed_train_type(self, train_type: str) -> bool:
        """
        Checks if the train type is allowed.

        Args:
            train_type (str): Train type.

        Returns:
            bool: True if the train type is allowed, False otherwise.
        """
        return train_type in ' '.join(self.allowed_train_types)

    def _get_prices_trip_id(self, train: WebElement) -> Union[str, None]:
        """
        Returns the trip id of a train.

        Args:
            train (WebElement): Train element.

        Returns:
            Union[str, None]: Trip id of the train or None if there are no purchase options.
        """
        try:
            trip_id_data = train.find_element(By.CLASS_NAME, 'estilo-box-card')
        except NoSuchElementException:
            return None
        return trip_id_data.get_attribute('data-cod-tpenlacesilencio').split('#')[-1]

    def _get_prices_train_type(self, train: WebElement) -> str:
        """
        Returns the train type.

        Args:
            train (WebElement): Train element.

        Returns:
            str: Train type of the train.
        """
        train_type_img = train.find_element(By.CLASS_NAME, 'img-fluid')
        train_type = train_type_img.get_attribute('alt').split('Tipo de tren')[-1]
        return train_type

    def _get_prices_train_schedule(
            self,
            train: WebElement,
            date: datetime.date
    ) -> Tuple[datetime.datetime, datetime.datetime, datetime.timedelta]:
        """
        Returns the schedule of a train, with departure, arrival and duration in minutes.

        Args:
            train (WebElement): Train element.
            date (datetime.date): Date of the train (day, month, year).

        Returns:
            Tuple[datetime.datetime, datetime.datetime, datetime.timedelta]: Schedule of the train, with departure,
                arrival and duration.
        """
        train_schedule_data = train.find_element(By.CSS_SELECTOR, '.col-md-8.trenes')
        departure_data, arrival_data = train_schedule_data.find_elements(By.TAG_NAME, 'h5')
        departure = self._time_to_datetime(departure_data.text, date)
        arrival = self._time_to_datetime(arrival_data.text, date)
        duration = arrival - departure
        return departure, arrival, duration

    def _time_to_datetime(self, time: str, date: datetime.date) -> datetime.datetime:
        """
        Converts a time string to a datetime object.

        Args:
            time (str): Time string.
            date (datetime.date): Date of the time.

        Returns:
            datetime.datetime: Datetime object of the time.
        """
        time = time.replace(' h', '')
        return datetime.datetime.strptime(str(date) + ' ' + time, '%Y-%m-%d %H:%M')

    def _get_prices_train(self, train: WebElement) -> Dict[str, float]:
        """
        Returns the prices of a train in a dictionary with the seat types as keys and the prices as values.

        Args:
            train (WebElement): Train element.

        Returns:
            Dict[str, float]: Prices of the train. Keys are the seat types and values are the prices.
        """
        prices = {}
        prices_data = train.find_elements(By.CSS_SELECTOR, '.seleccion-resumen-bottom.card.bg-light.mb-3')
        for price_data in prices_data:
            seat_type = price_data.get_attribute('data-titulo-tarifa')
            price = price_data.get_attribute('data-precio-tarifa')
            prices[seat_type] = float(price.replace(',', '.'))
        return prices

    def scrape_stations(self, url: str) -> Dict[str, str]:
        """
        Scrapes the stations from the Renfe main menu.

        Args:
            url (str): URL of the Renfe main menu.

        Returns:
            Dict[str, str]: A dictionary with the Renfe station ids (str) as keys and the station names (str) as values.
        """
        menu = self._request_url(url=url, find_by=By.CLASS_NAME, find_value='irf-select')
        if not menu:
            raise NotAvailableStationsException

        stations = menu.find_elements(By.TAG_NAME, 'option')

        ids_names = {}
        for station in stations:
            station_name = station.text
            if station_name == 'Estaciones de Origen':
                continue
            station_id = station.get_attribute('value')
            ids_names[station_id] = station_name

        return ids_names
    
    def scrape_trips(
            self,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> Union[None, pd.DataFrame]:
        """
        Obtains two pandas dataframes from Renfe website, one with the trips information and another with the stops,
            which are saved to CSV files.

        Args:
            origin_id (str): Origin station id.
            destination_id (str): Destination station id.
            date (datetime.date): Initial date to search for trips.

        Returns:
            pd.DataFrame: DataFrame with the scraped trips data.
        """
        url = self._get_renfe_schedules_url(origin_id, destination_id, date)
        trips = self._request_url(url=url, find_by=By.CLASS_NAME, find_value='irf-travellers-table__container-table')
        if not trips:
            print('Error retrieving trips. Skipping...')
            return pd.DataFrame()
        df_trips = self._get_df_trips(trips, origin_id, destination_id, date)
        return df_trips

    def _get_df_trips(
            self,
            trips: WebElement,
            origin_id: str,
            destination_id: str,
            date: datetime.date
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with the trips information.

        Args:
            table (WebElement): WebElement with the trips table.
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            date (datetime.date): Date of the trip.

        Returns:
            pd.DataFrame: DataFrame with the trips information.
        """
        trains = trips.find_elements(By.CSS_SELECTOR, '.odd.irf-travellers-table__tr')
        records = []
        for train in trains:
            trip_id, train_type = self._get_trips_trip_id_train_type(train)
            schedule = self._get_trips_schedule(train, date)
            departure = self._get_trips_departure()
            duration = self._get_trips_duration()
            if not self._is_allowed_train_type(train_type):
                continue
            trip_record = [trip_id, train_type, schedule, departure, duration]
            records.append(trip_record)

        if not records:
            return pd.DataFrame()
        col_names = ['trip_id', 'train_type', 'schedule', 'departure', 'duration']
        return self._get_dataframe_from_records(records=records, col_names=col_names)

    def _get_trips_trip_id_train_type(self, train: WebElement) -> Tuple[str, str]:
        """
        Returns the trip id and train type of the train.

        Args:
            train (WebElement): Trip element.

        Returns:
            Tuple[str, str]: Trip id and train type of the trip.
        """
        train_info = train.find_element(
            By.CSS_SELECTOR,
            '.irf-travellers-table__tbody-lnk.irf-travellers-table__tbody-lnk--icon-left'
        )
        trip_id, *train_type = train_info.text.split(' ')
        train_type = ' '.join(train_type)
        return trip_id, train_type

    def _get_trips_schedule(
            self,
            train: WebElement,
            date: datetime.date
    ) -> pd.DataFrame:
        train_info = train.find_element(
            By.CSS_SELECTOR,
            '.irf-travellers-table__tbody-lnk.irf-travellers-table__tbody-lnk--icon-left'
        )
        schedule_link = train_info.get_attribute('href')
        schedule_link = schedule_link.replace('javascript:abrirNuevaVentana("', '')
        schedule_link = schedule_link.replace('%22)', '')
        schedule_link = 'https://horarios.renfe.com/HIRRenfeWeb/' + schedule_link
        self.driver.get(schedule_link)

        schedule = {}
        schedule_table = self.driver.find_element(By.CLASS_NAME, 'irf-renfe-travel__container-table')
        train_stops = schedule_table.find_elements(By.CSS_SELECTOR, '.irf-renfe-travel__td.txt_gral')
        it = iter(train_stops)
        for train_stop, arrival, departure in zip(it, it, it):
            print(train_stop.text, arrival.text, departure.text)
            schedule[train_stop.text] = (arrival.text, departure.text)

        print(schedule)
        return schedule

    def _get_trips_departure(self):
        pass

    def _get_trips_duration(self):
        pass

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
            allowed_train_types: List[str] = LR_RENFE_SERVICES
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

    def get_renfe_station_id(self, adif_id: str) -> str:
        """
        Returns the Renfe station id given the Adif station id.

        Args:
            adif_id (str): Adif station id.

        Returns:
            str: Renfe station id.
        """
        return self.stations_df[self.stations_df['stop_id'] == adif_id]['renfe_id'].values[0]

    def get_df_stops(self, df_trips: pd.DataFrame):
        # Create a dictionary with the service_id as key and the schedule as value
        schedules_dict = dict(zip(df_trips.service_id, df_trips.schedule))

        # Create a list of dictionaries with the stop information
        rows = []
        for service_id, schedule in schedules_dict.items():
            for stop_id, (arrival, departure) in schedule.items():
                rows.append({'service_id': service_id, 'stop_id': stop_id, 'arrival': arrival, 'departure': departure})

        return pd.DataFrame(rows)

    def _save_df_stops(
            self,
            df_stops: pd.DataFrame,
            origin_id: str,
            destination_id: str,
            init_date: datetime.date,
            end_date: datetime.date,
            save_path: str
    ) -> None:
        """
        Saves the dataframe with the stops information to a CSV file.

        Args:
            df_stops (pd.DataFrame): Dataframe with the stops information.
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            init_date (datetime.date): initial date of the trip.
            end_date (datetime.date): end date of the trip.
            save_path (str): Path to save the CSV file.
        """
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
        origin_id = self.get_renfe_station_id(origin)
        destination_id = self.get_renfe_station_id(destination)

        # Assert that the origin and destination stations are in the list of stations operated by Renfe
        pair_of_stations_in_csv = all(s in self.available_stations.keys() for s in (origin_id, destination_id))
        assert pair_of_stations_in_csv, 'Invalid origin or destination'

        # If no initial date is provided, use today's date
        if not init_date:
            init_date = datetime.date.today()

        # Scrape trips
        df_trips, _ = self.scrape_trips(
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
                    org_id = self.get_renfe_station_id(org)
                    des_id = self.get_renfe_station_id(des)
                    new_df_prices = self.driver.scrape_prices(origin_id=org_id, destination_id=des_id, date=date)
                    if new_df_prices.empty:
                        print(f'No prices found for {org_id} - {des_id} on {date}. Exiting...')
                        continue
                    df_prices = pd.concat([df_prices, new_df_prices], ignore_index=True)
                    date += datetime.timedelta(days=1)

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
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrame containing the scraped trips and scraped stops.
        """
        # Scrape trips from the schedules table
        date = init_date
        end_date = init_date + datetime.timedelta(days=range_days)
        df_trips = pd.DataFrame()
        for _ in range(range_days):
            new_df_trips = self.driver.scrape_trips(origin_id=origin_id, destination_id=destination_id, date=date)
            if new_df_trips is None:
                print(f'No trips found for {origin_id} - {destination_id} on {date}. Exiting...')
                continue
            df_trips = pd.concat([df_trips, new_df_trips], ignore_index=True)
            date += datetime.timedelta(days=1)

        if df_trips.empty:
            return pd.DataFrame()

        df_stops = self.get_df_stops(df_trips)

        self._save_df_stops(
            df_stops=df_stops,
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            end_date=end_date,
            save_path=save_path
        )
        return df_stops
