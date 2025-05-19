"""Entities for the scraping Renfe module."""

import datetime
import os
import pandas as pd

from robin.scraping.renfe.constants import (
    ALLOWED_RENFE_SERVICES, DEFAULT_PATIENCE, MAIN_MENU_URL, MAX_ATTEMPTS,
    ONE_DAY, PRICES_URL, RENFE_TSP, RENFE_STATIONS_CSV, SAVE_PATH, SCHEDULE_URL
)
from robin.scraping.renfe.exceptions import NotAvailableStationsException
from robin.scraping.renfe.utils import time_str_to_minutes, time_to_datetime, time_to_minutes

from loguru import logger
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from typing import Any, Dict, List, Mapping, Set, Tuple, Union


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
        allowed_train_types: List[str] = ALLOWED_RENFE_SERVICES
    ) -> None:
        """
        Initializes the DriverManager object.

        Args:
            stations_df (pd.DataFrame): Dataframe with the stations' information.
            allowed_train_types (List[str]): List of allowed train types.
        """
        driver_options = Options()
        driver_options.add_argument('--disable-extensions')
        driver_options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=driver_options)
        self.stations_df = stations_df
        self.allowed_train_types = allowed_train_types

    def _get_df_from_records(self, records: List[List[Any]], col_names: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with the information retrieved from the scraping encoded in a list of lists.

        Each list in the list of lists represents the data of a service, and it becomes a row in the DataFrame.

        Args:
            records (List[List[Any]]): List of lists with the information retrieved from the scraping.
            col_names (List[str]): List of column names for the DataFrame.

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
            if self._is_transfer_train(train):
                continue
            trip_id = self._get_prices_trip_id(train)
            if not trip_id:
                continue
            origin = self.get_value_from_stations(search_column='RENFE_ID', value=origin_id, objective_column='ADIF_ID')
            destination = self.get_value_from_stations(search_column='RENFE_ID', value=destination_id, objective_column='ADIF_ID')
            tsp = RENFE_TSP
            train_type = self._get_prices_train_type(train)
            departure, arrival, duration = self._get_prices_train_schedule(train, date)
            train_prices = self._get_prices_train(train)
            if not self._is_allowed_train_type(train_type):
                continue
            train_record = [trip_id, origin, destination, tsp, train_type, departure, arrival, duration, train_prices]
            records.append(train_record)

        if not records:
            return pd.DataFrame()
        col_names = ['trip_id', 'origin', 'destination', 'tsp', 'train_type', 'departure', 'arrival', 'duration', 'prices']
        return self._get_prices_dataframe(records=records, col_names=col_names)

    def _get_df_trips(self, trips: WebElement, date: datetime.date) -> pd.DataFrame:
        """
        Returns a DataFrame with the trips information.

        Args:
            trips (WebElement): WebElement with the trips table.
            date (datetime.date): Date of the trip.

        Returns:
            pd.DataFrame: DataFrame with the trips information.
        """
        trains = trips.find_elements(By.CSS_SELECTOR, '.odd.irf-travellers-table__tr')
        records = []
        for train in trains:
            trip_id, train_type = self._get_trips_trip_id_train_type(train)
            if not trip_id:
                logger.warning('No trip id found. Skipping...')
                continue
            schedule = self._get_trips_schedule(train)
            departure = self._get_trips_departure(train, date)
            duration = self._get_trips_duration(train)
            if not self._is_allowed_train_type(train_type):
                continue
            trip_record = [trip_id, train_type, schedule, departure, duration]
            records.append(trip_record)

        if not records:
            return pd.DataFrame()
        col_names = ['trip_id', 'train_type', 'schedule', 'departure', 'duration']
        return self._get_df_from_records(records=records, col_names=col_names)

    def _get_prices_dataframe(self, records: List[List[Any]], col_names: List[str]) -> pd.DataFrame:
        """
        Returns a DataFrame with the information retrieved from the scraping encoded in a list of lists.

        Args:
            records (List[List[Any]]): List of lists with the information retrieved from the scraping.
            col_names (List[str]): List of column names for the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the information retrieved from the scraping.
        """
        df_prices = self._get_df_from_records(records, col_names)

        # Extract the values of the prices dictionary and add them as new columns
        new_columns = df_prices.apply(lambda row: {k: v for k, v in row['prices'].items()}, axis=1, result_type='expand')
        df_prices = pd.concat([df_prices, new_columns], axis=1)

        # Drop the prices column as it is no longer needed
        df_prices = df_prices.drop('prices', axis=1)
        return df_prices

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
        train_type = train_type_img.get_attribute('alt').split('Tipo de tren ')[-1]
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
        departure = time_to_datetime(departure_data.text, date)
        arrival = time_to_datetime(arrival_data.text, date)
        duration = arrival - departure
        return departure, arrival, duration

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

    def _get_relative_schedule(self, train_stops: List[WebElement]) -> Mapping[str, Tuple[int, int]]:
        """
        Converts absolute schedule times to relative times.

        Args:
            train_stops (List[WebElement]): List of WebElements with the schedule times.

        Returns:
            Mapping[str, Tuple[int, int]]: Dictionary of stops, where keys are each station as
                adif ids and values are a tuple of relative times in minutes.
        """
        schedule = {}
        it = iter(train_stops)
        total_stations = len(train_stops) // 3
        init_time = 0
        init_minutes = 0
        prev_departure = 0

        for current_station, (train_stop, arrival, departure) in enumerate(zip(it, it, it)):
            adif_id = self.get_value_from_stations(
                search_column='STATION_NAME', value=train_stop.text, objective_column='ADIF_ID'
            )
            arrival_absolute = time_to_minutes(arrival.text)
            departure_absolute = time_to_minutes(departure.text)

            # If the train arrives before the previous departure, it means that the train has passed midnight
            if arrival_absolute < prev_departure:
                init_minutes += ONE_DAY
            prev_departure = departure_absolute
            arrival_absolute += init_minutes
            departure_absolute += init_minutes

            # Convert absolute times to relative times
            if current_station == 0:
                init_time = departure_absolute
                arrival_relative = 0
                departure_relative = 0
            elif current_station == total_stations - 1:
                arrival_relative = arrival_absolute - init_time
                departure_relative = arrival_relative
            else:
                arrival_relative = arrival_absolute - init_time
                departure_relative = departure_absolute - init_time
            schedule[adif_id] = (arrival_relative, departure_relative)
        return schedule

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
        date_str = date.strftime('%d/%m/%Y')
        url = PRICES_URL.format(origin_id=origin_id, destination_id=destination_id, date_str=date_str)
        logger.info(url)
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
        year, month, day = date.strftime('%Y-%m-%d').split('-')
        weekday = date.weekday() + 1
        url = SCHEDULE_URL.format(origin_id=origin_id, destination_id=destination_id, year=year, month=month, day=day, weekday=weekday)
        logger.info(url)
        return url

    def _get_trips_trip_id_train_type(self, train: WebElement) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Returns the trip id and train type of the train.

        Args:
            train (WebElement): Trip element.

        Returns:
            Tuple[Union[str, None], Union[str, None]]: Trip id and train type of the trip. Both values are None if
                elements are not found.
        """
        try:
            train_info = train.find_element(
                By.CSS_SELECTOR,
                '.irf-travellers-table__tbody-lnk.irf-travellers-table__tbody-lnk--icon-left'
            )
        except NoSuchElementException:
            return None, None
        trip_id, *train_type = train_info.text.split(' ')
        train_type = ' '.join(train_type)
        return trip_id, train_type

    def _get_trips_schedule(self, train: WebElement) -> Mapping[str, Tuple[int, int]]:
        """
        Returns the schedule of a train.

        Args:
            train (WebElement): Train element.

        Returns:
            Mapping[str, Tuple[int, int]]: Dictionary of stops, where keys are each station as
                adif ids and values are a tuple of relative times in minutes.
        """
        # Switch to the schedule window
        train_info = train.find_element(
            By.CSS_SELECTOR,
            '.irf-travellers-table__tbody-lnk.irf-travellers-table__tbody-lnk--icon-left'
        )
        train_info.click()
        wait = WebDriverWait(self.driver, DEFAULT_PATIENCE)
        wait.until(lambda driver: len(driver.window_handles) > 1)
        self.driver.switch_to.window(self.driver.window_handles[1])

        # Build the schedule dictionary
        schedule_table = self.driver.find_element(By.CLASS_NAME, 'irf-renfe-travel__container-table')
        train_stops = schedule_table.find_elements(By.CSS_SELECTOR, '.irf-renfe-travel__td.txt_gral')
        schedule = self._get_relative_schedule(train_stops)

        # Close the schedule window and switch back to the main window
        self.driver.close()
        wait.until(lambda driver: len(driver.window_handles) == 1)
        self.driver.switch_to.window(self.driver.window_handles[0])
        return schedule

    def _get_trips_departure(self, train: WebElement, date: datetime.date) -> datetime.datetime:
        """
        Returns the departure time of a train.

        Args:
            train (WebElement): Train element.
            date (datetime.date): Date of the trip.

        Returns:
            datetime.datetime: Departure time of the train.
        """
        train_info = train.find_elements(By.CSS_SELECTOR, '.txt_borde1.irf-travellers-table__td')
        departure_minutes = time_to_minutes(train_info[1].text)
        departure = datetime.datetime(year=date.year, month=date.month, day=date.day) + datetime.timedelta(minutes=departure_minutes)
        return departure

    def _get_trips_duration(self, train: WebElement) -> int:
        """
        Returns the departure time of a train.

        Args:
            train (WebElement): Train element.

        Returns:
            int: Duration of the service in minutes.
        """
        train_info = train.find_elements(By.CSS_SELECTOR, '.txt_borde1.irf-travellers-table__td')
        duration_minutes = time_str_to_minutes(train_info[3].text)
        return duration_minutes

    def _is_allowed_train_type(self, train_type: str) -> bool:
        """
        Checks if the train type is allowed.

        Args:
            train_type (str): Train type.

        Returns:
            bool: True if the train type is allowed, False otherwise.
        """
        return train_type in ' '.join(self.allowed_train_types)

    def _is_transfer_train(self, train: WebElement) -> bool:
        """
        Checks if the train is a transfer train.

        Args:
            train (WebElement): Train element.

        Returns:
            bool: True if the train is a transfer train, False otherwise.
        """
        try:
            train.find_element(By.CLASS_NAME, 'enlace-tren')
            return True
        except NoSuchElementException:
            return False

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
        try:
            div = WebDriverWait(self.driver, patience).until(EC.visibility_of_element_located((find_by, find_value)))
        except TimeoutException:
            return None
        return div

    def get_value_from_stations(self, search_column: str, value: str, objective_column: str) -> str:
        """
        Gets the value of a column given the value of another column.

        Args:
            search_column (str): Column to search for the value.
            value (str): Value to search for.
            objective_column (str): Column to get the value from.

        Returns:
            str: Value of the objective column.
        """
        return self.stations_df[self.stations_df[search_column] == value][objective_column].values[0]

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

        Returns:
            pd.DataFrame: DataFrame with the scraped data.
        """
        url = self._get_renfe_prices_url(origin_id, destination_id, date)
        prices = self._request_url(url=url, find_by=By.ID, find_value='listaTrenesTBodyIda')
        if not prices:
            return pd.DataFrame()
        df_prices = self._get_df_prices(prices, origin_id, destination_id, date)
        return df_prices

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
            station_id = self.get_value_from_stations(search_column='RENFE_ID', value=station_id, objective_column='ADIF_ID')
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
            return pd.DataFrame()
        df_trips = self._get_df_trips(trips, date)
        return df_trips


class RenfeScraper:
    """
    Renfe Scraping class.

    Attributes:
        stations_df (pd.DataFrame): A pandas DataFrame to parse stations data.
        driver (DriverManager): A DriverManager object to manage the browser.
        available_stations (List[str]): A list with the available stations in the Renfe website.
    """

    def __init__(
        self,
        stations_csv_path: str = RENFE_STATIONS_CSV,
        menu_url: str = MAIN_MENU_URL,
        allowed_train_types: List[str] = ALLOWED_RENFE_SERVICES
    ) -> None:
        """
        Initialize a RenfeScraper object.

        Args:
            stations_csv_path (str): Path to the CSV file with the available stations.
            menu_url (str): URL of the Renfe main menu.
            allowed_train_types (List[str]): List of allowed train types to scrape.
        """
        self.stations_df = pd.read_csv(stations_csv_path, sep=';', dtype={'ADIF_ID': str, 'RENFE_ID': str})
        self.driver = DriverManager(
            stations_df=self.stations_df,
            allowed_train_types=allowed_train_types
        )
        self.available_stations = self.driver.scrape_stations(menu_url)

    def _get_df_stops(self, df_trips: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with the stops' information.
        
        Args:
            df_trips (pd.DataFrame): DataFrame with the trips information.
        
        Returns:
            pd.DataFrame: DataFrame with the stops' information.
        """
        # Create a dictionary with the service_id as key and the schedule as value
        schedules_dict = dict(zip(df_trips.service_id, df_trips.schedule))

        # Create a list of dictionaries with the stop information
        rows = []
        for service_id, schedule in schedules_dict.items():
            for stop_id, (arrival, departure) in schedule.items():
                rows.append({'service_id': service_id, 'stop_id': stop_id, 'arrival': arrival, 'departure': departure})
        return pd.DataFrame(rows)

    def _od_pairs_from_trip(self, trip: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        """
        Returns a set of origin-destination pairs from a trip.

        Args:
            trip (Tuple[str, ...]): Tuple with the trip information.

        Returns:
            Set[Tuple(str, str)]: Set of origin-destination pairs.
        """
        od_pairs = set()
        for i in range(len(trip)):
            for j in range(i + 1, len(trip)):
                origin = self.driver.get_value_from_stations(search_column='ADIF_ID', value=trip[i], objective_column='RENFE_ID')
                destination = self.driver.get_value_from_stations(search_column='ADIF_ID', value=trip[j], objective_column='RENFE_ID')
                od_pairs.add((origin, destination))
        return od_pairs

    def _save_df(
        self,
        df: pd.DataFrame,
        df_name: str,
        origin_id: str,
        destination_id: str,
        init_date: datetime.date,
        end_date: datetime.date,
        save_path: str
    ) -> None:
        """
        Saves the dataframe to a CSV file.

        Args:
            df_stops (pd.DataFrame): Dataframe to save.
            df_name (str): Name of the dataframe.
            origin_id (str): Renfe id of the origin station.
            destination_id (str): Renfe id of the destination station.
            init_date (datetime.date): initial date of the trip.
            end_date (datetime.date): end date of the trip.
            save_path (str): Path to save the CSV file.
        """
        os.makedirs(f'{save_path}/{df_name}/', exist_ok=True)
        df.to_csv(
            f'{save_path}/{df_name}/{df_name}_{origin_id}_{destination_id}_{init_date}_{end_date}.csv',
            index=False
        )

    def scrape(
        self,
        origin: str,
        destination: str,
        init_date: datetime.date = None,
        range_days: int = 1,
        all_pairs: bool = False,
        save_path: str = SAVE_PATH
    ) -> None:
        """
        Scrapes the Renfe website for the trips and prices of services between two stations.

        Args:
            origin (str): Adif station id of the origin station.
            destination (str): Adif station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            all_pairs (bool): If True, scrape prices for all origin-destination pairs in the trips dataframe.
            save_path (str): Path to save the CSV files.
        """
        # Assert that the origin and destination stations are in the list of stations operated by Renfe
        pair_of_stations_in_csv = all(trip in self.available_stations.keys() for trip in (origin, destination))
        assert pair_of_stations_in_csv, 'Invalid origin or destination'

        # If no initial date is provided, use today's date
        if not init_date:
            init_date = datetime.date.today()

        # Convert Adif station ids to Renfe station ids
        origin_id = self.driver.get_value_from_stations(search_column='ADIF_ID', value=origin, objective_column='RENFE_ID')
        destination_id = self.driver.get_value_from_stations(search_column='ADIF_ID', value=destination, objective_column='RENFE_ID')

        # Scrape trips
        df_trips = self.scrape_trips(
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            range_days=range_days,
            save_path=save_path
        )
        end_date = init_date + datetime.timedelta(days=range_days)
        if df_trips.empty:
            logger.warning(f'No trips found between {origin_id} and {destination_id} from {init_date} to {init_date + datetime.timedelta(days=range_days)}')
        else:
            unique_trips = df_trips.groupby('service_id').size()
            logger.success(f'Scraped {len(unique_trips)} trips between {origin_id} and {destination_id} from {init_date} to {end_date}')
            logger.info(f'First five trips:\n{df_trips.head()}')

        # Scrape prices
        df_prices = self.scrape_prices(
            origin_id=origin_id,
            destination_id=destination_id,
            init_date=init_date,
            range_days=range_days,
            df_trips=df_trips if all_pairs else None,
            save_path=save_path
        )
        if df_prices.empty:
            logger.warning(f'No prices found between {origin_id} and {destination_id} from {init_date} to {end_date}')
        else:
            logger.success(f'Scraped {len(df_prices)} services between {origin_id} and {destination_id} from {init_date} to {end_date}')
            logger.info(f'First five services:\n{df_prices.head()}')

    def scrape_prices(
        self,
        origin_id: str,
        destination_id: str,
        init_date: datetime.date = None,
        range_days: int = 1,
        df_trips: pd.DataFrame = None,
        save_path: str = SAVE_PATH,
        max_attempts: int = MAX_ATTEMPTS
    ) -> pd.DataFrame:
        """
        Scrapes the Renfe website for the prices of services between two stations.

        Args:
            origin_id (str): Renfe station id of the origin station.
            destination_id (str): Renfe station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            df_trips (pd.DataFrame): DataFrame containing the scraped trips. If None, it will scrape only the prices
                between the origin and destination stations.
            save_path (str): Path to save the scraped data.
            max_attempts (int): Maximum number of attempts to scrape prices.
        
        Returns:
            pd.DataFrame: DataFrame containing the scraped prices.
        """
        if df_trips is not None:
            # Get set of trips from the trips dataframe
            trips = set(df_trips.groupby('service_id')['stop_id'].apply(tuple))

            # Get the origin-destination pairs from the trips dataframe
            od_pairs = {pair for trip in trips for pair in self._od_pairs_from_trip(trip)}
        else:
            od_pairs = {(origin_id, destination_id)}

        end_date = init_date + datetime.timedelta(days=range_days)
        df_prices = pd.DataFrame()
        for origin, destination in od_pairs:
            date = init_date
            for _ in range(range_days):
                logger.info(f'Scraping prices for {origin} - {destination} on {date}')
                attempt = 0
                while attempt < max_attempts:
                    new_df_prices = self.driver.scrape_prices(origin_id=origin, destination_id=destination, date=date)
                    if new_df_prices.empty:
                        logger.warning(f'No prices found for {origin} - {destination} on {date}. Attempt {attempt}/{max_attempts}.')
                        attempt += 1
                    else:
                        df_prices = pd.concat([df_prices, new_df_prices], ignore_index=True)
                        break
                if attempt == max_attempts:
                    logger.error(f'Failed to scrape prices for {origin} - {destination} on {date} after {max_attempts} attempts.')
                date += datetime.timedelta(days=1)

        self._save_df(
            df=df_prices,
            df_name='prices',
            origin_id=self.driver.get_value_from_stations(search_column='RENFE_ID', value=origin_id, objective_column='ADIF_ID'),
            destination_id=self.driver.get_value_from_stations(search_column='RENFE_ID', value=destination_id, objective_column='ADIF_ID'),
            init_date=init_date,
            end_date=end_date,
            save_path=save_path
        )
        return df_prices

    def scrape_trips(
        self,
        origin_id: str,
        destination_id: str,
        init_date: datetime.date,
        range_days: int = 1,
        save_path: str = SAVE_PATH,
        max_attempts: int = MAX_ATTEMPTS
    ) -> pd.DataFrame:
        """
        Scrapes the Renfe website for the trips between two stations.

        Args:
            origin_id (str): Renfe station id of the origin station.
            destination_id (str): Renfe station id of the destination station.
            init_date (datetime.date): Initial date to start scraping.
            range_days (int): Number of days to scrape.
            save_path (str): Path to save the scraped data.
            max_attempts (int): Maximum number of attempts to scrape trips.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrame containing the scraped trips and scraped stops.
        """
        # Scrape trips from the schedules table
        date = init_date
        end_date = init_date + datetime.timedelta(days=range_days)
        df_trips = pd.DataFrame()
        for _ in range(range_days):
            logger.info(f'Scraping trips for {origin_id} - {destination_id} on {date}')
            attempt = 0
            while attempt < max_attempts:
                new_df_trips = self.driver.scrape_trips(origin_id=origin_id, destination_id=destination_id, date=date)
                if new_df_trips.empty:
                    logger.warning(f'No trips found for {origin_id} - {destination_id} on {date}. Skipping...')
                    attempt += 1
                else:
                    df_trips = pd.concat([df_trips, new_df_trips], ignore_index=True)
                    break
            if attempt == max_attempts:
                logger.error(f'Failed to scrape trips for {origin_id} - {destination_id} on {date} after {max_attempts} attempts.')
            date += datetime.timedelta(days=1)

        if df_trips.empty:
            return pd.DataFrame()

        df_stops = self._get_df_stops(df_trips)
        self._save_df(
            df=df_stops,
            df_name='stopTimes',
            origin_id=self.driver.get_value_from_stations(search_column='RENFE_ID', value=origin_id, objective_column='ADIF_ID'),
            destination_id=self.driver.get_value_from_stations(search_column='RENFE_ID', value=destination_id, objective_column='ADIF_ID'),
            init_date=init_date,
            end_date=end_date,
            save_path=save_path
        )
        return df_stops
