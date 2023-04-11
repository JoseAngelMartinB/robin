""" Utility functions for scraping module"""

import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from typing import Union


def is_number(s: str) -> bool:
    """
    Function to check if a string is a number

    Args:
        s (str): String to check

    Returns:
        bool: True if string is a number, False otherwise
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def time_to_minutes(time: str) -> int:
    """
    Function receives "time", a string with time formatted as "HH.MM" and returns the number of minutes

    Args:
        time (str): String with time formatted as "HH.MM"

    Returns:
        int: Number of minutes
    """
    try:
        h, m = time.split(".")
    except ValueError:
        return 0

    return int(h) * 60 + int(m)


def format_duration(x: str) -> int:
    """
    Function receives "x", a string with time formatted as "2 h. 30 m." and returns a string H:M

    Args:
        x (str): String with time formatted as "2 h. 30 m."

    Returns:
        str: String with duration in minutes
    """
    tuple_hour_min = tuple(filter(lambda t: is_number(t), x.split(" ")))
    if len(tuple_hour_min) == 1:
        m = tuple_hour_min[0]
        return int(m)
    h, m = tuple_hour_min
    return int(h) * 60 + int(m)


def remove_blanks(s: str, replace_by: str = '') -> str:
    """
    Removes blank spaces (tabs, newlines, etc.) from a string.

    Args:
        s (str): string to remove blank spaces from
        replace_by (str): string to replace the blank spaces with

    Returns:
        str: string without blank spaces
    """
    return re.sub(r'\s+', replace_by, s)


class ChromeDriverManager:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--disable-extensions")
        # self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--headless")  # Don't open browser window
        self.driver = webdriver.Chrome(options=self.chrome_options)

    def request_price(self, url: str, patience: int = 25) -> Union[str, int]:
        """
        Function to request a URL with selenium

        Args:
            url (str): URL to request
            patience (int, optional): Patience in seconds. Defaults to 12.

        Return:
            str: HTML code of the page
            returns bool False if page fails to load
        """
        self.driver.get(url)

        try:
            WebDriverWait(self.driver, patience).until(EC.presence_of_element_located((By.CLASS_NAME, 'trayectoRow')))
        except TimeoutException:
            return False

        return self.driver.page_source
