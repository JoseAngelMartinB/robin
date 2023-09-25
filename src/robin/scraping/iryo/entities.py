"""Entities for Iryo scraping."""

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from typing import List

LANDING_URL = 'https://iryo.eu/es/home'
SAVE_PATH = 'data/iryo'


class IryoScraper:

    def __init__(self, main_page_url: str = LANDING_URL) -> None:
        """
        Initializes the IryoScraper object.

        Args:
            main_page_url (str): URL to scrape the data from.
        """
        self.driver = self._get_web_driver()
        self.stations = self.get_stations(main_page_url)

    def _get_web_driver(self):
        driver_options = Options()
        driver_options.add_argument('--disable-extensions')
        driver_options.add_argument("--enable-javascript")
        # driver_options.add_argument('--disable-gpu')
        # driver_options.add_argument('--headless')  # Don't open browser window
        chrome_driver = webdriver.Chrome(options=driver_options)
        chrome_driver.set_window_size(1900, 1200)
        chrome_driver.maximize_window()
        return chrome_driver

    def get_stations(self, url: str = LANDING_URL) -> List[str]:
        """
        Scrapes the data from the Renfe website.

        Args:
            url (str): URL to scrape the data from.
        """
        self.driver.get(url)
        menu_dropdown_class = 'ilsa-dropdown__menu ilsa-dropdown__menu--bottom ilsa-dropdown__menu--shadow-01'

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "ilsa-main-search__fields"))
            )
        except TimeoutException:
            print("Loading took too much time!")

        html_str = self.driver.page_source
        soup = BeautifulSoup(html_str)

        print("Soup: ", soup)
        menu_list = soup.find('div', {'class': menu_dropdown_class})
        print(menu_list)
        stations = menu_list.find_all('div', {'class': 'ilsa-menu__item'})

        return [station.text for station in stations]


if __name__ == '__main__':
    iryo_scraper = IryoScraper()
    print(iryo_scraper.stations)
