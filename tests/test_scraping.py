import datetime

from src.robin.scraping.renfe.entities import RenfeScraper

scraper = RenfeScraper()

origin = '60000'
destination = '71801'
date = datetime.date(day=23, month=6, year=2023)
scraper.scrape(origin=origin, destination=destination, init_date=date)
