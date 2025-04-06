import datetime

from robin.scraping.renfe.entities import RenfeScraper

scraper = RenfeScraper()

origin = '60000'
destination = '71801'
date = datetime.date(day=6, month=9, year=2023)
scraper.scrape(origin=origin, destination=destination, init_date=date)
