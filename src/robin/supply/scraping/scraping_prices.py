from renfetools import *

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

import numpy as np


def renfe_scraping_prices(origin_id, destination_id, date, range_days):
    print("Scraping prices...")

    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")

    init_date = date
    for i in range(range_days):
        # Parse today date to string with format %d-%m-%Y
        date_str = date.strftime("%d-%m-%Y")

        # Renfe search for tickets from origin to destination on date
        # Schedules table is loaded dynamically with javascript, so we need to use selenium!!!

        root = "https://venta.renfe.com/vol/"
        search_url = f"buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional="

        url = root + search_url
        print("Search url: ", url)
        print("Date: ", date)

        driver = webdriver.Chrome(options=chrome_options)

        driver.get(url)

        delay = 12  # seconds
        try:
            WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'trayectoRow')))
            print("Page is ready!")
        except TimeoutException:
            print("Loading took too much time!")
            date += datetime.timedelta(days=1)
            continue

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        table = soup.find("div", {"class": "tab-content"})

        count = 0
        trains = []
        for row in table.find_all("tr"):
            if isTrain(row):
                # trip_id, train_type, origin_id, destination_id, date_service, departure_time
                trains.append(get_train(row, origin_id, destination_id, date_str))
            else:
                continue
            count += 1

        # for train in trains:
        #     print(train)

        # print("Total trains: ", count)
        driver.close()

        df1 = pd.DataFrame.from_records(trains, columns=['trip_id', 'train_type', 'arrival', 'departure', 'duration', 'prices'])
        df1["service_id"] = df1.apply(lambda x: x["trip_id"] + "_" + x["departure"].strftime("%d-%m-%Y-%H.%M"), axis=1)

        if not i:
            df = df1
        else:
            df = pd.concat([df, df1])

        # Sum one day to date
        date += datetime.timedelta(days=1)

    df = df.reset_index(drop=True)
    df = df[['service_id', 'trip_id', 'train_type', 'arrival', 'departure', 'duration', 'prices']]

    print(df.columns)
    print(df.iloc[-1])

    end_date = date - datetime.timedelta(days=1)

    # Save numpy file with prices
    prices = dict(zip(df.service_id, df.prices))

    list_prices = []
    for k, v in prices.items():
        list_prices.append([k, *list(v.values())])

    df_prices = pd.DataFrame(list_prices, columns=['service_id', '0', '1', '2'])

    # np.save(f"datasets/prices/prices_{origin_id}_{destination_id}_{init_date}_{end_date}.npy", prices)
    df_prices.to_csv(f"datasets/prices/prices_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False)


if __name__ == "__main__":
    date = datetime.date.today()
    range_days = 1
    origin_id = 'MADRI'
    destination_id = 'BARCE'
    renfe_scraping_prices(origin_id, destination_id, date, range_days)