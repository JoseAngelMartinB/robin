from renfetools import *

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

def renfe_scraping_prices(origin_id: str, destination_id: str, date: datetime.date, range_days: int):
    """
    Scraping prices

    :param
        origin_id: Renfe station id
        destination_id: Renfe station id
        date: datetime.date object
        range_days: Range of days to scrape

    :returns: None

    Generates one csv files with the three different prices for each service_id

    prices: service_id,0,1,2
    0, 1 and 2 are the prices for the three different seat types offered by Renfe
    """
    print("Scraping prices...")

    chrome_options = Options()
    chrome_options.add_argument("--disable-extensions")
    # chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--headless")  # Don't open browser window

    # Initialize selenium web driver (will use local chrome driver) Requires chromedriver to be installed
    driver = webdriver.Chrome(options=chrome_options)

    init_date = date # Save initial date
    for i in range(range_days):
        # Parse date to string with format %d-%m-%Y to generate the URL
        date_str = date.strftime("%d-%m-%Y")

        # URL Renfe search for tickets from origin i to destination j on date d
        # Schedules table is loaded dynamically with javascript, so we need to use selenium and wait until it is loaded

        root = "https://venta.renfe.com/vol/"
        search_url = f"buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={origin_id}&cdgoDestino={destination_id}&idiomaBusqueda=s&FechaIdaSel={date_str}&_fechaIdaVisual={date_str}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional="

        url = root + search_url
        print("Search url: ", url)
        print("Date: ", date)

        driver.get(url)

        delay = 12  # Patience in seconds
        try:
            WebDriverWait(driver, delay).until(EC.presence_of_element_located((By.CLASS_NAME, 'trayectoRow')))
        except TimeoutException:
            date += datetime.timedelta(days=1)
            continue  # If page fails to load, skip to next day

        soup = BeautifulSoup(driver.page_source, 'html.parser')

        table = soup.find("div", {"class": "tab-content"})

        count = 0
        trains = []
        for row in table.find_all("tr"):
            if isTrain(row):
                trains.append(get_train(row, origin_id, destination_id, date_str))
            else:
                continue
            count += 1

        df1 = pd.DataFrame.from_records(trains, columns=['trip_id', 'train_type', 'departure', 'arrival', 'duration', 'prices'])
        df1["service_id"] = df1.apply(lambda x: x["trip_id"] + "_" + x["departure"].strftime("%d-%m-%Y-%H.%M"), axis=1)

        if not i:  # If first iteration, create csv file
            df = df1
        else:
            df = pd.concat([df, df1])

        # Sum one day to date
        date += datetime.timedelta(days=1)

    driver.close()  # Close web driver

    df = df.reset_index(drop=True)

    # Reorder columns
    df = df[['service_id', 'trip_id', 'train_type', 'departure', 'arrival', 'duration', 'prices']]

    print(df.columns)
    print(df.iloc[-1])

    # Get end date
    end_date = date - datetime.timedelta(days=1)

    # Save numpy file with prices
    prices = dict(zip(df.service_id, df.prices))

    # Build dataframe with prices
    list_prices = []
    for k, v in prices.items():
        list_prices.append([k, *list(v.values())])

    df_prices = pd.DataFrame(list_prices, columns=['service_id', '0', '1', '2'])

    df_prices.to_csv(f"../../datasets/scraping/renfe/prices/prices_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False)


if __name__ == "__main__":
    date = datetime.date.today()
    date += datetime.timedelta(days=1)  # Tomorrow
    range_days = 1
    origin_id = 'MADRI'
    destination_id = 'BARCE'
    renfe_scraping_prices(origin_id, destination_id, date, range_days)