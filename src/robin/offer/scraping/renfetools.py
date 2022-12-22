from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import requests

import pandas as pd
import datetime
import re


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_stations(soup):
    menu = soup.find('div', {'class': 'irf-search-shedule__container-ipt'})
    options = menu.find_all('option')

    return {opt["value"]: " ".join(filter(lambda x: x != "", opt.text.split(" ")))for opt in options}


def get_date(soup, selenium=False):
    table = soup.find('div', {'class': 'irf-travellers-table__container-time'})

    p = table.find_all('span', {'class': 'irf-travellers-table__txt irf-travellers-table__txt--bold'})

    months = {'Enero': 1,
              'Febrero': 2,
              'Marzo': 3,
              'Abril': 4,
              'Mayo': 5,
              'Junio': 6,
              'Julio': 7,
              'Agosto': 8,
              'Septiembre': 9,
              'Octubre': 10,
              'Noviembre': 11,
              'Diciembre': 12}

    date = []

    for i in p:
        s = i.find('script')

        if s is not None:
            s = s.text.split('\'')[1]

            if s in months.keys():
                date.append(str(months[s]))

            # TODO: Merge both solutions into one that works for both cases
            if selenium:
                # Selenium
                num = tuple(filter(lambda x: x != '', re.sub(r'\s+', " ", i.text).split(" ")))[1]

                if is_number(num):
                    date.append(num)
            else:
                # Requests
                num = tuple(filter(lambda x: x != '', re.sub(r'\s+', " ", i.text).split(" ")))[0]

                if is_number(i.text):
                    x = re.sub(r'\s+', '', i.text)
                    date.append(x)

    date = '-'.join(date)

    date = datetime.datetime.strptime(date, '%d-%m-%Y').date()
    return date


def get_stops(url):
    """
    Returns dictionary of stops from url with stops information from renfe
    :param url: url with stops information from renfe
    :return: dictionary of stops, where keys are each station and values are a tuple with (arrival, departure) times
    """
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")

    # sdriver = webdriver.Chrome(options=chrome_options)
    # sdriver.get(url)

    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    # soup = BeautifulSoup(sdriver.page_source, 'html.parser')

    # sdriver.close()

    table = soup.find('table', {'class': 'irf-renfe-travel__table cabecera_tabla'})

    stops = {}
    for row in table.find_all('tr'):
        aux = row.find_all('td')

        if aux:
            # Define blacklist to remove words from stop name
            blacklist = ["", "PTA", "PUERTA", "CAMP", "DE"]

            # Remove non-alphanumeric characters
            raw_name = re.sub(r'[^a-zA-Z0-9 -]', '', aux[0].text)

            # Split raw_name
            raw_words = re.split(r'\W+', raw_name)

            # Remove blacklist words from stop name and get first word of each stop name
            station = tuple(filter(lambda w: w not in blacklist and len(w) > 1, raw_words))[0]

            departure_time = re.sub(r'\s+', "", aux[1].text).replace(".", ":")

            arrival_time = re.sub(r'\s+', "", aux[2].text).replace(".", ":")

            stops[station.upper()] = (departure_time, arrival_time)

    # Get first and last keys of stops dictionary
    first_key = list(stops.keys())[0]
    last_key = list(stops.keys())[-1]

    stops[first_key] = (stops[first_key][1], stops[first_key][1])
    stops[last_key] = (stops[last_key][0], stops[last_key][0])

    return stops


def get_table(soup, url):
    main_table = soup.find('div', {'class': 'irf-travellers-table__container-table'})
    root = "https://horarios.renfe.com/HIRRenfeWeb/"

    table = []
    for tr in main_table.find_all("tr", {'class': "odd irf-travellers-table__tr"}):
        cols = tr.find_all("td", {'class': "txt_borde1 irf-travellers-table__td"})

        if not cols or len(cols) < 6:
            continue

        train_number, train_type = tuple(filter(lambda x: x != "", re.sub(r"\s+", " ", cols[0].text).split(" ")))
        train_id = {train_type : train_number}

        stops_link = cols[0].find('a')["href"]

        # Get javascript link
        js_link = str(stops_link).replace("\n", "").replace("\t", "").replace(" ", "%20")

        pattern = r'\("(.+)"\)'
        match = re.search(pattern, js_link)

        if match:
            js_link = match.group(1)

        stops = get_stops(root + js_link)

        """
        p = cols[4].find("a")
        p_link = p["href"]

        links = cols[4].find_all("a")

        for l in links:
            if "javascript:comprarVOL" in l["href"]:
                p_link = l["href"]
                get_prices(p_link, url)
                break
        # Stop execution
        """
        p = cols[4].find("div")

        i = re.sub(r'\s+', '', p.text)
        raw_prices = re.sub(r'PrecioInternet|:', '', i).replace(",", ".")

        p = re.findall(r'[a-zA-Z]+|[0-9.]+', raw_prices)

        assert len(p) % 2 == 0, "Error parsing prices"

        prices = {p[i]: float(p[i + 1]) for i in range(0, len(p), 2)}

        train = (train_id, stops, cols[1].text, cols[3].text, prices)

        # Assert non empty values
        assert all(v for v in train), "Error parsing train"

        table.append(train)

    return table


def format_time(x):
    """ Function receives "x", a string with time formatted as "2 h. 30 m." and returns a timedelta object """
    h, m = filter(lambda t: is_number(t), x.split(" "))
    return f'{h}:{m}'


def to_timedelta(x):
    """ Function receives "x", a string with time formatted as "2:30" and returns a timedelta object """
    h, m = x.split(":")
    return datetime.timedelta(hours=int(h), minutes=int(m))


def to_dataframe(s, d, url):
    table = get_table(s, url)

    dfs = pd.DataFrame(table, columns=['Train', 'Stops', 'Departure', 'Duration', 'Price'])
    dfs = dfs[dfs["Train"].apply(lambda x: "AVE" in x)].reset_index(drop=True)

    dfs['Duration'] = dfs['Duration'].apply(lambda x: format_time(x))
    dfs['Departure'] = dfs['Departure'].apply(lambda x: datetime.datetime.strptime(str(d) + "-" + x, '%Y-%m-%d-%H.%M'))
    dfs['Arrival'] = dfs['Departure'] + dfs['Duration'].apply(lambda x: to_timedelta(x))

    return dfs


def get_prices(org, dest, date):
    root = "https://venta.renfe.com/vol/"
    search_url = f"buscarTren.do?tipoBusqueda=autocomplete&currenLocation=menuBusqueda&vengoderenfecom=SI&cdgoOrigen={org}&cdgoDestino={dest}&idiomaBusqueda=“s”&FechaIdaSel={date}&_fechaIdaVisual={date}&adultos_=1&ninos_=0&ninosMenores=0&numJoven=0&numDorada=0&codPromocional="








