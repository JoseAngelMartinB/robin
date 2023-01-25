from bs4 import BeautifulSoup
import requests
import pandas as pd
import datetime
import re


def isTrain(r):
    try:
        r['cdgotren']
        return True
    except KeyError:
        return False


def get_prices(pcols):
    prices = {}
    for i, col in enumerate(pcols):
        cb = col.find_all("button")

        if cb:
            for p in cb:
                pd = p.find_all("div")
                if pd:
                    price = pd[1].text.split(" ")[0].replace(",", ".")
                else:
                    price = float("NaN")
        else:
            # price = re.sub(r'\s+', '', col.text)
            return {j: float("NaN") for j in range(3)}

        prices[i] = price

    return prices


def get_train(r, origin_id, destination_id, date):
    train_id = r['cdgotren']
    cols = r.find_all("td")

    c1 = cols[1].find_all("div")
    departure, duration = c1[0].text, c1[1].text

    c3 = cols[3].find_all("div")
    arrival, train_type = c3[0].text, c3[2].text

    remove_blanks = lambda s: re.sub(r'\s+', '', s)

    departure = remove_blanks(departure)
    departure = datetime.datetime.strptime(str(date) + "-" + departure, '%d-%m-%Y-%H.%M')

    duration = re.sub(r'\s+', ' ', duration)
    duration_str = format_time(duration)
    duration = to_timedelta(duration_str)

    arrival = departure + duration

    train_type = remove_blanks(train_type)

    prices = get_prices(cols[4:-1])

    return [train_id, train_type, departure, arrival, duration_str, prices]


# Old scraping version using requests from here
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

    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')

    table = soup.find('table', {'class': 'irf-renfe-travel__table cabecera_tabla'})

    renfe_stations = pd.read_csv('../../datasets/scraping/renfe/renfe_stations.csv', sep=',', dtype={'stop_id': str})

    gtfs_names = renfe_stations['stop_name'].values.tolist()
    gtfs_names = list(map(lambda s: s.lower(), gtfs_names))
    gtfs_names = list(map(lambda s: re.sub(r'[-/]', ' ', s), gtfs_names))

    stops = {}
    for row in table.find_all('tr'):
        aux = row.find_all('td')

        if aux:
            # Define blacklist to remove words from stop name
            # blacklist = ["", "PTA", "PUERTA", "CAMP", "DE"]

            # Remove non-alphanumeric characters
            raw_name = re.sub(r'[^a-zA-Z0-9 -]', '', aux[0].text)

            # Split raw_name
            raw_words = [w for w in re.split(r'\W+', raw_name) if w]
            name = " ".join(raw_words)
            name = name.lower()

            best_match = "Unknown"
            bml = 0
            for gn in gtfs_names:
                gnl = gn.split(" ")

                if name == gn:
                    best_match = gn
                    break
                elif sum([True for w in name.split(" ") if w in gnl]) > bml:
                    best_match = gn

            try:
                i = gtfs_names.index(best_match)
                station_id = renfe_stations.iloc[i]['stop_id']
            except ValueError:
                station_id = "00000"

            # Remove blacklist words from stop name and get first word of each stop name
            # station = tuple(filter(lambda w: w not in blacklist and len(w) > 1, raw_words))[0]

            departure_time = re.sub(r'\s+', "", aux[1].text).replace(".", ":")

            arrival_time = re.sub(r'\s+', "", aux[2].text).replace(".", ":")
            # station.upper()
            stops[station_id] = (departure_time, arrival_time)

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

        train_id, train_type = tuple(filter(lambda x: x != "", re.sub(r"\s+", " ", cols[0].text).split(" ")))

        stops_link = cols[0].find('a')["href"]

        # Get javascript link
        js_link = str(stops_link).replace("\n", "").replace("\t", "").replace(" ", "%20")

        pattern = r'\("(.+)"\)'
        match = re.search(pattern, js_link)

        if match:
            js_link = match.group(1)

        stops = get_stops(root + js_link)

        p = cols[4].find("div")

        i = re.sub(r'\s+', '', p.text)
        raw_prices = re.sub(r'PrecioInternet|:', '', i).replace(",", ".")

        p = re.findall(r'[a-zA-Z]+|[0-9.]+', raw_prices)

        assert len(p) % 2 == 0, "Error parsing prices"

        prices = {p[i]: float(p[i + 1]) for i in range(0, len(p), 2)}

        train = (train_id, train_type, stops, cols[1].text, cols[3].text, prices)

        # Assert non empty values
        assert all(v for v in train), "Error parsing train"

        table.append(train)

    return table


def format_time(x):
    """ Function receives "x", a string with time formatted as "2 h. 30 m." and returns a string H:M """
    h, m = filter(lambda t: is_number(t), x.split(" "))
    return f'{h}:{m}'


def to_timedelta(x):
    """ Function receives "x", a string with time formatted as "2:30" and returns a timedelta object """
    h, m = x.split(":")
    return datetime.timedelta(hours=int(h), minutes=int(m))


def to_dataframe(s, d, url):
    table = get_table(s, url)

    dfs = pd.DataFrame(table, columns=['trip_id', 'train_type', 'stops', 'departure', 'duration', 'price'])

    # Filter only AVE trains
    dfs = dfs[dfs["train_type"].apply(lambda x: "AVE" in x)].reset_index(drop=True)

    # Get service id using data from "trip_id" and "departure" columns
    dfs['duration'] = dfs['duration'].apply(lambda x: format_time(x))
    dfs['departure'] = dfs['departure'].apply(lambda x: datetime.datetime.strptime(str(d) + "-" + x, '%Y-%m-%d-%H.%M'))
    dfs['arrival'] = dfs['departure'] + dfs['duration'].apply(lambda x: to_timedelta(x))

    dfs["service_id"] = dfs.apply(lambda x: x["trip_id"] + "_" + x["departure"].strftime("%d-%m-%Y-%H.%M"), axis=1)

    return dfs

