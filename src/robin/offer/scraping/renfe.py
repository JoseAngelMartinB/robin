from bs4 import BeautifulSoup
import requests
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

    return {" ".join(filter(lambda x: x != "", opt.text.split(" "))): opt["value"] for opt in options}


def get_date(soup):
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
            if is_number(i.text):
                x = re.sub(r'\s+', '', i.text)
                date.append(x)

    date = '-'.join(date)

    date = datetime.datetime.strptime(date, '%d-%m-%Y').date()
    return date


def get_stops(url):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    table = soup.find('table', {'class': 'irf-renfe-travel__table cabecera_tabla'})

    stops = {}
    for row in table.find_all('tr'):
        aux = row.find_all('td')

        if aux:
            station = " ".join(
                filter(lambda x: x != "", map(lambda x: re.sub(r'\s+', "", str(x)), aux[0].text.split(" "))))
            departure_time = re.sub(r'\s+', "", aux[1].text)
            arrival_time = re.sub(r'\s+', "", aux[2].text)
            stops[station] = (departure_time, arrival_time)

    return stops


def get_table(soup):
    main_table = soup.find('div', {'class': 'irf-travellers-table__container-table'})
    root = "https://horarios.renfe.com/HIRRenfeWeb/"

    table = []
    for tr in main_table.find_all("tr", {'class': "odd irf-travellers-table__tr"}):
        cols = tr.find_all("td", {'class': "txt_borde1 irf-travellers-table__td"})

        if not cols:
            continue

        train = cols[0].text

        try:
            stops_link = cols[0].find('a')["href"]
        except TypeError:
            continue

        js_link = str(stops_link).replace("\n", "").replace("\t", "").replace(" ", "%20")

        pattern = r'\("(.+)"\)'
        match = re.search(pattern, js_link)

        if match:
            js_link = match.group(1)
            train_name = " ".join(filter(lambda x: x != "", re.sub(r"\s+", " ", train).split(" ")))

        stops = get_stops(root + js_link)

        p = cols[4].find("div")
        i = re.sub(r'\s+', '', p.text)
        raw_prices = re.sub(r'PrecioInternet|:', '', i).replace(",", ".")

        p = re.findall(r'[a-zA-Z]+|[0-9.]+', raw_prices)

        assert len(p) % 2 == 0, "Error parsing prices"

        prices = {p[i]: float(p[i + 1]) for i in range(0, len(p), 2)}

        table.append((train_name, stops, cols[1].text, cols[3].text, prices))

    return table







