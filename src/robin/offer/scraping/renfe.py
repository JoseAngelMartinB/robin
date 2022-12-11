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


def get_prices(soup):
    table = soup.find('div', {'class': 'irf-travellers-table__container-table'})

    prices = []
    for tr in table.find_all("tr"):
        trs = tr.find_all("td")

        for td in trs:

            t = td.find_all("div")
            for d in t:
                i = re.sub(r'\s+', '', d.text)
                if "PrecioInternet" in i:
                    raw_prices = re.sub(r'PrecioInternet|:', '', i).replace(",", ".")

                    p = re.findall(r'[a-zA-Z]+|[0-9.]+', raw_prices)

                    assert len(p) % 2 == 0, "Error parsing prices"

                    d = {p[i]: float(p[i + 1]) for i in range(0, len(p), 2)}

                    prices.append(d)

    return prices


def get_stations(soup):
    menu = soup.find('div', {'class': 'irf-search-shedule__container-ipt'})
    options = menu.find_all('option')

    return {" ".join(filter(lambda x: x != "", opt.text.split(" "))): opt["value"] for opt in options}


def get_stops(soup):
    stops_links = soup.find_all('a', {'class': 'irf-travellers-table__tbody-lnk irf-travellers-table__tbody-lnk--icon-left'})
    root = "https://horarios.renfe.com/HIRRenfeWeb/"

    stops_main = {}

    for train in stops_links:
        if "recorrido.do" in train["href"]:
            js_link = str(train["href"]).replace("\n", "").replace("\t", "").replace(" ", "%20")

            pattern = r'\("(.+)"\)'
            match = re.search(pattern, js_link)

            if match:
                js_link = match.group(1)
                key = " ".join(filter(lambda x: x != "", re.sub(r"\s+", " ", train.text).split(" ")))
                stops_main[key] = root + js_link

    for stop in stops_main:
        req = requests.get(stops_main[stop])
        soup = BeautifulSoup(req.text, 'html.parser')
        table = soup.find('table', {'class': 'irf-renfe-travel__table cabecera_tabla'})

        stops = {}
        for row in table.find_all('tr'):
            aux = row.find_all('td')

            if aux:
                station = " ".join(filter(lambda x: x != "", map(lambda x: re.sub(r'\s+', "", str(x)), aux[0].text.split(" "))))
                departure_time = re.sub(r'\s+', "", aux[1].text)
                arrival_time = re.sub(r'\s+', "", aux[2].text)
                stops[station] = (departure_time, arrival_time)

        stops_main[stop] = stops
    return stops_main






