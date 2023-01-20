from renfetools import *


def renfe_scraping_trips(origin_id: str, destination_id: str, date: datetime.date, range_days: int):
    """
    Scraping trips from Renfe website from origin to destination and from date to date + range_days

    :param:
        origin_id: Renfe station id
        destination_id: Renfe station id
        date: datetime.date object
        range_days: number of days to scrape

    :returns: None

    Generates two csv files: trips and stop_times
        trips: service_id,trip_id,train_type,departure,arrival,duration
        stop_times: service_id,stop_id,arrival,departure,stop_sequence
    """

    assert range_days > 0, "range_days must be greater than 0"

    print("Scraping trips...")

    init_date = date
    for i in range(range_days):
        # Get year, month and day from date as strings to generate the URL
        year, month, day = str(date).split("-")

        # Get day of week starting at sunday = 0 to generate the URL
        weekday = date.weekday() + 1

        url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
        print("Search url: ", url)
        print("Date: ", date)

        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')

        if not i:  # If first iteration, create the dataframe
            df = to_dataframe(soup, date, url)
        else:
            df = pd.concat([df, to_dataframe(soup, date, url)])

        # Sum one day to date
        date += datetime.timedelta(days=1)

    df = df.reset_index(drop=True)
    df = df[['service_id', 'trip_id', 'train_type', 'stops', 'departure', 'arrival', 'duration', 'price']]

    print(df.columns)
    print(df.iloc[-1])

    end_date = date - datetime.timedelta(days=1)

    # Dictionary with stops for each service_id
    # keys: service_id
    # values: other dictionary with stops {station_id: (arrival, departure), ...}
    stop_times = dict(zip(df.service_id, df.stops))

    # List of lists for the previous dictionary to build a dataframe with the following columns:
    # 'service_id', 'stop_id', 'arrival', 'departure', 'stop_sequence'
    list_stop_times = []
    for k, v in stop_times.items():
        stops = list(zip(v.keys(), v.values()))

        for i, ts in enumerate(stops, 1):
            list_stop_times.append([k, ts[0], *ts[1], i])

    df_stops = pd.DataFrame(list_stop_times, columns=['service_id', 'stop_id', 'arrival', 'departure', 'stop_sequence'])

    # Save stop_times dataframe to csv
    df_stops.to_csv(f"../../datasets/scraping/renfe/stop_times/TEST_stopTimes_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False, header=True)
    print("Saved stop times")

    # Remove stops and price columns
    df = df.drop('stops', axis=1)
    df = df.drop('price', axis=1)

    # Save trips dataframe to csv in datasets folder
    df.to_csv(f"../../datasets/scraping/renfe/trips/TEST_trips_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False)
    print("Saved trips")


if __name__ == "__main__":
    date = datetime.date.today()
    date += datetime.timedelta(days=1)
    range_days = 1
    origin_id = 'MADRI'  # Renfe id for Madrid Puerta de Atocha
    destination_id = 'BARCE'  # Renfe id for Barcelona Sants
    renfe_scraping_trips(origin_id, destination_id, date, range_days)

