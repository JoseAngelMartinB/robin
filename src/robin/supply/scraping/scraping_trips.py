from renfetools import *
import numpy as np


def renfe_scraping_trips(origin_id, destination_id, date, range_days):
    print("Scraping trips...")

    init_date = date
    for i in range(range_days):
        # Get year, month and day from date as strings
        year, month, day = str(date).split("-")

        # Get day of week starting at sunday = 0
        weekday = date.weekday() + 1

        url = f'https://horarios.renfe.com/HIRRenfeWeb/buscar.do?O={origin_id}&D={destination_id}&AF={year}&MF={month}&DF={day}&SF={weekday}&ID=s'
        print("Search url: ", url)
        print("Date: ", date)

        req = requests.get(url)
        soup = BeautifulSoup(req.text, 'html.parser')

        if not i:
            df = to_dataframe(soup, date, url)
        else:
            df = pd.concat([df, to_dataframe(soup, date, url)])

        # Sum one day to date
        date += datetime.timedelta(days=1)

    df = df.reset_index(drop=True)
    df = df[['service_id', 'trip_id', 'train_type', 'stops', 'departure', 'arrival', 'duration', 'price']]

    # print(df.describe(datetime_is_numeric=True))
    print(df.columns)
    print(df.iloc[-1])

    end_date = date - datetime.timedelta(days=1)

    # Save numpy file with prices
    stop_times = dict(zip(df.service_id, df.stops))
    print(stop_times)

    list_stop_times = []
    for k, v in stop_times.items():
        stops = list(zip(v.keys(), v.values()))

        for i, ts in enumerate(stops, 1):
            list_stop_times.append([k, ts[0], *ts[1], i])

    df_stops = pd.DataFrame(list_stop_times, columns=['service_id', 'stop_id', 'arrival', 'departure', 'stop_sequence'])
    
    df_stops.to_csv(f"datasets/stop_times/stopTimes_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False, header=True)
    print("Saved stop times")

    # np.save(f"datasets/stop_times/stopTimes_{origin_id}_{destination_id}_{init_date}_{end_date}.npy", stop_times)

    df = df.drop('stops', axis=1)
    df = df.drop('price', axis=1)

    # Save dataframe to csv in datasets folder
    df.to_csv(f"datasets/trips/trips_{origin_id}_{destination_id}_{init_date}_{end_date}.csv", index=False)
    print("Saved trips")


if __name__ == "__main__":
    date = datetime.date.today()
    range_days = 1
    origin_id = 'MADRI'
    destination_id = 'BARCE'
    renfe_scraping_trips(origin_id, destination_id, date, range_days)

