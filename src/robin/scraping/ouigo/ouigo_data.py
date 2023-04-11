from src.robin.supply.utils import get_time
import pandas as pd
import datetime

origin = "60000"
destination = "71801"

# Trips .CSV
trips_raw = [["MB01", "OUIGO", "07:05", "09:50", "2:45"],
             ["MB02", "OUIGO", "10:05", "13:00", "2:55"],
             ["MB03", "OUIGO", "14:15", "16:45", "2:30"],
             ["MB04", "OUIGO", "17:20", "19:50", "2:30"],
             ["MB05", "OUIGO", "21:00", "23:45", "2:45"]]

date = datetime.datetime.strptime("2023-03-01", "%Y-%m-%d")
time_delta = datetime.timedelta(days=1)
end_date = datetime.datetime.strptime("2023-03-31", "%Y-%m-%d")

trips = []
while date <= end_date:
    for trip in trips_raw:
        service_id = trip[0] + "_" + date.strftime("%Y-%m-%d") + "_" + trip[2].replace(":", ".")
        trip_id = trip[0]
        train_type = trip[1]
        departure = trip[2]
        arrival = trip[3]
        duration = trip[4]
        trips.append([service_id, trip_id, train_type, departure, arrival, duration])
    date += time_delta

# trips nested list to pandas dataframe
df_trips = pd.DataFrame(trips, columns=["service_id", "trip_id", "train_type", "departure", "arrival", "duration"])

print(df_trips.head())

# save dataframe in ../../../data/scraping/ouito/trips/
df_trips.to_csv(f"../../../data/scraping/ouigo/trips/trips_{origin}_{destination}_2023-03-01_2023-03-31.csv", index=False)


# Prices .CSV
def get_prices(x):
    hour, minute = x.split("_")[2].split(".")
    d = datetime.time(int(hour), int(minute))
    weekday = date.weekday()

    if weekday in (0, 1, 2, 3, 5):
        if d < datetime.time(hour=10) or d > datetime.time(hour=20):
            return 25
        else:
            return 30
    else:
        return 40


df_trips['prices'] = df_trips['service_id'].apply(lambda x: get_prices(x))

df_prices = df_trips.filter(['service_id', 'prices'], axis=1)

print(df_prices.head())

# save dataframe in ../../../data/scraping/ouito/trips/
df_prices.to_csv(f"../../../data/scraping/ouigo/prices/prices_{origin}_{destination}_2023-03-01_2023-03-31.csv", index=False)


# Stop Times .CSV

service_id_raw = ["MB01", "MB02", "MB03", "MB04", "MB05"]
stops_raw = [[("60000", "07:05", "07:05"), ("04040", "08:20", "08:23"), ("71801", "09:50", "09:50")],
             [("60000", "10:05", "10:05"), ("04040", "11:22", "11:25"), ("71500", "12:21", "12:24"), ("71801", "13:00", "13:00")],
             [("60000", "14:15", "14:15"), ("71801", "16:45", "16:45")],
             [("60000", "17:20", "17:20"), ("71801", "19:50", "19:50")],
             [("60000", "21:00", "21:00"), ("04040", "22:15", "22:18"), ("71801", "23:45", "23:45")]]

date = datetime.datetime.strptime("2023-03-01", "%Y-%m-%d")
time_delta = datetime.timedelta(days=1)
end_date = datetime.datetime.strptime("2023-03-31", "%Y-%m-%d")

stop_times = []
while date <= end_date:
    for sid, trip in zip(service_id_raw, stops_raw):
        service_id = sid + "_" + date.strftime("%Y-%m-%d") + "_" + trip[0][1].replace(":", ".")
        relative = get_time(trip[0][1])
        for i, stop in enumerate(trip, start=1):
            arrival = (get_time(stop[1]) - relative).seconds // 60
            departure = (get_time(stop[2]) - relative).seconds // 60
            # service_id, stop_id, arrival, departure, stop_sequence
            stop_times.append([service_id, stop[0], arrival, departure, i])

    date += time_delta

# trips nested list to pandas dataframe
df = pd.DataFrame(stop_times, columns=["service_id", "stop_id", "arrival", "departure", "stop_sequence"])

print(df.head())

# save dataframe in ../../../data/scraping/ouito/trips/
df.to_csv(f"../../../data/scraping/ouigo/stop_times/stopTimes_{origin}_{destination}_2023-03-01_2023-03-31.csv", index=False)

