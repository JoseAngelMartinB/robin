import datetime

from src.scraping.yaml_utils import *

import numpy as np
import yaml

import os
import glob


def load_scraping(file):
    # 0. Import data

    # Get last updated file in selected folder
    # updated_file = max(glob.iglob(f'../../datasets/scraping/renfe/trips/*.csv'), key=os.path.getmtime)
    # file = updated_file

    # File with Renfe data for february 2023 (AVE)
    # E.g updated_file = 'datasets/scraping/renfe/trips/trips_MADRI_BARCE_2022-12-30_2023-01-03.csv'

    # 0.1 Import trips
    trips = pd.read_csv(file, delimiter=',', dtype={'trip_id': str})

    # E.g file_name = 'trips_MADRI_BARCE_2022-12-30_2023-01-03'
    file_name = file.split('/')[-1].split(".")[0]

    # E.g file_name = 'MADRI_BARCE_2022-12-30_2023-01-03'
    file_name = "_".join(file_name.split("_")[1:])

    # 0.2 Import prices
    prices = pd.read_csv(f'../../../data/scraping/renfe/prices/prices_{file_name}.csv', delimiter=',')

    # 0.3 Import stops
    stop_times = pd.read_csv(f'../../../data/scraping/renfe/stop_times/stopTimes_{file_name}.csv',
                             delimiter=',',
                             dtype={'stop_id': str})

    # Get metadata from file name
    # E.g. origin_id = 'MADRI', destination_id = 'BARCE', start_date = '2022-12-30', end_date = '2023-01-03'
    origin_id, destination_id, start_date, end_date = file_name.split('_')

    print(f"Origin:{origin_id} - Destination:{destination_id}\nSince: {start_date}, Until: {end_date}")

    # 0.4 Build seats for Renfe AVE
    renfe_seats = (Seat(1, "Turista", 1, 1), Seat(2, "Turista Plus", 1, 2), Seat(3, "Preferente", 2, 1))

    trips['prices'] = trips['service_id'].apply(lambda x: get_trip_price(x, renfe_seats, prices))

    # Filter trips by price column to remove trips with any NaN value
    trips = trips[trips['prices'].apply(lambda x: not any(np.isnan(p) for p in x.values()))]

    # 1. Build Corridor MAD-BAR

    # 1.1 Group dataframe by trip_id
    grouped_df = stop_times.groupby('service_id')

    # 1.2 Get nested list with stops for each trip
    list_stations = grouped_df.apply(lambda d: list(d['stop_id'])).values.tolist()

    # 1.3 Initialize corridor with max length trip
    corridor = list_stations.pop(list_stations.index(max(list_stations, key=len)))

    # 1.4 Complete corridor with other stops that are not in the initial defined corridor
    for trip in list_stations:
        for i, s in enumerate(trip):
            if s not in corridor:
                corridor.insert(corridor.index(trip[i + 1]), s)

    # 1.5 Parse stations. Use Adif stop_id retrieve station info (name, lat, lon)
    renfe_stations = pd.read_csv(f'../../../data/scraping/renfe/renfe_stations.csv', delimiter=',', dtype={'stop_id': str})

    # 1.6 Build dictionary of stations with stop_id as key and Station() object as value
    stations = {}
    for s in corridor:
        name = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
        city = renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0]
        shortname = str(renfe_stations[renfe_stations['stop_id'] == s]['stop_name'].values[0])[:3].upper()
        coords = tuple(renfe_stations[renfe_stations['stop_id'] == s][['stop_lat', 'stop_lon']].values[0])

        stations[s] = Station(s, name, city, shortname, coords)

    print(list(stations.values())[0])

    def write_to_yaml(filename, objects, key):
        def represent_list(dumper, data):
            return dumper.represent_sequence(u'tag:yaml.org,2002:seq', data, flow_style=True)

        # yaml.SafeDumper.add_representer(list, represent_list)

        with open(filename, 'r') as yaml_file:
            yaml_file_mod = yaml.safe_load(yaml_file)
            try:
                yaml_file_mod.update(objects)
            except AttributeError:
                yaml_file_mod = objects

        if yaml_file_mod:
            with open(filename, 'w') as yaml_file:
                yaml.safe_dump(yaml_file_mod, yaml_file, sort_keys=False, allow_unicode=True)

    write_to_yaml('../../../data/supply_data.yml',
                  {'stations': [station_to_dict(stn) for stn in stations.values()]},
                  'stations')

    write_to_yaml('../../../data/supply_data.yml',
                  {'seat': [seat_to_dict(s) for s in renfe_seats]},
                  'seat')

    # 1.7 Build Corridor
    first_station, last_station = tuple(stations.values())[::len(stations) - 1]
    corr_name = first_station.shortname + "-" + last_station.shortname
    corrMadBar = Corridor(1, corr_name, list(stations.keys()))

    write_to_yaml('../../../data/supply_data.yml',
                  {'corridor': [corridor_to_dict(corr) for corr in [corrMadBar]]},
                  'corridor')

    # 2. Build Lines
    routes_lines = grouped_df.apply(lambda x: get_line(x, corrMadBar))

    set_lines = {line.id: line for line in list(set(routes_lines.values.tolist()))}

    trips['lines'] = trips['service_id'].apply(lambda x: get_trip_line(x, set_lines))

    write_to_yaml('../../../data/supply_data.yml',
                  {'line': [line_to_dict(ln) for ln in trips['lines'].values.tolist()]},
                  'line')

    # 4. Build RollingStock for Renfe AVE
    renfe_rs = [RollingStock(1, "S-114", {1: 250, 2: 50})]

    write_to_yaml('../../../data/supply_data.yml',
                  {'rollingStock': [rolling_stock_to_dict(rs) for rs in renfe_rs]},
                  'rollingStock')

    # 5. Build TSP for Renfe
    renfe_tsp = TSP(1, "Renfe", [rs.id for rs in renfe_rs])

    write_to_yaml('../../../data/supply_data.yml',
                  {'trainServiceProvider': [tsp_to_dict(t) for t in [renfe_tsp]]},
                  'trainServiceProvider')

    # 6. Build Services
    trips['service'] = trips.apply(lambda x: get_service(x['service_id'],
                                                         x['departure'],
                                                         x['arrival'],
                                                         x['prices'],
                                                         x['lines'],
                                                         renfe_tsp,
                                                         renfe_rs[0],
                                                         corrMadBar),
                                   axis=1)

    my_services = trips['service'].values.tolist()
    time_slots = {s.timeSlot.id: s.timeSlot for s in my_services}

    write_to_yaml('../../../data/supply_data.yml',
                  {'timeSlot': [time_slot_to_dict(s) for s in time_slots.values()]},
                  'timeSlot')

    write_to_yaml('../../../data/supply_data.yml',
                  {'service': [service_to_dict(s) for s in trips['service'].values.tolist()]},
                  'service')

    return trips['service'].values.tolist()


if __name__ == '__main__':
    path = '../../../data/supply_data_example.yml'
    with open(path, 'r') as file:
        yml_example = yaml.safe_load(file)

    print(yml_example)

    file_path = '../../../data/scraping/renfe/trips/trips_MADRI_BARCE_2023-03-01_2023-03-31.csv'
    services = load_scraping(file_path)
    print(services[0])
