from cartopy.io import shapereader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from entities import *
import pandas as pd
import geopandas
import requests
import datetime
import shutil
import errno
import os


def download_data(url, path):
    """
    Download data from url to path
    url: url to download
    path: path to save downloaded file
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    # Download zip file
    req = requests.get(url)

    # Get file name from url
    filename = url.split('/')[-1]

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        file_handle = os.open(path + filename, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            print('File already exists!')
        else:  # Something unexpected went wrong so reraise the exception.
            raise
    else:  # No exception, so the file must have been created successfully.
        # Writing the file to the local file system
        with open(path + filename, 'wb') as output_file:
            output_file.write(req.content)

        # Unzip
        shutil.unpack_archive(path + filename, path)

        print('File downloaded successfully!')

    return filename


def import_datasets(path):
    """Import datasets from local files"""
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.txt'):
            data[fn.split(".")[0]] = pd.read_csv(path + fn, delimiter=',')

    return data


def parse_date(d):
    """
    Parse date string to datetime object
    d: date string
    """
    return datetime.datetime.strptime(str(d), '%Y%m%d').date().strftime("%d/%m/%Y")


def plot_route(stops_coords):
    """
    Plot route
    stops_coords: list of coordinates of each station in the route
    """
    # get natural earth data (http://www.naturalearthdata.com/)

    # get country borders
    resolution = '10m'
    category = 'cultural'
    name = 'admin_0_countries'

    shapefilename = shapereader.natural_earth(resolution, category, name)

    # read the shapefile using geopandas
    df = geopandas.read_file(shapefilename)

    # read Spain borders
    poly = df.loc[df['ADMIN'] == 'Spain']['geometry'].values[0]

    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.add_geometries([poly], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='0.5')

    # Unpack x and y coordinates from trip_coords array
    x, y = zip(*stops_coords)

    ax.scatter(x, y)

    for i, _ in enumerate(range(len(stops_coords) - 1)):
        ax.annotate(text='', xy=stops_coords.iloc[i + 1], xytext=stops_coords.iloc[i], arrowprops=dict(arrowstyle='->'))

    bounds = poly.bounds  # minx, miny, max_x, max_y
    # (-18.167225714999915, 27.642238674000055, 4.337087436000047, 43.79344310100004)
    bounds = list(bounds)
    bounds[0] = -9.4
    bounds[1] = 35.7
    m = 0.4  # Margin
    bounds = [l - m if i < 2 else l + m for i, l in enumerate(bounds)]

    ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]])

    ax.set_title("Spain Borders (Peninsula)")
    plt.show()


def parse_time(times):
    """
    Parse time string to datetime object
    time: time string
    """
    return tuple(map(lambda t: datetime.datetime.strptime(t, '%H:%M:%S').time(), times))


def get_trip(org_id, dest_id, stop_times, stations, shortest=True):
    """
    Get trip from origin to destination
    :param org_id: id of origin station
    :param dest_id: id of destination station
    :param stop_times: dataframe of stop times
    :param stations: dataframe with stops information
    :param shortest: boolean to indicate if shortest or fastest trip is desired
    :return: None
    """
    # Group dataframe by trip_id
    grouped_df = stop_times.groupby('trip_id')

    # Dictionary keys: trip_id, value: list of stop_ids
    routes_series = grouped_df.apply(lambda g: tuple(g['stop_id']))

    # Search routes from org_id to dest_id
    routes = routes_series[routes_series.apply(lambda r: r[0] == org_id and r[-1] == dest_id)]

    # Apply function to each group and save results in new series
    num_stops_series = routes.apply(lambda r: len(r))

    if shortest:
        # Get the trip_id of the shortest route
        trip_id = num_stops_series.idxmin()
    else:
        # Get the trip_id of the fastest route
        trip_id = num_stops_series.idxmax()

    # Get the stops in the trip
    trip_stops = stop_times[stop_times['trip_id'] == trip_id]['stop_id'].unique()

    def get_station(stop_id):
        df_loc = stations.query('stop_id == @stop_id')
        name, lon, lat = df_loc[['stop_name', 'stop_lon', 'stop_lat']].values[0]
        short_name = name[:3]
        return Station(stop_id, name, short_name, (lon, lat))

    # Get the stations in the trip as Station objects
    stations = stop_times[stop_times['trip_id'] == trip_id]['stop_id'].apply(get_station)

    # Get the coordinates of the stops in the trip from stations
    stops_coords = stations.apply(lambda s: s.coords)

    # Plot route
    plot_route(stops_coords)

    # Get arrival and departure times for each station
    def get_times(stop_id, trip_id):
        df_loc = stop_times.query('trip_id == @trip_id and stop_id == @stop_id')
        return df_loc[['arrival_time', 'departure_time']].values[0]

    times = stop_times[stop_times['trip_id'] == trip_id]['stop_id'].apply(get_times, args=(trip_id,))

    # Parse times to datetime objects
    times = times.apply(parse_time)

    # Define corridor of stations
    new_corr = Corridor(1, stations)

    new_service = []
    for i, t in enumerate(times):
        # if At != Dt, then this Station is attended by the service
        # or 1st / last Station (both considered attended)
        if i in (0, len(times) - 1) or t[0] != t[1]:
            new_service.append(1)
        else:
            new_service.append(0)

    new_service = tuple(new_service)

    new_line = Line(1, new_corr, new_service, times)

    print("Train stops 'j' in Line")
    stop_times = [t for (t, b) in zip(new_line.schedule, new_service) if b]

    for j, schedule in zip(new_line.J, stop_times):
        at = schedule[0]
        dt = schedule[1]
        print(j.name, j.id, "- AT: ", at, " - DT: ", dt)
