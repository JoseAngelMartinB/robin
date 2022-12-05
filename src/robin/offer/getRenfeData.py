from cartopy.io import shapereader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import geopandas
import requests
import datetime
import shutil
import errno
import os


def download_data(url, path):
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY

    # Download zip file
    req = requests.get(url)

    # Get file name from url
    filename = url.split('/')[-1]

    if not os.path.exists(path):
       os.makedirs(path)

    try:
        file_handle = os.open(path+filename, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            print('File already exists!')
        else:  # Something unexpected went wrong so reraise the exception.
            raise
    else:  # No exception, so the file must have been created successfully.
        # Writing the file to the local file system
        with open(path+filename, 'wb') as output_file:
            output_file.write(req.content)

        # Unzip
        shutil.unpack_archive(path+filename, path)

        print('File downloaded successfully!')

    return filename


def import_datasets(path):
    data = {}
    for fn in os.listdir(path):
        if fn.endswith('.txt'):
            data[fn.split(".")[0]] = pd.read_csv(path + fn, delimiter=',')

    return data


def parse_date(d):
    return datetime.datetime.strptime(str(d), '%Y%m%d').date().strftime("%d/%m/%Y")


def plot_route(stops_coords):
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

    x, y = zip(*stops_coords)
    ax.scatter(x, y)

    for i, _ in enumerate(range(len(stops_coords) - 1)):
        ax.annotate(text='', xy=stops_coords[i + 1], xytext=stops_coords[i], arrowprops=dict(arrowstyle='->'))

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
