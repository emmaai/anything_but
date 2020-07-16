from datacube import Datacube
from datacube.virtual.impl import VirtualDatasetBox
from datacube.virtual import construct
from datacube.virtual.transformations import ApplyMask
from datacube.utils.geometry import CRS, Geometry
import yaml
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging, sys
import io
import pandas as pd
import fiona
from os import path
from rasterio import features
from rasterio.warp import calculate_default_transform, transform_geom, aligned_target
from pandas.plotting import register_matplotlib_converters
from wit_tooling import shape_list, convert_shape_to_polygon

from mpi4py.futures import MPIPoolExecutor
from osgeo import ogr
from osgeo import gdal
from osgeo import osr

register_matplotlib_converters()
_LOG = logging.getLogger('volcano')
stdout_hdlr = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s')
stdout_hdlr.setFormatter(formatter)
_LOG.addHandler(stdout_hdlr)
_LOG.setLevel(logging.DEBUG)

landsat_path_shapefile = '/g/data/u46/users/ea6141/aus_map/landsat_au.shp'
au_coastline_shapefile = '/g/data/u46/users/ea6141/aus_map/aus_land.shp'
def load_timeslice(product_def, time_slice):
    results = product_def.fetch(time_slice)
    results = ApplyMask('pixelquality', apply_to=['blue', 'green', 'swir2']).compute(results)
    return results

def darkest_pixels(pixels_array, amount=5):
    threshold = np.nanpercentile(pixels_array, amount)
    if threshold > 800:
        return None, None
    dark_pixels = pixels_array[pixels_array <= threshold]
    return dark_pixels.mean(), dark_pixels.std()

def plot_mean_std(time_mark, darkest_mean, darkest_std):
    fig, ax = plt.subplots(figsize = (22,6))
    line_mean = ax.plot(time_mark, darkest_mean, 'o-', color='blue', label='mean')
    line_std = ax.plot(time_mark, darkest_mean+darkest_std, '--', color='red', label='1 std')
    ax.plot(time_mark, darkest_mean-darkest_std, '--', color='red', label='1 std')
    lines = line_mean + line_std
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels)
    fmt_xdata = mdates.DateFormatter('%Y-%m-%d')
    locator = mdates.MonthLocator(range(1, 13), interval=2)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt_xdata)
    ax.set_title("5% darkest pixels over time (green band)")
    fig.autofmt_xdate()
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    plt.close()
    return bytes_image

def clip_coastline(geobox):
    source_ds = ogr.Open(au_coastline_shapefile)
    source_layer = source_ds.GetLayer()

    yt, xt = geobox.shape
    xres = 25
    yres = -25
    no_data = 0
    xcoord = geobox.coords['x'].values.min()
    ycoord = geobox.coords['y'].values.max()
    geotransform = (xcoord - (xres*0.5), xres, 0, ycoord - (yres*0.5), 0, yres)

    target_ds = gdal.GetDriverByName('MEM').Create('', xt, yt, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geotransform)
    albers = osr.SpatialReference()
    albers.ImportFromEPSG(3577)
    target_ds.SetProjection(albers.ExportToWkt())
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data)

    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
    return band.ReadAsArray()

def generate_raster(shapes, geobox):
    yt, xt = geobox.shape
    transform, width, height = calculate_default_transform(
        geobox.crs, geobox.crs.crs_str, xt, yt, *geobox.extent.boundingbox)
    transform, width, height = aligned_target(transform, xt, yt, resolution=(25, 25))
    target_ds = features.rasterize(shapes,
        (yt, xt), fill=0, transform=transform, all_touched=False)
    return target_ds

def load_cal(product_def, time_slice, mask):
    data_before = load_timeslice(product_def, time_slice)
    data_before = data_before.where((data_before != -999) & (mask==1))
    valid_pixels_count = np.count_nonzero(mask)
    if (data_before.green.count().data / valid_pixels_count) < 0.7:
        return None, None
    _LOG.debug("valid percent of %s is %s", time_slice.box.time.data[0], data_before.green.count().data / valid_pixels_count)
    tmp_mean, tmp_std = darkest_pixels(data_before.green.data.reshape(-1))
    return (time_slice.box.time.data[0], tmp_mean)


def cal_mean_std(query_poly):
    landsat_yaml = 'nbart_ld.yaml'
    with open(landsat_yaml, 'r') as f:
        recipe = yaml.safe_load(f)
    landsat_product = construct(**recipe)
    query = {'time': ('1987-01-01', '2000-01-01')}
    location = {'geopolygon': Geometry(query_poly, CRS("EPSG:3577"))}
    query.update(location)

    dc = Datacube()
    datasets = landsat_product.query(dc, **query)
    grouped = landsat_product.group(datasets, **query)
    _LOG.debug("datasets %s", grouped)

    mask = generate_raster([(query_poly, 1)], grouped.geobox)
    coastline_mask = clip_coastline(grouped.geobox)
    mask[coastline_mask == 0] = 0
    _LOG.debug("mask size %s none zero %s", mask.size, np.count_nonzero(mask))
    if np.count_nonzero(mask) == 0:
        return [], []

    darkest_mean = []
    time_mark = []
    future_list = []

    with MPIPoolExecutor() as executor:
        for i in range(grouped.box.time.shape[0]):
            time_slice = VirtualDatasetBox(grouped.box.sel(time=grouped.box.time.data[i:i+1]), grouped.geobox,
                            grouped.load_natively, grouped.product_definitions, grouped.geopolygon)
            future = executor.submit(load_cal, landsat_product, time_slice, mask)
            future_list.append(future)

    for future in future_list:
        r = future.result()
        if r[1] is not None:
            _LOG.debug("darkest time %s", r[0])
            _LOG.debug("darkest mean %s", r[1])
            time_mark.append(r[0])
            darkest_mean.append(r[1])
    return time_mark, darkest_mean

def stats_by_month(df):
    df = df.groupby("time").mean().reset_index()
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)
    monthly_median = []
    monthly_thresh = []
    months = []
    for m in df.index.month.unique().sort_values():
        month_data = df[(df.index.month==m) & (df.index <= np.datetime64('1991-06-15', 'D'))]
        if month_data.size < 2:
            continue
        months.append(m)
        monthly_median.append(month_data.dark_mean.median())
        qtl3 = np.percentile(month_data.dark_mean, 75, interpolation = 'midpoint')
        qtl1 = np.percentile(month_data.dark_mean, 25, interpolation = 'midpoint')
        monthly_thresh.append(qtl1 + (qtl3-qtl1) * 1.5)
    return months, monthly_median, monthly_thresh

def month_ppnormal(df, standard, min_start_days=1, min_end_days=1):
    volcano_erupt_time = np.datetime64('1991-06-15')
    last_normal = volcano_erupt_time
    last_abnormal = None
    abnormal_start = None
    normal_reset = None
    rolling_window = None
    df = df.groupby("time").mean().reset_index()
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index)
    tmp_df = df.loc[(df.index.year >= 1991) & (df.index.year < 1996)]
    for index, m in tmp_df.iterrows():
        if index.month not in standard[0]:
            continue
        tmp_thresh = standard[1][np.where(standard[0] == index.month)[0]]
        if m['dark_mean'] > tmp_thresh:
            if abnormal_start is None and index > volcano_erupt_time and (index - volcano_erupt_time) / np.timedelta64(1, "D") <= min_start_days:
                if last_normal <= volcano_erupt_time:
                    last_normal = volcano_erupt_time
                    #if 6 in standard[0]:
                    #    last_normal = volcano_erupt_time
                    #else:
                    #    last_normal = int(standard[0][standard[0] > 6].min())
                    #    if len(str(last_normal)) == 1:
                    #        last_normal = '0' + str(last_normal)
                    #    else:
                    #        last_normal = str(last_normal)
                    #    last_normal = np.datetime64(str(index.year) + "-" + last_normal, 'D')
                    #    print("should go here", last_normal)

                abnormal_start = last_normal  + (index - last_normal) / 2
            last_abnormal = index
        else:
            if abnormal_start is None:
                last_normal = index
            elif index > abnormal_start:
                if (index - abnormal_start) / np.timedelta64(1, "D") <= min_end_days:
                    abnormal_start = None
                    last_normal = index
                    continue
                if (index - volcano_erupt_time) / np.timedelta64(1, "D") <= 400:
                    continue
                normal_reset = index
                count = 0
                observe_count = 0
                rolling_window = tmp_df[(tmp_df.index >= index) & (tmp_df.index <= index + np.timedelta64(1, 'Y'))]
                for index, m in rolling_window.iterrows():
                    if index.month not in standard[0]:
                        continue
                    observe_count += 1
                    tmp_thresh = standard[1][np.where(standard[0] == index.month)[0]]
                    if m['dark_mean'] <= tmp_thresh:
                        count += 1
                if count > 0.5 * observe_count:
                    normal_reset = last_abnormal + (normal_reset - last_abnormal)  /2
                    break
    return volcano_erupt_time, normal_reset
    #return abnormal_start, normal_reset

def main():
    with fiona.open(landsat_path_shapefile) as allshapes:
        crs = allshapes.crs_wkt
    for shape in shape_list(landsat_path_shapefile):
        csv_fname = 'threshold/' + str(shape['properties'].get('WRSPR')) + '.csv'
        if path.exists(csv_fname):
            continue
        _LOG.debug("process %s", str(shape['properties'].get('WRSPR')))
        shape['geometry'] = transform_geom(crs, "EPSG:3577", shape['geometry'])
        query_poly = convert_shape_to_polygon(shape['geometry'])
        query_poly = query_poly.buffer(-4500, join_style=3)
        _LOG.debug("query poly %s", query_poly)
        time_mark, darkest_mean = cal_mean_std(query_poly)
        df = pd.DataFrame(data={"time": np.array(time_mark).astype("datetime64[D]"), "dark_mean": darkest_mean}).to_csv(csv_fname, index=False)
        #pd.DataFrame(data={'month': np.array(months).astype('int32'), 'median': np.array(monthly_median).astype('float32'),
        #    'threshold': np.array(monthly_thresh).astype('float32')}).to_csv(csv_fname, index=False)

if __name__ == "__main__":
    main()
