import os
import re
from glob import glob

import pandas as pd
import numpy as np

import shapely
import geopandas as gpd
import rasterio
import rasterio.merge
from rasterio.enums import Resampling
import rioxarray as riox
from rioxarray.merge import merge_arrays

DATASET = "orthophotos/nw"

# Convert an EPSG:4326 (lat, lon) pair into a buffer polygon, and return the
# range limits in EPSG:25832 values.
# [465000, 5769000] = [8.49, 52.07]
def pt_to_25832Range(lat: float, lon: float, buffer_dist: float = 100) -> pd.Series:
    pt = shapely.Point([lon, lat])
    pt = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[pt])
    # Re-project to "cea" = "Equal Area Cylindrical" which has
    # "unit_name=metre" (`.crs.axis_info`), and buffer:
    pt = pt.to_crs('+proj=cea').buffer(buffer_dist).to_crs('25832')
    bounds = pt.bounds.iloc[0]

    minx = bounds.loc['minx'].item()
    miny = bounds.loc['miny'].item()
    maxx = bounds.loc['maxx'].item()
    maxy = bounds.loc['maxy'].item()

    minx_i = int (minx // 1000)
    miny_i = int (miny // 1000)
    maxx_i = int (maxx // 1000 + 1)
    maxy_i = int (maxy // 1000 + 1)

    return [minx_i, miny_i, maxx_i, maxy_i]

# Convert an EPSG:4326 (lat, lon) pair into a buffer polygon, and return the
# coordinates pairs in EPSG:25832 which that polygon overlaps. Uses the
# `pt_to_25832Range` function.
def range_to_files(lat: float, lon:float, buffer_dist: float = 100):
    rng = pt_to_25832Range(lat, lon, buffer_dist)

    # Create sequences of integers:
    xvals = np.arange(rng[0], rng[2] + 1)
    yvals = np.arange(rng[1], rng[3] + 1)
    x_grid, y_grid = np.meshgrid(xvals, yvals)
    combs = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    comb_strings = ['_'.join(map(str, row)) for row in combs]

    # Plus index the extreme tiles, which by definition only partially overlap,
    # while all others must include full data:
    mask_x = np.logical_or(combs[:, 0] == rng[0], combs[:, 0] == rng[2])
    mask_y = np.logical_or(combs[:, 1] == rng[1], combs[:, 1] == rng[3])
    mask = np.logical_or(mask_x, mask_y).astype(int) # true == 1

    comb_strings = np.column_stack((comb_strings, mask))

    return comb_strings

# Use preceding functions to accept a (lat, lon) pair and buffer distance in
# metres, and return an array of the corresponding file names which the buffer
# overlaps.
def get_file_names(lat: float, lon:float, buffer_dist: float = 100):
    latlon_combs = range_to_files(lat, lon, buffer_dist)
    latlon_mask = latlon_combs[:, 1].tolist()
    latlon_combs = latlon_combs[:, 0].tolist()
    files = glob(DATASET + "/*.jp2")

    filtered = []
    for f in files:
        if any(re.search(ll, f) for ll in latlon_combs):
            filtered.append(f)

    return filtered

def aggregate_one_file (f, out_size: int = 256):
    r0 = rasterio.open(f)
    out_size = int(256)
    r0ag = r0.read(
        out_shape=(r0.count, out_size, out_size),
        resampling=Resampling.average
    )
    r0ag = r0.read(out_shape=(r0.count, out_size, out_size), resampling=Resampling.average)
    new_transform = r0.transform * r0.transform.scale((r0.width / out_size), (r0.height / out_size))
    dst_kwargs = r0.meta.copy()
    dst_kwargs.update({
        'transform': new_transform,
        'count': r0.count,
        'width': out_size,
        'height': out_size
    })
    # Then use those updated args to write new aggregate data:
    fnew = os.path.basename(f)
    dst = rasterio.open(fnew, 'w', **dst_kwargs)
    dst.write(r0ag)
    dst.close()

    return fnew

# d = rasterio.open('junk.tif')
# rasterio.plot.show(d)
files = get_file_names(52.07, 8.49)
fnew = []
for f in files:
    fnew.append(aggregate_one_file(f))
    # os.remove(f)

r = []
for f in fnew:
    r.append(riox.open_rasterio(f))


merged_raster = merge_arrays(dataarrays = r, crs='epsg:25832')
# This errors:
# merged_raster.rio.to_raster("merged.tif")
# # So this code does it manually:
transform = merged_raster.rio.transform()
# Read one file to get kwargs:
ftmp = rasterio.open(fnew[0])
dst_kwargs = ftmp.meta.copy()
dst_kwargs.update({
    'transform': transform,
    'count': merged_raster.shape[0],
    'width': merged_raster.shape[1],
    'height': merged_raster.shape[2]
})

merged_arr = merged_raster.to_numpy()
merged_data = rasterio.open('junk.tif', 'w', **dst_kwargs)
merged_data.write(merged_arr)
merged_data.close()

for f in fnew:
    os.remove(f)
