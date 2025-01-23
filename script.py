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
from rasterio.plot import reshape_as_image
import rioxarray as riox
from rioxarray.merge import merge_arrays
from PIL import Image

DATASET = "orthophotos/nw"

# Convert an EPSG:4326 (lat, lon) pair into a buffer polygon, and return the
# range limits in EPSG:25832 values.
# [465000, 5769000] = [8.49, 52.07]
def pt_to_25832Range(lat: float, lon: float, buffer_dist: float = 100, round: bool = True) -> pd.Series:
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

    # Note that images are named by (left, bottom), so max values are also
    # rounded down:
    if round:
        minx = int (minx // 1000)
        miny = int (miny // 1000)
        maxx = int (maxx // 1000)
        maxy = int (maxy // 1000)

    return [minx, miny, maxx, maxy]

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

# Aggregate one high-resolution file into lower resolution defined by
# `out_size` parameter.
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

def generate_merged_files(lat: float, lon:float, buffer_dist: float = 100, outfile: str = 'output.tif'):
    files = get_file_names(lat, lon, buffer_dist)
    fnew = []
    for f in files:
        fnew.append(aggregate_one_file(f))

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
    merged_data = rasterio.open(outfile, 'w', **dst_kwargs)
    merged_data.write(merged_arr)
    merged_data.close()

    for f in fnew:
        os.remove(f)

    return outfile

def get_image_crop_limits(img: rasterio.io.DatasetReader, lat: float, lon: float, buffer_dist: float):
    bounds = img.bounds
    pt_range = pt_to_25832Range(lat, lon, buffer_dist, round = False)

    height_img = bounds.top - bounds.bottom
    width_img = bounds.right - bounds.left

    height_crop = int(img.shape[0] * (pt_range[3] - pt_range[1]) / height_img)
    width_crop = int(img.shape[1] * (pt_range[2] - pt_range[0]) / width_img)
    bottom_crop = int(img.shape[0] * (pt_range [1] - bounds.bottom) / height_img)
    left_crop = int(img.shape[1] * (pt_range[2] - bounds.left) / width_img)

    return [left_crop, bottom_crop, left_crop + width_crop, bottom_crop + height_crop]

def get_image(lat: float, lon:float, buffer_dist: float = 100):
    f = generate_merged_files(lat, lon, buffer_dist)
    img = rasterio.open(f)
    crop_limits = get_image_crop_limits(img, lat, lon, buffer_dist)
    img = reshape_as_image(img.read())
    os.remove(f)
    img = img[crop_limits[0]:crop_limits[2]:1, crop_limits[1]:crop_limits[3]:1, :]
    img = Image.fromarray(img)
    return img

lat = 52.07
lon = 8.49
buffer_dist = 100

import time
def measure_get_image_scaling(lat: float, lon: float, npts: int = 3):
    buffers = [50, 100, 200, 500, 1000]
    durations = []
    rpts = 3
    for b in buffers:
        start = time.time()
        for r in range(rpts):
            img = get_image(lat, lon, b)
        stop = time.time()
        durations.append((stop - start) / npts)

    return [buffers, durations]

dat = measure_get_image_scaling(lat, lon)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_xlabel("Buffer dist(m)")
ax.set_ylabel("Calculation time(s)")
ax.set_title("Scaling of 'get_image' with buffer radius")
ax.plot(dat[0], dat[1])
plt.show()
