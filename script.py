import pandas as pd
import numpy as np
import shapely
import geopandas as gpd
import pyproj

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

def range_to_files(lat: float, lon:float, buffer_dist: float = 100):

    rng = pt_to_25832Range(lat, lon, buffer_dist)

    # Create sequences of integers:
    xvals = np.arange(rng[0], rng[2] + 1)
    yvals = np.arange(rng[1], rng[3] + 1)
    x_grid, y_grid = np.meshgrid(xvals, yvals)
    combs = np.column_stack((x_grid.flatten(), y_grid.flatten()))
    comb_strings = ['_'.join(map(str, row)) for row in combs]

    return comb_strings
