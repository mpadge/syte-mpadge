import pandas as pd
import numpy as np
import shapely
import geopandas as gpd
import pyproj

def pt_to_25832Range(lat: float, lon: float) -> pd.Series:
    pt = shapely.Point([lon, lat])
    pt = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[pt])
    # Re-project to "cea" = "Equal Area Cylindrical" and buffer:
    pt = pt.to_crs('+proj=cea').buffer(100).to_crs('25832')
    return pt.bounds.iloc[0]
