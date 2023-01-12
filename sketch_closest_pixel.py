import geopandas as gpd
import numba
import numpy as np
import pandas as pd

from src.data.gedi_database import GediDatabase
from src.data.jrc_loading import load_jrc_data


@numba.njit
def argnearest(array, values):
    """Finds the index of the nearest value in `array' to each element of `values'.
    Args:
        array: 1D array
        values: 1D array
    Returns:
        A 1D array of length equal to len(values).
         The ith element is the index in `array' such that array[index] is closest to values[i].
    """
    # initialize an array of length len(values)
    argmins = np.zeros_like(values, dtype=np.int64)
    # for each element in values
    for i, value in enumerate(values):
        # 1. Find the absolute difference between the value and all the elements of `array'
        # 2. Select the index where that difference is smallest (argmin)
        argmins[i] = (np.abs(array - value)).argmin()
    return argmins

@numba.njit
def arg_toptwo_nearest_centers(array, values):
    half_pixel = (array[1] - array[0]) / 2
    array_center = array + half_pixel
    argmins = np.zeros((*values.shape, 2), dtype=np.int64)
    for i, value in enumerate(values):
        argmins[i, 0] = (np.abs(array_center - value)).argmin()
        if value < array_center[argmins[i, 0]]:
            argmins[i, 1] = argmins[i, 0] - 1
        else:
            argmins[i, 1] = argmins[i, 0] + 1

    return argmins

def run_program(shapefile):
    # Might need to break up AOI into smaller regions and run repeatedly
    # 1. (DATA LOAD)
    #   a. Load GEDI data for AOI and year 2020
    shp_file = gpd.read_file("/home/okml2/biomass-recovery/shp_files/CIF_Alto_Mayo/aoi.shp")
    geometry = shp_file['geometry']
    database = GediDatabase() 

    gedi_shots = database.query(
        table_name="level_4a",
        columns=[
            "shot_number",
            "absolute_time",
            "lon_lowestmode",
            "lat_lowestmode",
            "agbd",
            "agbd_pi_lower",
            "agbd_pi_upper",
            "agbd_se",
            "l2_quality_flag",
            "l4_quality_flag",
            "degrade_flag",
            "beam_type",
            "sensitivity",
            "geometry",
        ],
        geometry=geometry,
        crs= "WGS84",
        start_time=f"2020-01-01",
        end_time=f"2021-01-01",
    )
    
    #   b. Load JRC AFC raster for AOI 2020
    jrc_raster = load_jrc_data(*geometry.bounds.values[0], dataset="AnnualChange", years=[2020])[0]

    # 2. (DATA PREPROCESSING)
    #   a. Match all GEDI shots with their closest pixel in AFC raster
    x_inds = argnearest(jrc_raster.x.data, gedi_shots.lon_lowestmode.values)
    y_inds = argnearest(jrc_raster.y.data, gedi_shots.lat_lowestmode.values)
    shot_class = jrc_raster.data[y_inds, x_inds]

    #Adding shot_class to gedi_shots dataframe
    gedi_shots["shot_class"] = shot_class

    # 3. (CALCULATION)
    #   a. Group GEDI shots by AFC class
    by_class = gedi_shots.groupby('shot_class')['agbd']
    
    #   b. Get description of AGBD values
    agbd_val = by_class.describe()
    
    # 4. (OUTPUT)
    return agbd_val.to_csv('r/home/okml2/biomass-recovery/xxxx')

"""
    class_ids = gedi_shots["shot_class"].unique()
    for class_id in class_ids:
        subset = gedi_shots[gedi_shots["shot_class"] == class_id]
        #   b. Take mean AGBD for each class value
        means = subset["agbd"].mean()
        by_class.append(subset["agbd"].mean())
    print(by_class)
        #   c. Take histogram dist of AGBD for each class value
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
       # hists.append(np.histogram(subset["agbd"]))
    print(class_ids)
    # 4. (OUTPUT)
    for class_id in class_ids:
        #   a. Save k values: mean for each of k classes
        #   b. Save k histograms: histogram for each of k classes
        pass
"""