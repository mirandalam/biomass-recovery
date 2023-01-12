import os
import numba
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray as rxr

import pathlib
import argparse

from src.data.gedi_database import GediDatabase
from src.data.jrc_loading import load_jrc_data

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

def run_program(shapefile : pathlib.Path()):
    # 1. (DATA LOAD)
        #   a. Load GEDI data for AOI and year 2020
    shp_file = gpd.read_file(shapefile)
    project = os.path.basename(os.path.dirname(shapefile))
    geometry = shp_file['geometry']
    #geometry = shp_file['geometry'].to_crs(crs = 3857).buffer(30000, cap_style = 3, join_style = 2).to_crs(crs = 4326) # Getting geometry of shape file with a 30 km buffer
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
        end_time=f"2021-01-01"
    ).drop_duplicates()

    # Adding ID column for identifying unique rows
    gedi_shots["ID"] = gedi_shots.index

        #   b. Load JRC AFC raster for AOI 2020
    geometry_buffer = geometry.to_crs(crs = 3857).buffer(60, cap_style = 3, join_style = 2).to_crs(crs = 4326) # Adding 2 pixels on each edge of the geometry (30m per JRC pixel)
    jrc_raster = load_jrc_data(*geometry_buffer.bounds.values[0], dataset="AnnualChange", years=[2020])[0]

    # 2. (DATA PREPROCESSING)
        #   a. Matching GEDI shots with JRC raster pixels, labelling with class of shot and quality
    shot_ids = gedi_shots.ID.values
    shot_numbers = gedi_shots.shot_number.values
    x_inds = arg_toptwo_nearest_centers(jrc_raster.x.data, gedi_shots.lon_lowestmode.values)
    y_inds = arg_toptwo_nearest_centers(jrc_raster.y.data, gedi_shots.lat_lowestmode.values)
    len_x = jrc_raster.x.data.shape[0]
    len_y = jrc_raster.y.data.shape[0]

    filtered_shots = []
    for shot_id, shot_number, y_ind, x_ind in zip(shot_ids, shot_numbers, y_inds, x_inds):
        class_at_shot = jrc_raster.data[y_ind[0], x_ind[0]]

        if class_at_shot > 0:
            # OLD QUALITY: Check the nine surrounding pixels.
            class_around_shot = jrc_raster.data[
                y_ind[0] - 1 : y_ind[0] + 2, x_ind[0] - 1 : x_ind[0] + 2
            ]
            nine_pixels = class_around_shot.shape == (
                3,
                3,
            ) and np.alltrue(class_around_shot > 0)

        if nine_pixels and np.alltrue(
            class_around_shot == class_at_shot
        ):
            nine_pixels_same_class = True
        else:
            nine_pixels_same_class = False

        # NEW QUALITY: Check three surrounding pixels.
        if y_ind[1] < len_y and x_ind[1] < len_x:
            p2 = jrc_raster.data[y_ind[0], x_ind[1]]
            p3 = jrc_raster.data[y_ind[1], x_ind[0]]
            p4 = jrc_raster.data[y_ind[1], x_ind[1]]

            four_pixels = p2 > 0 and p3 > 0 and p4 > 0
            if four_pixels and p2 == p3 == p4 == class_at_shot:
                four_pixels_same_class = True
            else:
                four_pixels_same_class = False
        else:
            four_pixels = False
            four_pixels_same_class = False

        if nine_pixels_same_class:
            quality = 5
        elif four_pixels_same_class and nine_pixels:
            quality = 4
        elif nine_pixels:
            quality = 3
        elif four_pixels_same_class:
            quality = 2
        elif four_pixels:
            quality = 1
        else:
            quality = 0

        if nine_pixels: # Getting all shots that have nine surrounding pixels, should be the same as the total number of gedi_shots
            filtered_shots.append(
                        (shot_id, shot_number, class_around_shot[0], class_around_shot[1], class_around_shot[2], class_around_shot.max() - class_around_shot.min(), class_at_shot, quality)
                    )

    df = pd.DataFrame([])
    buffer_60 = pd.DataFrame(df.append(filtered_shots))
    buffer_60.columns = [
        'ID', 
        'shot_number',
        'class_around_shot[0]', 
        'class_around_shot[1]', 
        'class_around_shot[2]', 
        'class_around_shot.max() - class_around_shot.min()', 
        'shot_class', 
        'quality']

    # Merging gedi_shots with buffer_60 based on both ID and shot_number
    merged = gedi_shots.merge(
        buffer_60, how = 'left', on = ['ID','shot_number'], validate = '1:1'
    ).drop(columns = ['ID','class_around_shot[0]', 'class_around_shot[1]', 'class_around_shot[2]', 'class_around_shot.max() - class_around_shot.min()'])

        #   b. Discard lower quality shots
    high_quality = merged[(merged['quality'] == 5) & (merged['degrade_flag'] == 0) & (merged['beam_type'] == 'full') & (merged['l4_quality_flag'] == 1)]

    # 3. (CALCULATION)
        #   a. Group GEDI shots by AFC class
    by_class = high_quality.groupby('shot_class')['agbd']

        #   b. Get description of AGBD values
    agbd_val = by_class.describe()
    #print(agbd_val)
    agbd_val.to_csv(os.path.join('/home/okml2/', project + '_agbd.csv'))
    return agbd_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load pixels within AOI and extract GEDI L4A AGBD values"
    )
    parser.add_argument(
        "--shapefile",
        help="Shapefile for the AOI.",
        type=str,
    )
    args = parser.parse_args()
    
    shapefile = pathlib.Path(args.shapefile)
    if not shapefile.exists():
        print("Unable to locate file {}".format(shapefile))
        exit(1)
    
    run_program(shapefile)
    print("Done")