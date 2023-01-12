import logging

import geopandas as gpd
import numba
import numpy as np
import pandas as pd

from src.data.gedi_database import GediDatabase
from src.data.jrc_loading import load_jrc_data
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

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

# 1. (DATA LOAD)
#   a. Load GEDI data for AOI and year 2020
shp_file = gpd.read_file("/home/okml2/biomass-recovery/shp_files/WLT_Chaco_Pantanal/aoi.shp")
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

#def overlay_gedi_shots_and_jrc_raster(
#    gedi_shots, jrc_raster, keep_distribution: bool = False
#):

shot_ids = gedi_shots.shot_number.values
x_inds = arg_toptwo_nearest_centers(jrc_raster.x.data, gedi_shots.lon_lowestmode.values)
y_inds = arg_toptwo_nearest_centers(jrc_raster.y.data, gedi_shots.lat_lowestmode.values)
len_x = jrc_raster.x.data.shape[0]
len_y = jrc_raster.y.data.shape[0]
#logger.info("Filtering shots")
# This loop is quite fast in my test, does not need numba.
filtered_shots = []
for shot_id, y_ind, x_ind in zip(shot_ids, y_inds, x_inds):
    class_at_shot = jrc_raster.data[y_ind[0], x_ind[0]]

    if class_at_shot > 0:
        # OLD QUALITY: Check the nine surrounding pixels.
        class_around_shot = jrc_raster.data[
            y_ind[0] - 1 : y_ind[0] + 2, x_ind[0] - 1 : x_ind[0] + 2
        ]
        nine_recovering = class_around_shot.shape == (
            3,
            3,
        ) and np.alltrue(class_around_shot > 0)

    if nine_recovering and np.alltrue(
        class_around_shot == class_at_shot
    ):
        nine_recovering_same_class = True
    else:
        nine_recovering_same_class = False

    # NEW QUALITY: Check three surrounding pixels.
    if y_ind[1] < len_y and x_ind[1] < len_x:
        p2 = jrc_raster.data[y_ind[0], x_ind[1]]
        p3 = jrc_raster.data[y_ind[1], x_ind[0]]
        p4 = jrc_raster.data[y_ind[1], x_ind[1]]

        four_recovering = p2 > 0 and p3 > 0 and p4 > 0
        if four_recovering and p2 == p3 == p4 == class_at_shot:
            four_recovering_same_class = True
        else:
            four_recovering_same_class = False
    else:
        four_recovering = False
        four_recovering_same_class = False

    if nine_recovering_same_class:
        quality = 5
    elif four_recovering_same_class and nine_recovering:
        quality = 4
    elif nine_recovering:
        quality = 3
    elif four_recovering_same_class:
        quality = 2
    elif four_recovering:
        quality = 1
    else:
        quality = 0

    if nine_recovering:
        filtered_shots.append(
            (shot_id, class_around_shot[0], class_around_shot[1], class_around_shot[2], class_around_shot.max() - class_around_shot.min(), class_at_shot, quality)
        )

print(filtered_shots)
    # Return the shots
#return filtered_shots

df = pd.DataFrame([])
df = df.append(filtered_shots)
df.columns = ['shot_id', 'class_around_shot[0]', 'class_around_shot[1]', 'class_around_shot[2]', 'class_around_shot.max() - class_around_shot.min()', 'class_at_shot', 'quality']
print(df)