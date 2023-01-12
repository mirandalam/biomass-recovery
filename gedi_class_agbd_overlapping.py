#import logging

import geopandas as gpd
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data.gedi_database import GediDatabase
from src.data.jrc_loading import load_jrc_data
#from src.utils.logging import get_logger

#logger = get_logger(__name__)
#logger.setLevel(logging.DEBUG)

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
shp_file = gpd.read_file("/home/okml2/biomass-recovery/shp_files/AIDER_Tambopata/aoi_30km_buffer_right.shp")
#geometry = shp_file['geometry'].to_crs(crs = 3857).buffer(30000, cap_style = 3, join_style = 2).to_crs(crs = 4326) # Getting geometry of shape file with a 30 km buffer
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
    end_time=f"2021-01-01"
).drop_duplicates()

# Adding ID column for identifying unique rows
gedi_shots["ID"] = gedi_shots.index

#   b. Load JRC AFC raster for AOI 2020
geometry_buffer = geometry.to_crs(crs = 3857).buffer(60).to_crs(crs = 4326) # Adding 2 pixels on each edge of the geometry (30m per JRC pixel)
jrc_raster = load_jrc_data(*geometry_buffer.bounds.values[0], dataset="AnnualChange", years=[2020])[0]


#def overlay_gedi_shots_and_jrc_raster(
#    gedi_shots, jrc_raster, keep_distribution: bool = False
#):

shot_ids = gedi_shots.ID.values
shot_numbers = gedi_shots.shot_number.values
x_inds = arg_toptwo_nearest_centers(jrc_raster.x.data, gedi_shots.lon_lowestmode.values)
y_inds = arg_toptwo_nearest_centers(jrc_raster.y.data, gedi_shots.lat_lowestmode.values)
len_x = jrc_raster.x.data.shape[0]
len_y = jrc_raster.y.data.shape[0]
#logger.info("Filtering shots")
# This loop is quite fast in my test, does not need numba.
filtered_shots = []
no_pixel =[]
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

    if nine_pixels: # Getting all shots have nine surrounding pixels, should be the same as the total number of gedi_shots
        filtered_shots.append(
                    (shot_id, shot_number, class_around_shot[0], class_around_shot[1], class_around_shot[2], class_around_shot.max() - class_around_shot.min(), class_at_shot, quality)
                )
    else:
        no_pixel.append(
                    (shot_id, shot_number, class_at_shot, quality)
                )
    # Return the shots
#return filtered_shots
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

df2 = pd.DataFrame([])
no_pixel_60 = pd.DataFrame(df2.append(no_pixel))
no_pixel_60.columns = [
    'ID', 
    'shot_number',
    'shot_class', 
    'quality']

#   Merging gedi_shots with buffer_60 based on both ID and shot_number
merged = gedi_shots.merge(
    buffer_60, how = 'left', on = ['ID','shot_number'], validate = '1:1'
).drop(columns = ['ID','class_around_shot[0]', 'class_around_shot[1]', 'class_around_shot[2]', 'class_around_shot.max() - class_around_shot.min()'])


#For overlapping tiles with left and right shape files
test_merge = gedi_shots.merge(buffer_60, how = 'left', on = ['ID','shot_number'], indicator = True).drop(columns = ['ID','class_around_shot[0]', 'class_around_shot[1]', 'class_around_shot[2]', 'class_around_shot.max() - class_around_shot.min()'])


quality1 = no_pixel_60[(no_pixel_60['quality']==1)]
quality2 = no_pixel_60[(no_pixel_60['quality']==2)]

#Right side
geo_no_pixel_right = test_merge[(test_merge['_merge'] == 'left_only')]['geometry']
nine_pixel_right = test_merge[(test_merge['_merge'] == 'both')]['geometry']
geo_no_pixel_right.to_file('/home/okml2/Right_Madre_no_nine_pix.shp')
nine_pixel_right.to_file('/home/okml2/Right_Madre_with_nine_pix.shp')
#test_merge.to_csv(r'/home/okml2/Madre_pixels/Madre_test_merge_unioned.csv')

#left side
geo_no_pixel_left = test_merge[(test_merge['_merge'] == 'left_only')]['geometry']
nine_pixel_left = test_merge[(test_merge['_merge'] == 'both')]['geometry']
geo_no_pixel_left.to_file('/home/okml2/Left_Madre_no_nine_pix.shp')
nine_pixel_left.to_file('/home/okml2/Left_Madre_with_nine_pix.shp')

#merged = gedi_shots.merge(
#    buffer_60, how = 'inner', on = None, left_index = True, right_index = True
#    buffer_60, how = 'inner', on = 'shot_number'
#    ).drop(columns = ['class_around_shot[0]', 'class_around_shot[1]', 'class_around_shot[2]', 'class_around_shot.max() - class_around_shot.min()']
#    ).drop_duplicates()

merged = test_merge[(test_merge['_merge'] == 'both')]

#   Discard lower quality shots
high_quality = merged[(merged['quality'] == 5) & (merged['degrade_flag'] == 0) & (merged['beam_type'] == 'full') & (merged['l4_quality_flag'] == 1)]

    # 3. (CALCULATION)
    #   a. Group GEDI shots by AFC class
by_class = high_quality.groupby('shot_class')['agbd']

    #   b. Get description of AGBD values
agbd_val = by_class.describe()
print(agbd_val)
"""
# in dev:
means = []
hists = []
class_ids = high_quality["shot_class"].unique()
for class_id in class_ids:
    subset = high_quality[high_quality["shot_class"] == class_id]
    #   b. Take mean AGBD for each class value
    means.append(subset["agbd"].mean())
    #   c. Take histogram dist of AGBD for each class value
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    hists.append(np.histogram(subset["agbd"]))
class_means = pd.DataFrame(means)
#???plt.hist(hists)


class1 = high_quality[high_quality["shot_class"] == 1]
plt.hist(np.histogram(class1['agbd']), bins = 5)

class2 = high_quality[high_quality["shot_class"] == 2]
plt.hist(np.histogram(class2['agbd']))

class3 = high_quality[high_quality["shot_class"] == 3]['agbd']
a=np.histogram(class3)
plt.hist(np.histogram(class3))


sns.set_theme()
sns.set_style()
sns.set_context('paper')
sns.displot(high_quality[high_quality["shot_class"] == <class>]['agbd'], palette='deep')

# 4. (OUTPUT)
for class_id in class_ids:
    #   a. Save k values: mean for each of k classes

    #   b. Save k histograms: histogram for each of k classes

    pass

    # 4. (OUTPUT)
agbd_val.to_csv(r'/home/okml2/biomass-recovery/CIF_Alto_Mayo_agbd.csv')
Right_Madre_quality_shots.to_csv (r'/home/okml2/AIDER_Madre_de_Dios_30km_right_quality_shots.csv')
Left_Madre_quality_shots.to_csv (r'/home/okml2/AIDER_Madre_de_Dios_30km_left_quality_shots.csv')
"""
Left_Tambopata_quality_shots = high_quality.drop(columns = ['_merge'])
Right_Tambopata_quality_shots = high_quality.drop(columns = ['_merge'])

frames = [Left_Tambopata_quality_shots, Right_Tambopata_quality_shots]
combined = pd.concat(frames)
combined.to_csv(r'/home/okml2/AIDER_Tambopata_30km_quality_shots_combined.csv')

by_class = combined.groupby('shot_class')['agbd']

    #   b. Get description of AGBD values
agbd_val = by_class.describe()
print(agbd_val)

ini = gedi_shots[(gedi_shots['degrade_flag'] == 0) & (gedi_shots['beam_type'] == 'full') & (gedi_shots['l4_quality_flag'] == 1)]