import os
import geopandas as gpd
import pandas as pd

import rasterio
from rasterio.merge import merge
from rasterio.plot import show
from shapely.geometry import box, Polygon, shape

from typing import Optional, Tuple, List

    # 1. DATA LOAD
# Project aoi bounds
shp_file = gpd.read_file("/home/okml2/shp_files/WLT_Laipuna/aoi_30km_buffer_unioned.shp")
xmin,ymin,xmax,ymax =  shp_file.total_bounds
project_polygon_box = box(*shp_file.total_bounds)
# Tiles
directory = "/home/okml2/Hansen_to_JRC/Ecuador"

    # 2. Check if aoi intersect with tiles
raster_list = []
no_intersect = 0

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    raster = rasterio.open(file_path)
    raster_bounds = raster.bounds
    
    left = raster_bounds.left
    bottom = raster_bounds.bottom
    right = raster_bounds.right
    top = raster_bounds.top

    raster_poly = Polygon([[left, top], [left, bottom], [right, top], [right,bottom]])
    
    if raster_poly.intersects(project_polygon_box):
        raster_list.append(file_path)
    else: no_intersect = no_intersect + 1

df = pd.DataFrame([])
raster_list_df = pd.DataFrame(df.append(raster_list))
print(no_intersect)

    # 3. Creating a mosaic array
tiles_to_mosaic = []
for file_path in raster_list:
    tile = rasterio.open(file_path)
    tiles_to_mosaic.append(tile)

mosaic, out_trans = merge(tiles_to_mosaic)

    # 4. Outputting mosaic raster
out_fp = "/home/okml2/Hansen_to_JRC_Laipuna_mosaic.tif"

out_meta = tile.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": "WGS84"
                     }
                    )

with rasterio.open (out_fp, "w", **out_meta) as dest:
    dest.write(mosaic)

