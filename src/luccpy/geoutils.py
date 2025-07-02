import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import mapping
from sklearn.neighbors import KDTree

__all__ = [
    "clip_raster_by_shp",
    "match_coordinate_system",
    "extract_values_to_points",
]


def clip_raster_by_shp(un_img: xr.DataArray, vector_file: str, set_crs) -> xr.DataArray:
    # read mask vector by geopandas
    mask_vector = gpd.read_file(vector_file).to_crs(set_crs)
    # setting spatial reference information
    img = un_img.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    re_img = img.rio.set_crs(set_crs)
    # using mask vector, we're clipping image
    clip_img = re_img.rio.clip(mask_vector.geometry.apply(mapping), set_crs)
    return clip_img


def match_coordinate_system(inda: xr.DataArray, target: xr.DataArray, method=None) -> xr.DataArray:
    if method is None:
        method = 'nearest'
    rda = inda.squeeze()
    rds_regrid = rda.interp_like(target, method=method)
    return rds_regrid


def extract_values_to_points(point_vector_file: str, rasters: list[xr.DataArray],
                             variables: list[str]) -> pd.DataFrame:
    assert len(variables) == len(rasters), "Length of raster files must match the length of variables."

    crs = rasters[0].rio.crs
    point_gdf = gpd.read_file(point_vector_file).to_crs(crs)
    ilon = point_gdf.geometry.x.values.ravel()
    ilat = point_gdf.geometry.y.values.ravel()

    extract_df = pd.DataFrame()
    extract_df['lat'] = ilat
    extract_df['lon'] = ilon

    for var, raster in zip(variables, rasters):
        glon = raster.x.values.ravel()
        glat = raster.y.values.ravel()
        iglon, iglat = np.meshgrid(glon, glat)

        grids_lonlat = np.vstack((iglon.ravel(), iglat.ravel())).T
        tables_lonlat = np.vstack((ilon.ravel(), ilat.ravel())).T

        tree = KDTree(grids_lonlat, leaf_size=5)
        _, ind = tree.query(tables_lonlat, k=1)
        raster_values = raster.values.ravel()
        extracted_values = raster_values[ind].ravel()

        extract_df[var] = extracted_values

    return extract_df
