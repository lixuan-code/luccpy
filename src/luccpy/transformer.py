from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
import rioxarray as rio
import salem
import xarray as xr
from affine import Affine

__all__ = [
    "read_tif_to_ds",
    "read_wrfout_to_ds",
    "trans_tif_to_shp",
    "trans_h5_to_tif",
    "trans_grd_to_tif",
    "save_da_to_tif",
]


def _determine_input_var(var):
    if not isinstance(var, list):
        var = [var]
    return var


def _read_file_list(path, suffix):
    file_list = [file for file in os.listdir(path) if file.endswith(suffix)]

    assert len(file_list) > 0, f"No files with the suffix '{suffix}' exist in the specified directory."

    return sorted(file_list)


def read_tif_to_ds(raster_path: str, time_coords: pd.DatetimeIndex, band_name: str = "band_data") -> xr.Dataset:
    raster_files_sorted = _read_file_list(raster_path, ".tif")

    assert len(time_coords) == len(raster_files_sorted), "Length of raster files must match the length of time coords."

    raster_list = [os.path.join(raster_path, raster_file) for raster_file in raster_files_sorted]

    da_list: list[xr.DataArray] = [rio.open_rasterio(raster) for raster in raster_list]

    combined: xr.DataArray = xr.concat(da_list, dim="band")

    combined = combined.rename({"band": "time", "y": "lat", "x": "lon"})
    combined["time"] = time_coords
    combined.name = band_name

    out_dataset = combined.to_dataset()

    return out_dataset


def read_wrfout_to_ds(wrfout_file: str, varlist: list[str] = None) -> xr.Dataset:
    ds = salem.open_wrf_dataset(wrfout_file)
    ds = ds.reset_coords().drop_vars(["lat", "lon", "xtime"]).rename({"south_north": "lat", "west_east": "lon"})

    if varlist is not None:
        varlist = _determine_input_var(varlist)
        return ds[varlist]
    else:
        return ds


def trans_tif_to_shp(raster_file: str, outshap_file: str, extract_value=None, is_pvalues: bool = False) -> None:
    raster = rio.open_rasterio(raster_file).squeeze()
    raster_crs = raster.rio.crs

    rds = raster.drop("spatial_ref").drop("band")
    rds.name = "data"
    df = rds.to_dataframe().reset_index().dropna()

    if extract_value is not None:
        vaild_df = df[df["data"] == extract_value]
    elif is_pvalues:
        vaild_df = df[df["data"] < 0.05]
    else:
        vaild_df = df.copy()

    ilon, ilat = vaild_df.x.values, vaild_df.y.values

    ser = pd.Series([f'POINT ({ii} {jj})' for ii, jj in list(zip(ilon, ilat))], name="wkd")
    gs = gpd.GeoSeries.from_wkt(ser)
    gdf = gpd.GeoDataFrame(data=vaild_df, geometry=gs, crs=raster_crs)

    gdf.to_file(outshap_file)

    print(f"Transformation of {raster_file} completed and saved as {outshap_file}!")


def trans_h5_to_tif(in_hdf_file: str, out_raster_file: str, geo_params: list[float]) -> None:
    xds = xr.open_dataset(in_hdf_file)

    top_left_x, x_res, x_affine_angle, top_left_y, y_affine_angle, y_res = geo_params
    transform = Affine(x_res, x_affine_angle, top_left_x, y_affine_angle, y_res, top_left_y)

    xds = xds.rio.set_spatial_dims(x_dim="phony_dim_1", y_dim="phony_dim_0", inplace=True)
    xds = xds.rio.write_transform(transform, inplace=True)

    xds.rio.to_raster(out_raster_file)

    print(f"Transformation of {in_hdf_file} completed and saved as {out_raster_file}!")


def trans_grd_to_tif(grd_file: str, tif_file: str) -> None:
    grd_image = xr.open_dataarray(grd_file, engine="rasterio").squeeze()
    grd_image.rio.to_raster(tif_file)
    print(f"Transformation of {grd_file} completed and saved as {tif_file}!")


def save_da_to_tif(da: xr.DataArray, ofile: str, crs="epsg:4326", is_wrfout: bool = False) -> None:
    if is_wrfout:
        wrf_crs = '+proj=lcc +lat_0=43.5 +lon_0=126 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +datum=WGS84 +units=m ' \
                  '+no_defs=True '
        da_set_crs = da.rio.write_crs(wrf_crs)
    else:
        da_set_crs = da.rio.write_crs(crs)

    da_set_crs.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da_set_crs.rio.to_raster(ofile)
    print(f"save {ofile} done!")
