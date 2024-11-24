from __future__ import annotations

from typing import Union

import xarray as xr

from .calc_deterministic import (
    _calc_linregress,
    _calc_mk_test,
    _calc_bi_pcorr_rp,
    _calc_bi_corr_rp,
    _calc_corr_r,
    _linslope_spatial,
    _linslope_pval_spatial,
    _calc_sensity_spatial,
    _calc_maxshap_spatial
)

__all__ = [
    "calc_linregress",
    "calc_mk_test",
    "calc_bi_corr_rp",
    "calc_bi_pcorr_rp",
    "calc_corr_r",
    "linslope_spatial",
    "linslope_pval_spatial",
    "calc_sensity_spatial",
    "calc_maxshap_spatial"
]


def _stack_input_if_needed(data_list:list[xr.DataArray], dim):
    if len(dim) > 1:
        new_dim = "_".join(dim)
        new_data_list = [data.stack(**{new_dim: dim}) for data in data_list]
    else:
        new_dim = dim[0]
        new_data_list = data_list
    return new_data_list, new_dim


def _determine_input_core_dims(dim, data_list_num):
    if not isinstance(dim, list):
        dim = [dim]

    input_core_dims = [dim] * data_list_num

    return input_core_dims


def calc_linregress(data: xr.DataArray, dim: str = "time") -> xr.Dataset:
    # Calculate linear regression for the given data

    res_dataarray = xr.apply_ufunc(
        _calc_linregress,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": 1},
            "allow_rechunk": True,
        },
    )

    res_dataset = xr.Dataset(
        data_vars={
            "slope": res_dataarray[..., 0],
            "pvalue": res_dataarray[..., 1],
        }
    )

    return res_dataset


def calc_mk_test(data: xr.DataArray, dim: str = "time", alpha: float = 0.05) -> xr.Dataset:
    # Apply Mann-Kendall test to the data for trend detection

    res_dataarray = xr.apply_ufunc(
        _calc_mk_test,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"alpha": alpha,},
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": 1},
            "allow_rechunk": True,
        },
    )

    res_dataset = xr.Dataset(
        data_vars={
            "trend": res_dataarray[..., 0],
            "pvalue": res_dataarray[..., 1],
            "slope": res_dataarray[..., 2],
        }
    )

    return res_dataset


def calc_bi_corr_rp(a: xr.DataArray, b: xr.DataArray, dim=None,
                    method: str = "pearson", is_skipna: bool = False) -> xr.Dataset:
    # Calculate bi-variate correlation based on the specified method

    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed([a, b], dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    res_dataarray = xr.apply_ufunc(
        _calc_bi_corr_rp,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"method": method,
                "is_skipna": is_skipna},
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": 1},
            "allow_rechunk": True,
        },
    )

    res_dataset = xr.Dataset(
        data_vars={
            "r": res_dataarray[..., 0],
            "pvalue": res_dataarray[..., 1],
        }
    )

    return res_dataset


def calc_bi_pcorr_rp(data_list: list[xr.DataArray], variables: list[str],
                     x_name: str, y_name: str, dim=None, is_skipna: bool = False) -> xr.Dataset:
    # Calculate bi-variate partial correlation

    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    res_dataarray = xr.apply_ufunc(
        _calc_bi_pcorr_rp,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "x_name": x_name,
                "y_name": y_name,
                "is_skipna": is_skipna, },
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": 1},
            "allow_rechunk": True,
        },
    )

    res_dataset = xr.Dataset(
        data_vars={
            "r": res_dataarray[..., 0],
            "pvalue": res_dataarray[..., 1],
        }
    )

    return res_dataset


def calc_corr_r(data_list: list[xr.DataArray], variables: list[str], dim=None,
                method: str = "pearson", is_pcorr: bool = False, is_skipna: bool = False) -> xr.Dataset:
    # Calculate correlation (pearson or partial) for the given variables
    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    num_of_variables = len(variables) - 1

    res_dataarray = xr.apply_ufunc(
        _calc_corr_r,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "method": method,
                "is_pcorr": is_pcorr,
                "is_skipna": is_skipna, },
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": num_of_variables},
            "allow_rechunk": True,
        }
    )

    res_dataset = xr.Dataset(
        data_vars={
            var_name + "_corr": res_dataarray[..., ii] for ii, var_name in enumerate(variables[1:])
        }
    )

    return res_dataset


def linslope_spatial(data_list: list[xr.DataArray], variables: list[str],
                     y_name: str, dim: Union[str, list[str]] = "time", is_skipna: bool = False) -> xr.Dataset:
    # Calculate linear slope for spatial data
    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    num_of_variables = len(variables) - 1

    res_dataarray = xr.apply_ufunc(
        _linslope_spatial,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "y_name": y_name,
                "is_skipna": is_skipna, },
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": num_of_variables},
            "allow_rechunk": True,
        }
    )

    res_dataset = xr.Dataset(
        data_vars={
            var_name + "_slpoe": res_dataarray[..., ii] for ii, var_name in enumerate(variables[1:])
        }
    )

    return res_dataset


def linslope_pval_spatial(data_list: list[xr.DataArray], variables: list[str],
                          y_name: str, dim: Union[str, list[str]] = "time", is_skipna: bool = False) -> xr.Dataset:
    # Calculate p-values for linear regression slopes in spatial data

    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    num_of_variables = len(variables) - 1

    res_dataarray = xr.apply_ufunc(
        _linslope_pval_spatial,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "y_name": y_name,
                "is_skipna": is_skipna, },
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": num_of_variables},
            "allow_rechunk": True,
        }
    )

    res_dataset = xr.Dataset(
        data_vars={
            var_name + "_pval": res_dataarray[..., ii] for ii, var_name in enumerate(variables[1:])
        }
    )

    return res_dataset


def calc_sensity_spatial(data_list: list[xr.DataArray], variables: list[str],
                         y_name: str, dim: Union[str, list[str]] = "time", is_skipna: bool = False) -> xr.Dataset:
    # Calculate sensitivity for spatial data using Random Forest
    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    num_of_variables = len(variables) - 1

    res_dataarray = xr.apply_ufunc(
        _calc_sensity_spatial,
        *data_list,
        input_core_dims=input_core_dims,
        output_core_dims=[["parameter"]],
        output_dtypes=["float64"],
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "y_name": y_name,
                "is_skipna": is_skipna, },
        dask_gufunc_kwargs={
            "output_sizes": {"parameter": num_of_variables},
            "allow_rechunk": True,
        }
    )

    res_dataset = xr.Dataset(
        data_vars={
            var_name + "_sensity": res_dataarray[..., ii] for ii, var_name in enumerate(variables[1:])
        }
    )

    return res_dataset


def calc_maxshap_spatial(data_list: list[xr.DataArray], variables: list[str],
                         y_name: str, dim: Union[str, list[str]] = "time", is_skipna: bool = False) -> xr.Dataset:
    # Calculate the maximum SHAP value for spatial data using Random Forest
    if isinstance(dim, str):
        dim = [dim]

    data_list, dim = _stack_input_if_needed(data_list, dim)

    input_core_dims = _determine_input_core_dims(dim, len(data_list))

    res_dataarray = xr.apply_ufunc(
        _calc_maxshap_spatial,
        *data_list,
        input_core_dims=input_core_dims,
        dask="parallelized",
        vectorize=True,
        kwargs={"variables": variables,
                "y_name": y_name,
                "is_skipna": is_skipna, },
    )

    res_dataarray.name = "maxshap"
    res_dataset = res_dataarray.to_dataset()

    return res_dataset
