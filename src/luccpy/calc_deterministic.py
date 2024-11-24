import warnings
from typing import Union

import numpy as np
import pandas as pd
import pingouin as pg
import pymannkendall as mk
import shap
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TheilSenRegressor

warnings.filterwarnings("ignore")

__all__ = [
    "_calc_linregress",
    "_calc_mk_test",
    "_calc_bi_pcorr_rp",
    "_calc_bi_corr_rp",
    "_calc_corr_r",
    "_linslope_spatial",
    "_linslope_pval_spatial",
    "_calc_sensity_spatial",
    "_calc_maxshap_spatial",
]


def _has_any_invalid_data(input_data: list, is_skipna: bool = False) -> bool:
    if is_skipna:
        has_invalid: bool = np.isnan(np.array(input_data)).all()
    else:
        has_invalid: bool = np.isnan(np.array(input_data)).any()

    return has_invalid


def _calc_linregress(data: np.ndarray) -> np.ndarray:
    # Calculate linear regression for the given data
    x_data = np.arange(0, data.shape[0])
    linregress_result = linregress(x_data, data)
    slope = linregress_result.slope
    pvalue = linregress_result.pvalue
    return np.array([slope, pvalue])


def _calc_mk_test(data: np.ndarray, alpha: float, is_skipna: bool = False) -> np.ndarray:
    # Apply Mann-Kendall test to the data for trend detection

    if _has_any_invalid_data([data], is_skipna=is_skipna):
        return np.array([np.nan, np.nan, np.nan])

    mk_test_result = mk.original_test(data, alpha=alpha)

    trend = mk_test_result.trend
    if trend == "increasing":
        trend = 1
    elif trend == "decreasing":
        trend = -1
    elif trend == "no trend":
        trend = 0
    else:
        raise ValueError("Error `trend` type.")

    p = mk_test_result.p
    slope = mk_test_result.slope
    return np.array([trend, p, slope])


def _calc_bi_corr_rp(*args, method: str, is_skipna: bool = False) -> np.ndarray:
    # Calculate bi-variate correlation based on the specified method
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.array([np.nan, np.nan])

    a, b = args
    df: pd.DataFrame = pd.DataFrame(dict(vara=a,varb=b))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.array([np.nan, np.nan])

    corr_result = pg.corr(df['vara'].values, df['varb'].values, method=method)
    r = corr_result["r"]
    p_val = corr_result["p-val"]

    return np.array([r, p_val]).ravel()


def _calc_bi_pcorr_rp(*args, variables: list[str], x_name: str, y_name: str,
                      is_skipna: bool = False) -> np.ndarray:
    # Calculate bi-variate partial correlation
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.array([np.nan, np.nan])

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.array([np.nan, np.nan])

    covar = [col for col in variables if col not in [x_name, y_name]]
    partial_corr_result = pg.partial_corr(data=df, x=x_name, y=y_name, covar=covar)

    r = partial_corr_result["r"]
    p_val = partial_corr_result["p-val"]

    return np.array([r, p_val]).ravel()


def _calc_corr_r(*args, variables: list[str], method: str,
                 is_pcorr: bool, is_skipna: bool = False) -> np.ndarray:
    # Calculate correlation (pearson or partial) for the given variables
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.full(len(variables) - 1, np.nan)

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.full(len(variables) - 1, np.nan)

    if is_pcorr:
        cor = df.pcorr()[variables[0]]
    else:
        cor = df.corr(method=method)[variables[0]]

    return np.array(cor)[1:]


def _linslope_spatial(*args, variables: list[str], y_name: str,
                      is_skipna: bool = False) -> np.ndarray:
    # Calculate linear slope for spatial data
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.full(len(variables) - 1, np.nan)

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.full(len(variables) - 1, np.nan)

    x_name = [col for col in variables if col not in [y_name]]
    res = pg.linear_regression(df[x_name], df[y_name], coef_only=True)

    return res[1:]


def _linslope_pval_spatial(*args, variables: list[str], y_name: str,
                           is_skipna: bool = False) -> np.ndarray:
    # Calculate p-values for linear regression slopes in spatial data
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.full(len(variables) - 1, np.nan)

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.full(len(variables) - 1, np.nan)

    x_name = [col for col in variables if col not in [y_name]]
    lm = pg.linear_regression(df[x_name], df[y_name])

    res = lm.set_index("names").pval  # x_name

    return np.array(res)[1:]


def _calc_sensity_spatial(*args, variables: list[str], y_name: str,
                          is_skipna: bool = False) -> np.ndarray:
    # Calculate sensitivity for spatial data using Random Forest
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.full(len(variables) - 1, np.nan)

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.full(len(variables) - 1, np.nan)

    x_name = [col for col in variables if col not in [y_name]]
    rf = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=1, random_state=42)
    rf.fit(df[x_name].values, df[y_name])

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df[x_name])

    coef = np.full(len(x_name), np.nan)

    for v in range(len(x_name)):
        y = shap_values[:, v]
        x = df[x_name[v]]
        x = x.values.reshape(-1, 1)
        x_new = x[~np.isnan(y)]
        y_new = y[~np.isnan(y)]

        if all([not np.isnan(x_new).any(), not np.isinf(x_new).any(),
                np.any(x_new != 0), np.any(y_new != 0)]):
            tel = TheilSenRegressor().fit(x_new, y_new)
            coef[v] = tel.coef_

    return np.array([coef])


def _calc_maxshap_spatial(*args, variables: list[str], y_name: str,
                          is_skipna: bool = False) -> Union[np.ndarray, float]:
    # Calculate the maximum SHAP value for spatial data using Random Forest
    if _has_any_invalid_data([args], is_skipna=is_skipna):
        return np.nan

    df: pd.DataFrame = pd.DataFrame(dict(zip(variables, args)))

    if is_skipna:
        df = df.dropna()
        if df.shape[0] < 3:
            return np.nan

    x_name: list[str] = [col for col in variables if col not in [y_name]]
    rf: RandomForestRegressor = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=1, random_state=42)
    rf.fit(df[x_name].values, df[y_name])

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df[x_name])
    shap_sum_values = np.abs(shap_values).sum(axis=0)
    max_shap = np.argmax(shap_sum_values)

    return np.array([max_shap])
