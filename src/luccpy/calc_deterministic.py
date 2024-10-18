import numpy as np
import pandas as pd
import pingouin as pg
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import linregress
import pymannkendall as mk

import warnings


def suppress_warnings(msg=None):
    """Catch warnings with message msg."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", msg)
        yield


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

def _all_valid(arrays):
    # Check if all elements in the input arrays are valid (i.e., not NaN)
    return not np.isnan(np.array(arrays)).any()


def _calc_linregress(data):
    # Calculate linear regression for the given data
    x_data = np.arange(0, data.shape[0])
    linregress_result = linregress(x_data, data)
    slope = linregress_result.slope
    pvalue = linregress_result.pvalue
    return np.array([slope, pvalue])


def _calc_mk_test(data, alpha):
    # Apply Mann-Kendall test to the data for trend detection

    if _all_valid([data]):

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
    return np.array([np.nan, np.nan, np.nan])


def _calc_bi_corr_rp(*args, method):
    # Calculate bi-variate correlation based on the specified method
    if _all_valid([args]):
        a, b = args 
        corr_result = pg.corr(a, b, method=method)
        r = corr_result["r"]
        p_val = corr_result["p-val"]

        return np.array([r, p_val]).ravel()
    return np.array([np.nan, np.nan])


def _calc_bi_pcorr_rp(*args, variables, x_name, y_name):
    # Calculate bi-variate partial correlation
    if _all_valid([args]):
        df = pd.DataFrame(dict(zip(variables, args)))
        covar = [col for col in variables if col not in [x_name, y_name]]
        partial_corr_result = pg.partial_corr(data=df, x=x_name, y=y_name, covar=covar)

        r = partial_corr_result["r"]
        p_val = partial_corr_result["p-val"]

        return np.array([r, p_val]).ravel()
    return np.array([np.nan, np.nan])


def _calc_corr_r(*args, variables, method, is_pcorr):
    # Calculate correlation (pearson or partial) for the given variables
    if _all_valid([args]):

        df = pd.DataFrame(dict(zip(variables, args)))

        if is_pcorr:
            cor = df.pcorr()[variables[0]]
        else:
            cor = df.corr(method=method)[variables[0]]

        return np.array(cor)[1:]
    
    return np.full(len(variables)-1,np.nan)


def _linslope_spatial(*args, variables, y_name):
    # Calculate linear slope for spatial data
    if _all_valid([args]):

        df = pd.DataFrame(dict(zip(variables, args)))
        x_name = [col for col in variables if col not in [y_name]]
        res = pg.linear_regression(df[x_name], df[y_name], coef_only=True)

        return res[1:]
    
    return np.full(len(variables)-1,np.nan)


def _linslope_pval_spatial(*args, variables, y_name):
    # Calculate p-values for linear regression slopes in spatial data
    if _all_valid([args]):

        df = pd.DataFrame(dict(zip(variables, args)))
        x_name = [col for col in variables if col not in [y_name]]
        lm = pg.linear_regression(df[x_name], df[y_name])

        res = lm.set_index("names").pval  # x_name

        return np.array(res)[1:]
    
    return np.full(len(variables)-1,np.nan)


def _calc_sensity_spatial(*args, variables, y_name):
    # Calculate sensitivity for spatial data using Random Forest
    if _all_valid([args]):

        df = pd.DataFrame(dict(zip(variables, args)))
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
    
    return np.full(len(variables)-1,np.nan)


def _calc_maxshap_spatial(*args, variables, y_name):
    # Calculate the maximum SHAP value for spatial data using Random Forest
    if _all_valid([args]):

        df = pd.DataFrame(dict(zip(variables, args)))
        x_name = [col for col in variables if col not in [y_name]]
        rf = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=1, random_state=42)
        rf.fit(df[x_name].values, df[y_name])

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(df[x_name])
        shap_sum_values = np.abs(shap_values).sum(axis=0)
        maxshap = np.argmax(shap_sum_values)

        return np.array([maxshap])
    
    return np.nan
