from scipy.stats import t, boxcox_normmax
from statsmodels.formula.api import ols
from pandas import DataFrame
from helper import daytrunc
from numpy import abs as nabs
from numpy import any as nany
from statsmodels.api import OLS, add_constant
from numpy import array, arange, where, log, exp
from numpy import mean, tile, arange, std, column_stack
from numpy import sqrt, tile, ceil, ones, where, shape, zeros
from numpy import roll, diag, delete, dot, apply_along_axis, isnan


def BCLambda(data):
    """
    Gives optimal value of Box-Cox power transformation index
    lambda

    Parameters
    ----------
    data: 1D numpy array
        data in 1D numpy array

    Returns
    -------
    blambda: float
        Box-Cox power transformation index
    """
    blambda = 0
    data_in = array(data, dtype=float)
    data_in = abs(data_in)
    blambda = boxcox_normmax(data_in, brack=(-1.0, 2.0))
    if blambda > 2.0:
        blambda = 2.0
    if blambda < -1.0:
        blambda = -1.0
    return blambda


def YeoJohn(x, blambda):
    """
    Does Yeo-Johnson power transformation on data x with parameter blambda

    Parameters
    ----------
    x: 1D array
        one dimensional array of data
    blambda: float number
        power transformation parameter

    Returns
    -------
    out: 1D array
        power transformation result
    """
    out = array(x)
    if nany(x >= 0.0) and (blambda != 0.0):
        out[x >= 0.0] = ((((out[x >= 0.0] + 1.0) ** (blambda)) - 1.0) / blambda)
    if (nany(x >= 0.0)) and (blambda == 0.0):
        out[x >= 0.0] = log(out[x >= 0.0] + 1.0)
    if (nany(x < 0.0)) and (blambda != 2.0):
        out[x < 0.0] = -((((1 - out[x < 0]) ** (2.0 - blambda)) - 1.0) / (2.0 - blambda))
    if (nany(x < 0.0)) and (blambda == 2.0):
        out[x < 0.0] = - log(1.0 - out[x < 0.0])
    return out


def YeoJohnInv(x, blambda):
    """
    Does inverse Yeo-Johnson power transformation on data x with parameter blambda

    Parameters
    ----------
    x: 1D array
        one dimensional array of data
    blambda: float number
        power transformation parameter

    Returns
    -------
    out: 1D array
        inverse power transformation result
    """
    out = array(x)
    if (nany(x >= 0.0)) and (blambda != 0.0):
        out[x >= 0.0] = ((out[x >= 0.0] * blambda + 1.0) ** (1.0 / blambda)) - 1.0
    if (nany(x >= 0.0)) and (blambda == 0.0):
        out[x >= 0.0] = exp(out[x >= 0.0]) - 1.0
    if (nany(x < 0.0)) and (blambda != 2.0):
        out[x < 0.0] = 1.0 - ((1.0 - out[x < 0.0] * (2.0 - blambda)) ** (1.0 / (2.0 - blambda)))
    if (nany(x < 0.0)) and (blambda == 2.0):
        out[x < 0.0] = 1.0 - exp(-out[x < 0.0])
    return out

def StepY(x, sensitivity="low"):
    """
    calculating confidence interval for mean value of
    x, assuming that x has normal distribution or we have
    random walk process

    Parameters
    ----------
    x: array
        one dimensional array of time series

    sensitivity: string
        confidence level, i.e. sensitivity, possible values are "low", "medium", "high"

    Returns
    -------
    lower: numeric
        lower bound of confidence interval

    upper: numeric
        upper bound of confidence interval

    """
    if ((sensitivity != "low") & (sensitivity != "medium") & (sensitivity != "high")):
        sensitivity = "low"
    if (sensitivity == "low"):
        pvalue = 99.0
    if (sensitivity == "medium"):
        pvalue = 95.0
    if (sensitivity == "high"):
        pvalue = 90.0
    xin = array(x)  # casting x to numpy array for consistency
    n = len(xin)
    meanx = mean(xin)
    lower = None  # initial value of lower,upper are set to None
    upper = None
    s = std(xin, ddof=1)  # standard deviation of xin
    tfrac = t.ppf(0.5 - pvalue / 200.0, n - 1.0)  # stats.t.ppf is equivalent to qt() for R: quantile function for t-distribution
    w = tfrac * s * sqrt(1.0 + 1.0 / n)  # confidence interval half width
    lower = meanx + w  # lower bound
    upper = meanx - w  # upper bound
    return lower, upper


def TSLM(itseries, obsi, trend_only=False):
    """
    Linear regression model for time series with seasonal and trend components

    Parameters
    ----------
    itseries: numpy or other 1D array of numbers
        time series like vector with fixed step of observations

    obsi: integer
        frequency i.e. number of observations in unit of time

    trend_only: boolean
        should be seasonal component attached to a model or just
        trend will be predicted

    Returns
    -------
    lm_model: OLS regression model wrapper
        linear regresion model parameters and results
    """
    nobsi = len(itseries)
    if not trend_only:
        season = tile(arange(obsi, dtype=float), ceil(float(nobsi) / float(obsi)))[:nobsi] + 1
        trend = arange(nobsi, dtype=float) + 1
        col_names = ['data', 'trend', 'season']
        data_matrix = column_stack((itseries, trend, season))
        data_matrix = DataFrame(data_matrix, columns=col_names)
        lm_model = ols(formula='data ~ trend + C(season)', data=data_matrix).fit(method="qr")
    else:
        trend = arange(nobsi, dtype=float) + 1
        col_names = ['data', 'trend']
        data_matrix = column_stack((itseries, trend))
        data_matrix = DataFrame(data_matrix, columns=col_names)
        lm_model = ols(formula='data ~ trend', data=data_matrix).fit(method="qr")
    return lm_model


def PredictLM(lmfit, h=None, sensitivity="low", trend_only=False):
    """
    Daily prediction for linear regression model for time series

    Parameters
    ----------
    lmfit: OLS model with parameters
        linear regression model wrapper

    h: integer
        number of elements to be predicted

    sensitivity: string
        confidence level, i.e. sensitivity, possible values are "low", "medium", "high"

    trend_only: boolean
        should be seasonal component attached to a model or just
        trend will be predicted

    Returns
    -------
    pred: numpy array
        daily prediction of the model with upper and lower
        confidence intervals of 99% reliablity
    """
    if ((sensitivity != "low") & (sensitivity != "medium") & (sensitivity != "high")):
        sensitivity = "low"
    if (sensitivity == "low"):
        pvalue = 99.0
    if (sensitivity == "medium"):
        pvalue = 95.0
    if (sensitivity == "high"):
        pvalue = 90.0
    nobsi, nfreq = (shape(lmfit.model.exog) - array((0, 1)))
    if h is None:
        h = nfreq
    conf1 = (100.0 - 50.0) / 200.0
    conf2 = (100.0 - pvalue) / 200.0
    if not trend_only:
        season_end = nobsi - (nfreq * int(nobsi / nfreq)) + 1
        season_num = int(ceil(float(h) / float(nfreq)))
        new_season = arange(nfreq)
        new_season = tile(new_season, season_num)
        new_season = roll(new_season, -season_end) + 1
        new_season = new_season[0:h]
        new_trend = arange(h) + nobsi
        newdata = column_stack((new_trend, new_season))
        newdata = DataFrame(newdata, columns=['trend', 'season'])
        pred = array(lmfit.predict(newdata))
        pred_intercept = ones(h)
        pred_ses = diag(ones(nfreq))
        if season_num > 1:
            pred_ses = tile(pred_ses, (season_num, 1))
        new_season = new_season - 1
        pred_ses = pred_ses[new_season, :]
        rem_col = where(pred_ses[0, :] == 1.0)
        pred_ses = delete(pred_ses, rem_col, 1)
        new_exog = column_stack((pred_intercept, pred_ses, new_trend))
    else:
        new_trend = arange(h) + nobsi
        newdata = new_trend
        newdata = DataFrame(newdata, columns=['trend'])
        pred = array(lmfit.predict(newdata))
        pred_intercept = ones(h)
        new_exog = column_stack((pred_intercept, new_trend))
    predvar = lmfit.mse_resid + (new_exog * dot(lmfit.cov_params(), new_exog.T).T).sum(1)
    predstd = sqrt(predvar)
    nvar1 = t.isf(conf1, lmfit.df_resid)
    nvar2 = t.isf(conf2, lmfit.df_resid)
    pred_upper1 = pred + nvar1 * predstd
    pred_upper2 = pred + nvar2 * predstd
    pred_lower1 = pred - nvar1 * predstd
    pred_lower2 = pred - nvar2 * predstd
    pred = column_stack((pred_lower2, pred_lower1, pred, pred_upper1, pred_upper2))
    return pred

def Predicter(norm_times, norm_data, is_positive=True, trend_only=False, psense="low"):
    """
    Gives prediction based one day dynamic threshold for a data pair points provided in
    norm_times and norm_data

    Parameters
    ----------
    norm_times: 1D array
        one dimensional array of timestamps with fixed step, i.e. timestamps of some timeseries

    norm_data: 1D array
        one dimensional array of values with fixed step, i.e. values of some timeseries

    is_positive: boolean
        indicates if data is non-negative or not

    trend_only: boolean
        indicates if only trend should be predicted or not

    psense: string
        confidence level, i.e. sensitivity, possible values are "low", "medium", "high"

    Returns
    -------
    x_forecast: numpy array
        array containing 3 columns with timestamps, lower and upper bounds
    """
    last_day = daytrunc(norm_times[-1])
    delta_t = norm_times[1] - norm_times[0]
    obsi = ceil((24 * 60 * 60) / delta_t)
    pred_dates = arange((norm_times[-1] + delta_t), (last_day + 2.0 * 24.0 * 60.0 * 60.0), delta_t)
    pred_dates = pred_dates - (24.0 * 60.0 * 60.0)
    linear_prediction = {"thresholds": column_stack((pred_dates, zeros(pred_dates.size), zeros(pred_dates.size))), "reliability": 0.0}
    nahead = pred_dates.size
    mlambda = BCLambda(norm_data)
    linmod = TSLM(YeoJohn(norm_data, mlambda), obsi, trend_only=trend_only)
    pred_reliab = sqrt(linmod.rsquared_adj)
    x_forecast = PredictLM(linmod, h=nahead, sensitivity=psense, trend_only=trend_only)
    x_forecast = apply_along_axis(lambda(w): YeoJohnInv(w, mlambda), 1, x_forecast)
    if nany(isnan(x_forecast[:, 1])) or nany(isnan(x_forecast[:, 3])):
        if (sum(~isnan(x_forecast[:, 0])) > 1) and (sum(~isnan(x_forecast[:, 4])) > 1):
            divx = x_forecast[:, 4] - x_forecast[:, 0]
            divm = array([])
            if sum(isnan(x_forecast[:, 2])) == 0:
                divm = x_forecast[:, 2]
            else:
                if sum(isnan(x_forecast[:, 1])) == 0:
                    divm = x_forecast[:, 1]
                else:
                    if sum(isnan(x_forecast[:, 3])) == 0:
                        divm = x_forecast[:, 3]
            if sum(isnan(divx)) > 0:
                divm0 = divm[~isnan(divx)]
                divx0 = divx[~isnan(divx)]
            else:
                divm0 = divm
                divx0 = divx
            div_lm = OLS(divx0, add_constant(divm0)).fit()
            divx_est = div_lm.params[0] + div_lm.params[1] * divm
            if nany(isnan(divx)):
                divx[isnan(divx)] = divx_est[isnan(divx)]
            if nany(isnan(x_forecast[:, 0])):
                naninds = where(isnan(x_forecast[:, 0]))[0]
                x_forecast[naninds, 0] = x_forecast[naninds, 4] - divx[naninds]
            if nany(isnan(x_forecast[:, 4])):
                naninds = where(isnan(x_forecast[:, 4]))[0]
                x_forecast[naninds, 4] = x_forecast[naninds, 0] + divx[naninds]
        else:
            if ( sum(isnan(x_forecast[:, 0])) == sum(isnan(x_forecast[:, 2]))) and (sum(isnan(x_forecast[:, 4])) == sum(isnan(x_forecast[:, 2]))):
                div_low1 = x_forecast[:, 1]
                div_low2 = x_forecast[:, 0]
                if sum(isnan(div_low2)) > 0:
                    div_low10 = div_low1[~isnan(div_low2)]
                    div_low20 = div_low2[~isnan(div_low2)]
                else:
                    div_low10 = div_low1
                    div_low20 = div_low2
                div_lm_low = OLS(div_low20, add_constant(div_low10)).fit()
                upper_est = (x_forecast[:, 3] - div_lm_low.params[0]) / div_lm_low.params[1]
                div_up1 = x_forecast[:, 4]
                div_up2 = x_forecast[:, 3]
                if sum(isnan(div_low2)) > 0:
                    div_up10 = div_up1[~isnan(div_up2)]
                    div_up20 = div_up2[~isnan(div_up2)]
                else:
                    div_up10 = div_up1
                    div_up20 = div_up2
                div_lm_up = OLS(div_up20, add_constant(div_up10)).fit()
                lower_est = (x_forecast[:, 1] - div_lm_up.params[0]) / div_lm_up.params[1]
                if nany(isnan(x_forecast[:, 4])):
                    naninds_up = where(isnan(x_forecast[:, 4]))[0]
                    x_forecast[naninds_up, 4] = upper_est[naninds_up]
                if nany(isnan(x_forecast[:, 0])):
                    naninds_low = where(isnan(x_forecast[:, 0]))[0]
                    x_forecast[naninds_low, 0] = lower_est[naninds_low]
    x_forecast = delete(x_forecast, (1, 2, 3), 1)
    if is_positive:
        if nany(x_forecast[:, 0] < 0):
            x_forecast[x_forecast[:, 0] < 0, 0] = 0.0
        if nany(x_forecast[:, 1] < 0):
            x_forecast[x_forecast[:, 1] < 0, 1] = 0.0
    x_forecast = column_stack((pred_dates, x_forecast))
    if trend_only:
        x_forecast = x_forecast[-obsi:, ]
    linear_prediction = {"thresholds": x_forecast, "reliability": pred_reliab}
    return linear_prediction
