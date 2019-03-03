from scipy.stats import f, t, spearmanr
from statsmodels.tsa.arima_model import ARIMA as arima
from statsmodels.tsa.arima_model import AR as ar
from pandas import expanding_var as evar
from pandas import expanding_mean as emean
from numpy import arange, var, std, inf, array, empty
from numpy import sqrt, mean, trunc, cumsum, append
from numpy import diff, arange, log, ones, isnan
from numpy import column_stack, log, where, sort
from numpy.linalg import lstsq
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from helper import CumMean


def TestS(data):
    """
    Checking if trend exists in data vector.

    Parameters
    ----------
    data: 1D array
        numpy array or other array that can be converted into numpy array

    Returns
    -------
    cpvalue: float
        0.1 or 0,
        0.1 if any trend exists,
        0 otherwise
    """
    n_data = len(data)
    # indeces of data
    data_times = arange(1, (n_data + 1))
    # performing Spearman corelation test between indeces and data, taking rho value
    sp_res = abs(spearmanr(data, data_times)[0])
    # if strong correlation rho>=0.9 exists between indeces and data then trend
    # exists, otherwise too weak trend can be
    if sp_res >= 0.9:
        return 0.1
    else:
        return 0


def TestMV(itseries, ind, lobs):
    """
    Performs mean and varaince change statistical tests. Variance change test
    based on F-test(F-distribution), and mean test based on t-test (Student's
    distribution)

    Parameters
    ----------
    itseries: 1D array
        one dimensional array of time series

    ind: integer
        index in time series around which ARIMA(1,0,0) process performed

    lobs: integer
        maximal number of elements to left and right sides around ind index on
        which ARIMA(1,0,0) is performed

    Returns
    -------
    cpvalue: numeric
        0.1, 0.05 or 0, 0.1 if ind around ind mean and variance change
        significanly, 0.05 if only mean or variance changes significantly
        0 in case when none of them changes
    """
    cpvalue = 0
    tstart = max(ind - lobs + 1, 0)  # start of test interval around ind index
    tend = min(ind + lobs + 1, len(itseries))  # end of test interval around ind index
    if (len(itseries[tstart: (ind + 1)]) > 1) and (len(itseries[(ind + 1): tend]) > 1):
        # it is needed to have at least one element in both left and right sides
        # around ind index
        if (var(itseries[tstart: (ind + 1)], ddof=1) == 0) and (var(itseries[(ind + 1): tend], ddof=1) != 0):
            cpvalue = 0.1
        # in case when variance is zero on left side and non zero on rigth
        # side then definitely variance changes around ind index
        if (var(itseries[tstart: (ind + 1)], ddof=1) != 0) and (var(itseries[(ind + 1):tend], ddof=1) == 0):
            cpvalue = 0.1
        if (var(itseries[tstart: (ind + 1)], ddof=1) * var(itseries[(ind + 1): tend], ddof=1)) > 0:
            intseries = array(itseries[tstart: tend])  # slicing test data
            n = len(intseries)
            mid_ind = (n / 2) - 1  # ind element position in sliced data
            all_means = emean(intseries, min_periods=1)
            all_vars = evar(intseries, min_periods=1)
            rev_all_means = emean(intseries[::-1], min_periods=1)
            rev_all_vars = evar(intseries[::-1], min_periods=1)
            test_lens = arange((mid_ind + 1), (n + 1))
            if (rev_all_vars[mid_ind] > 0) and (all_vars[mid_ind] > 0):
                z = all_vars[mid_ind] / rev_all_vars[mid_ind]
                rz = 1 / z
            else:
                z = inf
                rz = 0.0
            ## variance change F-test with reliabilty value 99.8% (0.1%-99.9%)
            if (z > f.ppf(1 - 0.001, mid_ind, mid_ind)) or (z < f.ppf(0.001, mid_ind, mid_ind)):
                cpvalue = 0.05
            if (rz > f.ppf(1 - 0.001, mid_ind, mid_ind)) or (rz < f.ppf(0.001, mid_ind, mid_ind)):
                cpvalue = 0.05
            ## calculation of t-test statistics
            Sx_y = sqrt(((mid_ind * all_vars[mid_ind] + test_lens * all_vars[mid_ind:]) * (mid_ind + test_lens)) / ((mid_ind + test_lens - 2) * mid_ind * test_lens))
            t_jn = nabs((all_means[mid_ind] - all_means[mid_ind:]) / Sx_y)

            rSx_y = sqrt(((mid_ind * rev_all_vars[mid_ind] + test_lens * rev_all_vars[mid_ind:]) * (mid_ind + test_lens)) / ((mid_ind + test_lens - 2) * mid_ind * test_lens))
            rt_jn = nabs((rev_all_means[mid_ind] - rev_all_means[mid_ind:]) / rSx_y)

            t_stat = nmax((t_jn, rt_jn))
            dfree = n - 2
            # mean change t-test with reliabilty value 99.8% (0.1%-99.9%)
            if t_stat > t.ppf(1 - 0.001, dfree):
                cpvalue = cpvalue + 0.05
        if cpvalue > 0:
            # in case if cpvalue  is 0.1 then checking if detected changepoint is
            # significant by calculating sindic value for interval
            sindic = abs(std(itseries[tstart: (ind - 1)], ddof=1) - std(itseries[(ind + 1): tend], ddof=1))
            sindic = sindic * mean(itseries[tstart:tend])
            if sindic is not None:
                if sindic <= 0.03: cpvalue = 0  # if sindic is less than 0.03 then changepoint is not significant
    return cpvalue


def HurstWrapper(x):
    """
    Wrapper for Hurst exponent calculator, see Hurst() method for details

    Parameters
    ----------
    x: 1D array
        one dimensional array

    Returns
    -------
    hvalue: float
        Hurst exponent value, normally it lays in interval [0,1]
    """
    xin = x[~isnan(x)]
    hvalue = 0
    try:
        hvalue = Hurst(xin)
    except Exception:
        hvalue = 0
    return hvalue


def ARIMAWrapper(x):
    """
    Wrapper for autoregression coeffient calculator, see arima() method for details

    Parameters
    ----------
    x: 1D array
        one dimensional array

    Returns
    -------
    rhoval: float
        autoregression coeffient, normally it lays in interval [-1, 1]
    """
    xin = x[~isnan(x)]
    rhoval = 0.0
    try:
        xin_ar = ar(xin).fit(disp=False, method='mle', solver='bfgs', maxlag=1)
        rhoval = xin_ar.params[1]
    except Exception:
        try:
            xin_arima = arima(xin, (1, 0, 0)).fit(disp=False, method='css', solver='bfgs')
            rhoval = xin_arima.params[1]
        except Exception:
            rhoval = 0.92
    return rhoval


def TestA(itseries, ind, lobs):
    """
    Calculates ARIMA(1,0,0) process rho value for time series around ind index
    [ind-lobs,ind+lobs]. If rho is very close to unity ( here we check that
    rho>(1-alpha)) it is well-known that time series is non-stationar

    Parameters
    ----------
    itseries: 1D array
        one dimensional array of time series

    ind: integer
        index in time series around which ARIMA(1,0,0) process performed

    lobs: integer
        maximal number of elements to left and right sides around ind index on
        which ARIMA(1,0,0) is performed

    Returns
    -------
    cpvalue: float
        0.1 or 0, 0.1 if ind is really stationarity change by means of ARIMA,
        0 otherwise

    """
    cpvalue = 0
    alpha = 0.08
    arimastart = max(ind - lobs + 1, 0)  # start of test interval around ind index
    arimaend = min(ind + lobs + 1, len(itseries))  # end of test interval around ind index
    intseries = itseries[arimastart:arimaend]  # slicing test interval
    intseries_rho = ARIMAWrapper(intseries)
    if intseries_rho > (1 - alpha):
        cpvalue = 0.1  # if rho>(1-alpha) then we have non-stationar time series
    if cpvalue > 0:
        # in case if cpvalue  is 0.1 then checking if detected changepoint is
        # significant by calculating sindic value for interval
        sindic = abs(std(itseries[arimastart:(ind - 1)], ddof=1) - std(itseries[(ind + 1):arimaend], ddof=1))
        sindic = sindic * mean(intseries)
        if sindic is not None:
            if sindic <= 0.03:
                cpvalue = 0  # if sindic is less than 0.03 then changepoint is not significant
    return cpvalue


def Hurst(x):
    """
    Calculates Hurst exponent, long term memory in array. For random walk
    process it cloes to 0.5. For non-stationary data it goes above 1. For
    data with any periodic pattern Hurst exponent lies in between interval
    [0.75,0.9].

    Parameters
    ----------
    x: 1D array
        one dimensional array

    Returns
    -------
    hexp: float
        Hurst exponent value, normally it lays in interval [0,1]
    """
    def half(N):  # halving segments indeces in N
        return sort(append(N, (N[: -1] + trunc((diff(N) + 1.0) / 2.0))))

    def rscalc(x):  # simple R/S Hurst exponent estimation
        y = cumsum(x - mean(x))
        R = max(y) - min(y)
        S = std(x, ddof=1)
        return R / S

    try:
        n = len(array(x))
    except Exception:
        n = 0
    if n >= 16:
        X = array(n)
        Y = array(rscalc(x))
        N = array([0, trunc(n / 2.0), n])
        #  calculating Hurst exponent for halved segments of x
        while min(diff(N)) >= 8:
            xl = []
            yl = []
            for i in range(1, len(N)):
                rs = rscalc(x[N[i - 1]: N[i]])
                xl.append((N[i] - N[i - 1]))
                yl.append(rs)
            xl = array(xl)
            yl = array(yl)
            X = append(X, mean(xl))
            Y = append(Y, mean(yl))
            N = half(N)
        if nany(isnan(Y)):
            X = X[~isnan(Y)]
            Y = Y[~isnan(Y)]
        X = column_stack((log(X), ones(len(X))))
        Y = log(Y)
        # linear regerssion between log(n) and log(R/S) values
        hexp = lstsq(X, Y)[0][0]
    else:
        if n > 0:
            hexp = rscalc(x)
        else:
            hexp = None
    return hexp


def TestLD(real_times, norm_times, lobs):
    """
    Tests if too much losing data exists in timestamps

    Parameters
    ----------
    real_times: 1D array
        one dimensional array of timestamps of raw data

    norm_times: 1D array
        one dimensional array of timestamps with fixed step, i.e. timestamps of some timeseries

    lobs: integer
        maximal number of elements to left and right sides around ind index on
        which ARIMA(1,0,0) is performed

    Returns
    -------
    cpvalue: float
        0.1 or 0, 0.1 if if there is really a lot of lost data in timestamps,
        0 otherwise

    """
    idtime = 24.0 * 60.0 * 60.0 / lobs
    cpvalue = 0
    if (len(norm_times) > 1):
        notime = len(where((real_times <= norm_times[-1]) & (real_times > norm_times[0]))[0])
        ndtime = (norm_times[-1] - norm_times[0]) / float(idtime)
        if ((notime / ndtime) < 0.75):
            cpvalue = 0.1
    if (len(norm_times) == 1):
        notime1 = len(where((real_times <= (norm_times[0] + 11 * idtime)) & (real_times > norm_times[0]))[0])
        notime2 = len(where((real_times <= norm_times[0]) & (real_times > (norm_times[0] - 11 * idtime)))[0])
        if (notime1 < 7) or (notime2 < 7):
            cpvalue = 0.1
    return cpvalue
