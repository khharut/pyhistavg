from os import environ
from numpy import nan, isnan, mean, median, copy, cumsum
from numpy import array, trunc, datetime64, arange, sqrt
import datetime


def CumMean(x):
    """
    Calculates cumulative mean of give 1D array x

    Parameters
    ----------
    x: 1D numpy array
        values for which mean shold be calculated

    Returns
    -------
    xout: 1D numpy array
        values of cumulative mean
    """
    return cumsum(x) / arange(1, x.size + 1)


def obsicut(vec, obsi):
    """
    Replaces values that < obsi but > 0 in vector to nan

    Parameters
    ----------
    vec: 1D array
        vector of values

    obsi: float
        value of cutoff

    Returns
    -------
    temp: 1D array
        vector with replaced values
    """
    temp = copy(vec)
    temp = temp.astype(float)
    temp[temp < obsi] = nan
    return temp


def currentday(current_ts):
    """
    Gives timestamp of closest 24:00

    Parameters
    ----------
    current_ts: int
        timestamp of calculation time

    Returns
    -------
    current_day: timestamp of closest 24:00
    """
    environ['TZ'] = 'UTC'
    day_milisecs = 24 * 60 * 60 * 1000
    ts_hms = current_ts % day_milisecs
    if ts_hms < (12 * 60 * 60 * 1000):
        current_day = current_ts - ts_hms
    else:
        current_day = current_ts - ts_hms + day_milisecs
    current_day = int(current_day / 1000)
    return current_day


def Signif(x, digits=3):
    """
    round x float value leaving decimal digits equal to number digits

    Parameters
    ----------
    x: float
        float value to be rounded

    Returns
    -------
    value: float
        float value with precision defined by digits
    """
    return float("%.3f" % x)


def Mean(x):
    """
    mean value of numpy array when some nan exists

    Parameters
    ----------
    x: 1D numpy array
        data array with nans

    Returns
    -------
    np.mean(xin): numeric
        mean value
    """
    xin = array(x[~isnan(x)])
    if xin.size > 0:
        return mean(xin)
    else:
        return nan


def Median(x):
    """
    median value of numpy array when some nan exists

    Parameters
    ----------
    x: 1D numpy array
        data array with nans

    Returns
    -------
    value: numeric
        median value
    """
    xin = array(x[~isnan(x)])
    if xin.size > 0:
        return median(xin)
    else:
        return nan


def MinMax(x):
    """
    minimal and maximal value of numpy array when some nan exists

    Parameters
    ----------
    x: 1D numpy array
        data array with nans

    Returns
    -------
    value: 1D 2 element numpy array
        min and max values
    """
    xin = array(x[~isnan(x)])
    if xin.size > 0:
        minmax = array((min(xin), max(xin)))
    else:
        minmax = array((nan, nan))
    return minmax


def Max(x):
    """
    maximal value of numpy array when some nan exists

    Parameters
    ----------
    x: 1D numpy array
        data array with nans

    Returns
    -------
    value: 1D 2 element numpy array
        min and max values
    """
    xin = array(x[~isnan(x)])
    if xin.size > 0:
        return max(xin)
    else:
        return nan


def Min(x):
    """
    minimal value of numpy array when some nan exists

    Parameters
    ----------
    x: 1D numpy array
        data array with nans

    Returns
    -------
    value: 1D 2 element numpy array
        min and max values
    """
    xin = array(x[~isnan(x)])
    if xin.size > 0:
        return min(xin)
    else:
        return nan


def daytrunc(timestamps):
    """
    Gives timestamp of closest day start, i.e. 00:00:00

    Parameters
    ----------
    timestamps: 1D numpy array or single value
        timestamps to be converted to datetime

    Returns
    -------
    dtimestamps: 1D numpy array or single value
        of Unix timestamps of day start
    """
    if hasattr(timestamps, '__iter__'):
        dtimestamps = map(lambda(w): 24 * 60 * 60 * (w // (24 * 60 * 60)), timestamps)
    else:
        dtimestamps = 24 * 60 * 60 * (timestamps // (24 * 60 * 60))
    return dtimestamps


def HumanTime(timestamps):
    """
    Converts timestamps to human datetime

    Parameters
    ----------
    timestamps: 1D numpy array or single value
        timestamps to be converted to datetime

    Returns
    -------
    human_times: 1D numpy array or single value
        of datetime objects
    """
    if hasattr(timestamps, '__iter__'):
        human_times = map(datetime.datetime.utcfromtimestamp, timestamps)
    else:
        human_times = datetime.datetime.utcfromtimestamp(timestamps)
    return human_times


def StrTime(timestamps):
    """
    Converts timestamps to string like "2015-12-25 13:55"

    Parameters
    ----------
    timestamps: 1D numpy array or single value
        timestamps to be converted to string

    Returns
    -------
    str_times: 1D numpy array or single value
        strings of formatted timestamps
    """
    if hasattr(timestamps, '__iter__'):
        datatimes = map(datetime.datetime.utcfromtimestamp, timestamps)
        str_times = map(lambda(x): x.strftime("%Y-%m-%d %H:%M"), datatimes)
    else:
        datatimes = datetime.datetime.utcfromtimestamp(timestamps)
        str_times = datatimes.strftime("%Y-%m-%d %H:%M")
    return str_times


def NHumanTime(timestamps):
    """
    Converts timestamps to human datetime using numpy

    Parameters
    ----------
    timestamps: 1D numpy array or single value
        timestamps to be converted to datetime

    Returns
    -------
    human_times: 1D numpy array or single value
        of datetime objects
    """
    if hasattr(timestamps, '__iter__'):
        human_times = map(lambda x: datetime64(int(x), 's'), timestamps)
    else:
        human_times = datetime64(timestamps, 's')
    return human_times


def UnixTime(human_times):
    """
    Converts timestamps to human datetime
    Parameters
    ----------
    human_times: 1D numpy array or single value
        datetime objects to be converted to timestamps

    Returns
    -------
    timestamps: 1D numpy array or single value
        of Unix timestamps
    """
    if hasattr(human_times, '__iter__'):
        timestamps = map(lambda(w): (w - datetime.datetime(1970, 1, 1)).total_seconds(), human_times)
    else:
        timestamps = (human_times - datetime.datetime(1970, 1, 1)).total_seconds()
    return timestamps
