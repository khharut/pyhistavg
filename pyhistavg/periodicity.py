from scipy.stats import spearmanr
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelmax
from decompose import stl_decompose
from numpy import array, arange, std, median, inf, ceil, arctan


def DRel(itseries, nfreq):
    """
    Doing STL seasonal decomposition on time series itseries vector data with
    window length nfreq and returns standard deviation ratio of seasonal and
    reminder components of decomposition for last season of time series. This
    ratio can be used to determine if time series has periodicity with length
    nfreq or not. When it more than unity then it is possible to have periodic
    pattern, but auto-correlation function is needed to set reliability for
    periodicity.

    Parameters
    ----------
    itseries: 1D array
        one dimensional array of time series

    nfreq: integer
        possible periodic pattern length

    Returns
    -------
    ratio: numeric
        ratio of standard devaiation of seasonal and standard devaiation of remainder
        seasonal and remainder are stl decomposition components

    crho: numeric
        correlation rho value between last two days

    crho_last: numeric
        correlation rho value between last day and seasonal component

    relat: numeric
        median value for seasonal/remainder ratio
    """
    ratio = 0.0
    relat = 0.0
    crho = 0.0
    crho_last = 0.0
    nobsi = len(itseries)
    tryerror = False
    intseries = array(itseries)  # casting to numpy array
    coeff = 1.0
    if (nobsi / nfreq) > 2.0:
        indi = arange((nobsi - nfreq), nobsi)
        intseries_right = intseries[indi]
        intseries_left = intseries[(indi - nfreq)]
        try:
            crho = spearmanr(intseries_left, intseries_right)[0]
        except Exception:
            crho = 0
    if (nobsi / nfreq) >= 3:
        try:  # STL decomposition with window size nfreq
            xad_decomp = stl_decompose(intseries, period=nfreq, robust=False)
        except Exception:
            tryerror = True
        if not tryerror:
            rem = xad_decomp['remainder'][-nfreq:]
            ses = xad_decomp['seasonal'][-nfreq:]
            if std(rem, ddof=1) > 0:
                try:
                    crho_last = spearmanr(ses, intseries_right)[0]
                except Exception:
                    crho_last = 0
                # standard deviation ratio of seasonal/remainder for last season
                ratio = std(ses, ddof=1) / std(rem, ddof=1)
                relat = median(ses / rem)
                if (relat < 0) and (crho < 0.4) and (crho_last < 0.4):
                    coeff = 0.0
                if (min(crho, crho_last) < 0.1) and (relat < 0):
                    coeff = 0.0
                if (crho > 0.7) or (crho_last > 0.9):
                    coeff = 2.0
            else:
                if std(ses, ddof=1) > 0:
                    ratio = inf
                    relat = inf
    ratio = ratio * coeff
    return ratio, crho, crho_last, relat, coeff


def NRel(itseries, obsi):
    """
    Doing auto-correlation analysis of itseries time series to check and give
    reliabilty of having periodic pattern with length obsi. It also uses DRel()
    method to combine with auto-correlation peak value around lag value obsi.

    Parameters
    ----------
    itseries: 1D array
        one dimensional array of time series

    obsi: integer
        possible periodic pattern length

    Returns
    -------
    nacfpeak: array of length 2
        vector of length 2, nacfpeak[0] is periodic pattern length i.e. obsi
        nacfpeak[1] is periodic pattern reliabilty
        nacfpeak[2] is Kolmogorov-Smirnov lambda
    """
    nacfpeak = array([obsi, 0.0])
    instep = len(itseries)  # number of elements in time series
    if (float(instep) / float(obsi)) > 2.0:
        # if less than 3 periodic pattern exists in time series then no acf
        # and STL decomposition can be done
        x1d_acf = array(acf(itseries, nlags=ceil(1.5 * obsi)))
        # searching peaks in acf values
        peak1d = array(argrelmax(x1d_acf))
        # selecting peaks by pattern length 2/12, i.e. 4 hours for daily
        # pattern around acf lag value obsi
        peak1d = peak1d[peak1d >= (obsi - ceil(obsi / 12.0))]
        peak1d = peak1d[peak1d <= (obsi + ceil(obsi / 12.0))]
        # getting maximal value of selected peaks if possible
        if len(peak1d) > 0:
            corval = max(x1d_acf[peak1d])
        else:
            corval = 0.0  # otherwise corellation value is set 0
        per_markers = DRel(itseries, obsi)
        if (corval < 0.1):
            tin = 0
        else:
            tin = 1
        if (corval < 0.25):
            corval = 0
            if (per_markers[1] < 0.1) or (per_markers[2] < 0.1):
                tin = 0
        reliab = corval + tin * per_markers[0]
        reliab = arctan(reliab) / (2.0 * arctan(1.0))
        if ((instep / obsi) < 3) and (per_markers[0] == 0):
            if (corval > 0.25) and (per_markers[1] > 0.5):
                reliab = max(corval, per_markers[1])
        nacfpeak[1] = reliab
    return nacfpeak
