from stests import TestMV, TestA, TestS
from kstest import TestK, KSValues
from numpy import array, log, cumsum, arange, repeat
from numpy import zeros, std, mean, isnan, empty
from numpy import inf, sqrt, where, delete, argmax
from numpy import concatenate, sort, append
from numpy import abs as nabs
from numpy import any as nany


def MultipleChange(intseries, Q = 5, robust = False):
    """
    Originally a method from changepoint R package called multiple_var_css,
    which gives indeces of possible stationarity change points or changepoints
    of a vector. It searches changes in variance in a vector

    Parameters
    ----------
    intseries: 1D array
        numpy array or other array that can be converted into numpy array

    Q: integer
        number of desirable changepoints in data, i.e. how many changepoint
        to search, default is 5

    Returns
    -------
    changepoints: 1D array
        numpy 1D array i.e. vector of indeces of possible stationarity changes
        it can also be empty array, in case when no change exists
    """
    itseries = array(intseries)  # casting itseries into numpy array
    nobsi = itseries.size  # number of elements in intseries
    pen = log(log(nobsi))  # penalty value
    if (nobsi < 4):  # if number of points less than 4 nothing can be done
        raise Exception('Data must have at least 4 observations to fit a changepoint model.')
    if (Q > ((nobsi / 2) + 1)):  # in case of Q greater than number of vector segments
        raise Exception('Q is larger than the maximum number of segments')
    std_data = std(itseries, ddof=1)
    mean_data = mean(itseries)
    if isnan(std_data):
        std_data = 0
    if std_data > 0:
        if robust:
            data = itseries - mean_data
        else:
            data = itseries / std_data
        y2 = zeros(nobsi + 1)  # initializing intseries square vector
        y2[1:] = cumsum(data ** 2)
        tau = repeat((nobsi - 1), (Q + 2))  # array of segments indeces, primordal is (0,nobsi-1)
        tau[0] = 0
        cpt = zeros((2, Q), dtype=float)  # initializing changepoint array with indeces and penalty values
        cpt[1, 0] = inf
        lam = zeros(nobsi - 1)  # initializing vector of test statistics
        for q in xrange(Q):  # searching changepoints
            i = 0
            st = tau[0]
            end = tau[1]  # start and end indeces of segments
            for j in xrange(nobsi - 1):  # calculating lam test statistics for segments
                if j == end:
                    st = end + 1
                    i = i + 1
                    end = tau[i + 1]
                else:
                    if y2[end + 1] != y2[st]:
                        lam[j] = sqrt((end - st + 1) / 2) * abs((y2[j + 1] - y2[st]) / (y2[end + 1] - y2[st]) - (float(j) - st + 1) / (end - st + 1))
            k = argmax(lam)  # saving index of maximal value of test statistics
            cpt[0, q] = k
            if q > 0:
                cpt[1, q] = min(cpt[1, (q - 1)], lam[k])  # saving maximal value of test statistics
            else:
                cpt[1, q] = lam[k]
            tau[q + 1] = k  # adding changepoint index to segments indeces vector
            tau.sort()  # sorting segments start-end indeces
            lam[:] = 0.0
        criterion = where(cpt[1, :] >= pen)[0]  # checking if any changepoint meets test statistics
        op_cps = criterion.size  # number of discovered changepoints abeying penalty criterion
        if (op_cps > 0):  # any changepoint obeying penalty criterion returned
            changepoints = array(cpt[0, criterion], dtype=int)  # detected changepoints indeces, extracting changepoints indeces obeying penalty criterion
        else:
            changepoints = array([], dtype=int)  # empty array of detected changepoints indeces
    return changepoints


def SingleChange(itseries):
    """
    Originally a method from changepoint R package called cpt_meanvar,
    which gives indece of most possible stationarity change point or changepoints
    of a vector. It searches changes in variance and mean in a vector, assuming that it is Gaussian

    Parameters
    ----------
    itseries: 1D array
        numpy array or other array that can be converted into numpy array

    Returns
    -------
    tau: integer
        index of most possible changepoint in itseries 1D numpy array
        it can not be empty array, in any case it will try to find changepoint
    """
    intseries = array(itseries)
    n = intseries.size
    y = append(array([0]), intseries.cumsum())
    y2 = append(array([0]), (intseries ** 2).cumsum())
    taustar = arange(2, (n - 1))
    sigma1 = ((y2[taustar] - ((y[taustar] ** 2) / taustar)) / taustar)
    sigma1[where(sigma1 <= 0)[0]] = 10 ** (-10)
    sigman = ((y2[n] - y2[taustar]) - (((y[n] - y[taustar]) ** 2) / (n - taustar))) / (n - taustar)
    sigman[where(sigman <= 0)[0]] = 10 ** (-10)
    sigman = array(sigman, dtype=float)
    sigma1 = array(sigma1, dtype=float)
    tmp = taustar * log(sigma1) + (n - taustar) * log(sigman)
    tau = where(tmp == min(tmp))[0][0]
    tau = tau + 1
    tau = array([tau], dtype=int)
    return tau


def StatLen(itseries, iobs, xhint=False, xper=False, nchanges=None):
    """
    Detecting stationary changes in time series itseries using different methods

    Parameters
    ----------
    itseries: 1D array
        numpy array or other array that can be converted into numpy array

    iobs: integer
        number of observations per day

    xhint: boolean
        indices if data has bad behavior to treat with robust methods

    xper: boolean
        indices if data periodic or not to treat with less robust methods

    Returns
    -------
    stati2: dictionary
        containing to numpy arrays named "strong" and "weak",
        "strong" has stationary change indeces with strong changes
        "weak" has stationary change indeces with weak changes
        "lambdas" Kolmogorov-Smirnov test lambda values found used KSValues method
    """
    intseries = array(itseries)
    empty_array = empty(0, dtype=int)
    lchanges = empty_array
    l3dstat = 0
    trend_det = 0
    nobsi = intseries.size
    stati2 = {"strong": [0, (nobsi - 1)], "weak": [0, (nobsi - 1)]}
    obsi = iobs
    if ((nobsi - 2 * obsi) >= 0):
        rnch = int(round(nobsi / (3 * obsi)))
    else:
        rnch = 0
    if nchanges is not None and obsi >= 24:
        rnch = nchanges
    if rnch > 0:
        lchanges = MultipleChange(intseries, Q=rnch)
        if (lchanges.size < rnch) or (not xper):
            xchange = SingleChange(intseries)
        else:
            xchange = empty_array
        if not xper:
            if nobsi > 3 * obsi:
                trend_det = TestS(intseries[(nobsi - 2 * obsi):]) * TestS(intseries[(nobsi - obsi):])
                l3dstat = TestMV(intseries, (nobsi - obsi), obsi) + TestA(intseries, (nobsi - obsi), obsi)
            if (nobsi > (3 * obsi)) and (l3dstat > 0):
                lchanges0 = MultipleChange(intseries[(nobsi - 3 * obsi):], Q=3, robust=True)
                if lchanges0.size > 0:
                    lchanges0 = lchanges0 + (nobsi - 3 * obsi)
            else:
                lchanges0 = empty_array
            if (lchanges.size > 0) and (lchanges0.size > 0):
                dchanges = [nany((w - lchanges) < 3) for w in lchanges0]
                if sum(dchanges) > 0:
                    lchanges0 = lchanges0[~array(dchanges)]
            if lchanges0.size > 0:
                lchanges = sort(append(lchanges, lchanges0))
            if (lchanges.size == 0) and (lchanges0.size > 0):
                lchanges = lchanges0
            if (lchanges.size > 0) and (xchange.size > 0) and (min(abs(lchanges - xchange)) >= 3):
                lchanges = sort(append(lchanges, xchange))
            else:
                if (lchanges.size == 0) and (xchange.size > 0):
                    lchanges = xchange
        else:
            if (lchanges.size > 0) and (xchange.size > 0):
                if not nany(lchanges == xchange):
                    lchanges = sort(append(lchanges, xchange))
            if (lchanges.size == 0) and (xchange.size > 0):
                lchanges = xchange
    if (lchanges.size > 0):
        lchanges = sort(lchanges)
        if (trend_det > 0):
            lchanges = lchanges[lchanges <= (nobsi - obsi - 1)]
        ltest = array(map(lambda(w): TestA(intseries, w, obsi) + TestMV(intseries, w, obsi) + TestK(intseries, w, obsi, 2), lchanges))
        if xhint:
            kltest = array(map(lambda(w): TestK(intseries, w, obsi, 0), lchanges))
        else:
            kltest = array(map(lambda(w): TestK(intseries, w, obsi, 1), lchanges))
        if (not xhint) and nany(ltest > 0.19):
            stati2["strong"] = [0] + lchanges[ltest > 0.19].tolist() + [nobsi - 1]
        if (xhint) and nany(ltest > 0.06):
            stati2["strong"] = [0] + lchanges[ltest > 0.06].tolist() + [nobsi - 1]
        if nany(kltest > 0.06):
            stati2["weak"] = [0] + lchanges[kltest > 0.06].tolist() + [nobsi - 1]
    return stati2
