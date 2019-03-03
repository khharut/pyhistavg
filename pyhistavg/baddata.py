from stests import TestA, TestMV
from kstest import TestK
from numpy import ceil, array, append, setdiff1d
from numpy import where, intersect1d, ones, diff
from numpy import unique, delete, arange
from numpy import any as nany


def BadInd(norm_times, norm_data, real_times, change_inds, xhint=False):
    """
    Checks if some days in a data can be removed to have stationary data

    Parameters
    ----------
    norm_times: 1D array
        one dimensional array of timestamps with fixed step, i.e. timestamps of some timeseries

    norm_data: 1D array
        one dimensional array of values with fixed step, i.e. values of some timeseries

    real_times: 1D array
        one dimensional array of timestamps of raw data

    change_inds: 1D array
        indices of stationary changes

    xhint: boolean
        indicates if data is highly unstable or not

    Returns
    -------
    erase_inf: dictionary of 2 numpy arrays
        numpy array named "indices" is indeces in raw data to be removed
        numpy array named "stat" is indeces of new stationary change points, can be empty
    """
    nobsi = norm_data.size
    delta_t = norm_times[2] - norm_times[1]
    obsi = ceil(24.0 * 60.0 * 60.0 / delta_t)
    schanges = change_inds
    erase_inf = {"indices": [], "stat": schanges}
    if len(schanges) > 0:
        binds = where(diff(schanges) < obsi)[0]
        if len(binds) > 0:
            erase_inds = array([])
            rerase_inds = array([])
            chmark = ones(len(binds))
            for i in range(len(binds)):
                if ((schanges[binds[i]] + 1 + obsi) < (nobsi - obsi)):
                    chts = delete(norm_data, arange(schanges[binds[i]] + 1, schanges[binds[i]] + 1 + obsi))
                else:
                    chts = delete(norm_data, arange(schanges[binds[i]] + 1, schanges[binds[i] + 1] - 1))
                if TestK(chts, schanges[binds[i]], obsi, 2) == 0:
                    chmark[i] = TestA(chts, schanges[binds[i]], obsi) + TestMV(chts, schanges[binds[i]], obsi)
                else:
                    chmark[i] = 0.2
                if (not xhint) and (chmark[i] == 0.1):
                    chmark[i] = 0
                if (chmark[i] < 0.06):
                    if ((schanges[binds[i]] + obsi + 1) <= nobsi):
                        erase_inds = append(erase_inds, arange(schanges[binds[i]], (schanges[binds[i]] + obsi)))
                    else:
                        erase_inds = append(erase_inds, arange(schanges[binds[i]], nobsi))
                    remdate = real_times[max(where(real_times <= norm_times[schanges[binds[i]]])[0])]
                    rerase_inds = append(rerase_inds, where(((real_times <= (remdate + 24 * 60 * 60)) & (real_times > remdate)))[0])
            erase_inds = unique(erase_inds)
            rerase_inds = unique(rerase_inds)
            if nany(chmark < 0.06):
                temp_inds = binds[where(chmark < 0.06)[0]]
                temp_inds = append(temp_inds, (temp_inds + 1))
                schanges = delete(schanges, temp_inds)
            erase_stat = intersect1d(erase_inds, schanges)
            if (len(erase_stat) > 0):
                schanges = setdiff1d(schanges, erase_stat)  # damn asymetric setdiff!
            erase_inf["indices"] = rerase_inds
            erase_inf["stat"] = schanges
    return erase_inf
