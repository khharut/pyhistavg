from numpy import sqrt, array, arange, append, zeros, column_stack, sort, exp
from numpy import argsort, copy, diff, where, delete, intersect1d, unique
from periodicity import DRel, NRel
from decompose import stl_decompose
from numpy import all as nall
from scipy.stats import ks_2samp


def KSWrapper(itseries, ind, lobs):
    """
    Performs Kolmogorov-Smirnov test on itseries data around index ind i.e. [ind - lobs, ind + lobs]

    Parameters
    ----------
    itseries: 1D numpy array
        data array to be checked

    ind: integer
        index around which test performed

    lobs: integer
        window length

    Returns
    -------
    test_res: 2 element 1D array
        containing lambda sqrt(0.5*lobs)*distance and
    """
    test_res = 0
    ktend = ind + lobs + 1
    ktstart = ind - lobs + 1
    nobsi = len(itseries)
    if (ktend <= (nobsi - 1)) and (ktstart >= 0):
        lcoef = sqrt(0.5 * lobs)
        try:
            ks_test = ks_2samp(itseries[ktstart: (ind + 1)], itseries[(ind + 1): ktend])
            test_res = lcoef * ks_test[0]
        except Exception:
            test_res = 0
    return test_res

def TestK(itseries, ind, lobs, robust=1):
    """
    Performs Kolmogorov-Smirnov test on itseries data around index ind i.e. [ind - lobs, ind + lobs]
    and detecting if any stationary change exists using KSWrapper() method

    Parameters
    ----------
    itseries: 1D numpy array
        data array to be checked

    ind: integer
        index around which test performed

    lobs: integer
        window length

    robust: integer
        use more strict condition for ecdf() different distance
        values are 0, 1, 2

    Returns
    -------
    cpvalue: numeric
        0 if no change detected
        0.1 otherwise
    """
    cpvalue = 0
    nobsi = len(itseries)
    if (ind >= (lobs - 1)) and (ind <= (nobsi - lobs - 1)):
        ktest = KSWrapper(itseries, ind, lobs)
    else:
        if (nobsi >= (2 * lobs)):
            if (ind < (lobs - 1)):
                ktest = KSWrapper(itseries, lobs - 1, lobs)
            if (ind > (nobsi - lobs - 1)):
                ktest = KSWrapper(itseries, nobsi - lobs - 1, lobs)
    if (robust == 0) and (ktest >= exp(1)):
        cpvalue = 0.1
    if (robust == 1) and (ktest >= 4.2):
        cpvalue = 0.1
    if (robust == 2) and (ktest >= 7):  # exp(2) should be changed to 7
        cpvalue = 0.15
    return cpvalue


def KSValues(intseries, lobs):
    """
    Performs Kolmogorov-Smirnov test on timeseries like numpy
    array with predefined step lobs and gives result in ksvalues
    array

    Parameters
    ----------
    itseries: 1D numpy array
        data array to be checked

    lobs: integer
        predefined step

    Returns
    -------
    ksvalues: 2D array
        containing indeces around which Kolmogorov-Smirnov test
        is performed and lambda sqrt(0.5*lobs)*distance and
    """
    nobsi = len(intseries)
    obsi = lobs
    ksvalues = array([[0, nobsi - 1]])
    if (nobsi - 2 * obsi) >= 0:
        kindeces = arange(obsi, nobsi, obsi, dtype=int)
        if kindeces[-1] == (nobsi - 1):
            delete(kindeces, -1)
        else:
            if kindeces[-1] > (nobsi - obsi - 1):
                kindeces[-1] = (nobsi - obsi - 1)
            else:
                kindeces = append(kindeces, (nobsi - obsi - 1))
        if obsi >= 24:
            lambdas = array(map(lambda(w): KSWrapper(intseries, w, obsi), kindeces))
        else:
            lambdas = zeros(len(kindeces))
        ksvalues = column_stack((kindeces, lambdas))
    return(ksvalues)


def KSInfo(lambdas):
    """
    Performs Kolmogorov-Smirnov test on timeseries like numpy
    array with predefined step lobs and gives result in ksvalues
    array

    Parameters
    ----------
    lambdas: 2D numpy array
        array of Kolmogorov-Smirnov lambda obtained from KSValues() method

    Returns
    -------
    ksinfo: dictionary
        containing:
        "nchange"  : number of stationary changes in original data
        "weekly7"  : boolean value indicating whether any weekly pattern
                    (7 day) exist even in Kolmogorov-Smirnov lambdas
        "weekly52" : boolean value indicating whether any weekly pattern
                     (5 day and then 2 day) exist even in Kolmogorov-Smirnov lambdas
        "highly_unstable" : boolean value indicating whether original data highly_unstable,
                            mainly shows some remarkable trend
    """
    ee = exp(1)
    weekper = 0
    ksinfo = {"weekchange": array([], dtype=int), "type": 0, "nchange": 0, "weekly7": [False, False], "weekly52": [False, False], "highly_unstable": False}
    if sum(lambdas[:, 1] > 4) > 0:
        ee = 2.5
    change_num = sum(lambdas[:, 1] > ee)
    ksinfo["nchange"] = change_num
    ks_len = len(lambdas[:, 1])
    weeksn = ks_len / 7.0
    weeksnt = ks_len / 7
    if len(lambdas[:, 0]) > 1:
        obsi = lambdas[1, 0] - lambdas[0, 0]
        nobsi = max(lambdas[:, 0]) + obsi
        weeksn = float(nobsi) / (7.0 * float(obsi))
        weeksnt = round(weeksn)
    else:
        obsi = 0
        nobsi = 0
        weeksn = 0
        weeksnt = 0
    if (change_num == ks_len) and (change_num > 0):
        ksinfo["highly_unstable"] = True
    if weeksn > 2 and change_num > 0:
        if (lambdas[-1, 0] - lambdas[-2, 0]) < obsi:
            high_ks = copy(lambdas[: -1, :])
        else:
            high_ks = copy(lambdas[:])
        weekper = NRel(high_ks[:, 1], 7)[1]
        if (len(high_ks[:, 0]) > (7 * weeksnt)):
            high_ks = high_ks[(len(high_ks[:, 0]) - 7 * weeksnt):, ]
        low_ks = copy(high_ks)
        if sum(high_ks[:, 1] > ee) > 1:
            high_ks = high_ks[(high_ks[:, 1] > ee), :]
            ksorder = argsort(high_ks[:, 1])
            ksorder = ksorder[::-1]
            high_ks = high_ks[ksorder, :]
            if len(high_ks[:, 1]) >= weeksnt:
                pattern7d = sort(high_ks[: weeksnt, 0])
                kspattern7d = diff(pattern7d)
                kspattern7d = kspattern7d / obsi
                ksinfo["weekly7"][0] = nall(kspattern7d == 7)
            if len(high_ks[:, 1]) >= (2 * weeksnt):
                pattern52d = sort(high_ks[: (2 * weeksnt), 0])
                kspattern52d = diff(pattern52d)
                kspattern52d = kspattern52d / obsi
                kspattern52d = unique(kspattern52d)
                if len(kspattern52d) == 2:
                    ksinfo["weekly52"][0] = ((kspattern52d[0] + kspattern52d[1]) == 7)
        if sum(low_ks[:, 1] < ee) > 1:
            low_ks = low_ks[(low_ks[:, 1] <= ee), :]
            lksorder = argsort(low_ks[:, 1])
            low_ks = low_ks[lksorder, :]
            if len(low_ks[:, 1]) >= weeksnt:
                lpattern7d = sort(low_ks[: weeksnt, 0])
                lkspattern7d = diff(lpattern7d)
                lpattern7d = copy(lkspattern7d)
                lkspattern7d = lkspattern7d / obsi
                ksinfo["weekly7"][1] = nall(lkspattern7d == 7)
            if len(low_ks[:, 1]) >= (2 * weeksnt):
                lpattern52d = sort(low_ks[: (2 * weeksnt), 0])
                lkspattern52d = diff(lpattern52d)
                lkspattern52d = lkspattern52d / obsi
                lkspattern52d = unique(lkspattern52d)
                if (len(lkspattern52d) == 2):
                    ksinfo["weekly52"][1] = ((lkspattern52d[0] + lkspattern52d[1]) == 7)
    if ksinfo["weekly7"][0] or ksinfo["weekly52"][0]:
        ksinfo["type"] = 3
    else:
        if ksinfo["weekly7"][1] or ksinfo["weekly52"][1]:
            ksinfo["type"] = 2
    if (ksinfo["weekly52"][0]):
        ksinfo["weekchange"] = pattern52d
    if (ksinfo["weekly7"][0]) and (ksinfo["weekchange"].size == 0):
        ksinfo["weekchange"] = pattern7d
    ksinfo["weekly52"] = ksinfo["weekly52"][0] or ksinfo["weekly52"][1]
    ksinfo["weekly7"] = ksinfo["weekly7"][0] or ksinfo["weekly7"][1]
    if (ksinfo["type"] == 0) and (weekper > 0.75) and (change_num >= weeksn):
        ksinfo["type"] = 1
    if (ksinfo["type"] > 0) and (weeksn >= 3.0):
        if (lambdas[-1, 0] - lambdas[-2, 0]) < obsi:
            klambdas = copy(lambdas[: -1, :])
        else:
            klambdas = copy(lambdas[:])
        kses = array(stl_decompose(klambdas[:, 1], period=7, robust=False)["seasonal"][0:7], dtype=float)
        kses_order = argsort(kses)
        apattern7d = (kses_order[-1] + 1) % 7
        apattern52d = (kses_order[-2] + 1) % 7
        if (abs(apattern7d - apattern52d) <= 2) and (ksinfo["weekly7"]):
            ksinfo["weekly7"] = False
            ksinfo["weekly52"] = True
            ksinfo["weekchange"] = array([], dtype=int)
        apattern7d = kses_order[-1]
        apattern52d = kses_order[-2]
        apattern7d = arange(apattern7d, len(lambdas[:, 0]), 7)
        apattern52d = arange(apattern52d, len(lambdas[:, 0]), 7)
        apattern52d = sort(append(apattern7d, apattern52d))
        klindeces = lambdas[:, 0]
        apattern7d = klindeces[apattern7d.astype("int")]
        apattern52d = klindeces[apattern52d.astype("int")]
        if ksinfo["weekly52"] and (ksinfo["weekchange"].size == 0):
            ksinfo["weekchange"] = apattern52d
        if ksinfo["weekly7"] and (ksinfo["weekchange"].size == 0):
            ksinfo["weekchange"] = apattern7d
        if ksinfo["weekchange"].size == 0:
            ksinfo["weekchange"] = apattern52d
    if (ksinfo["type"] != 0) and (ksinfo["nchange"] % 2):
        ksinfo["nchange"] = ksinfo["nchange"] + 1
    if (ksinfo["weekchange"].size >= ksinfo["nchange"]) and (ksinfo["weekchange"].size > 0) and (ksinfo["type"] == 3):
        ksinfo["nchange"] = 0
    return ksinfo
