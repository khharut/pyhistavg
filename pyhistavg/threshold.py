from os import environ
from numpy import array, diff, arange, ceil, zeros, where, append, unique, sort, apply_along_axis
from numpy import delete, setdiff1d, median, percentile, concatenate, tile, identity, empty
from numpy import copy, trapz, isnan, mod, vstack, isinf, reshape, nan, var
from numpy import abs as nabs
from numpy import any as nany
from numpy import max as nmax
from numpy import all as nall
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from helper import obsicut, currentday
from time import time
from stationarity import StatLen
from decompose import stl_decompose
from kstest import KSInfo, KSValues
from divider import HistOutliers, DivGroups, HistQuantile, CalcOnGroups
from helper import MinMax, Median, daytrunc, HumanTime, Mean, Signif
from predicter import Predicter, StepY
from baddata import BadInd
from stests import ARIMAWrapper, HurstWrapper, TestS, TestLD
from periodicity import NRel
matplotlib.use("Agg")


def dynthresh(mon_id=None, time_series=None, raw_array=None,
              plot_in_file=False, out_path=None, dmin=None,
              sens="low", ltimestamp=-1):
    """
    Constructing dynamic threshold for monitor. Generic method for any type of data.

    Parameters
    ----------
    mon_id: string
        monitor ID of data

    time_series: 2D numpy array
        time series like 2D array of timestamps (1) and values

    raw_array: 2D numpy array
        2D array of origianl data containing timestamps and values

    out_path: string
        path where plot file should be created

    plot_in_file: boolean
        plot raw data into *.png file or not

    dmin: integer
        number of minutes to aggregate raw data

    sens: string
        one of three values string ["low", "medium", "high"]

    ltimestamp: integer
        timestamp of calculation Unix-time

    Returns
    -------
    outdict: dictionary
        dictionary containing dynamic thresholds, error and warning messages, reliability and periodicity
        dynamic thresholds given in format of list of lists ["time", lower_value, upper_value]....
    """
    ##################################################################
    # setting default values of parameters ###########################
    if (sens != "low") and (sens != "medium") and (sens != "high"):
        sens = "low"
    th_reliab = 0.0
    per_reliab = 0.0
    interval = 0.0
    environ['TZ'] = 'UTC'
    status = 1
    per7d = False
    statio = False
    peri = False
    hinto = False
    trendy = False
    hurd_thresh = 0.0
    pred_reliab = 0.0
    xstat = 0.0
    pquant = 1
    cquant = 100.0 - 2.0 * pquant
    cquant = cquant * 0.01
    probs = [pquant, 100 - pquant]
    errmess = " "
    warnmes = " "
    image_name = " "
    synchronisable = True
    ##################################################################
    # checking if input data is not empty and threshold can be #######
    # calculated for given time, synchronisable parameter ############
    if (time_series is not None) and (raw_array is not None):
        max_time = raw_array[-1, 0]
        if ltimestamp > 0:
            delta_time = 1000 * max_time - ltimestamp
            if abs(delta_time) > (1000 * 24 * 60 * 60):
                synchronisable = False
    ##################################################################
    #  in case when input data is not empty and threshold can be #####
    #  calculated for given time, doing calculation ##################
    if (time_series is not None) and (raw_array is not None) and (synchronisable):
        varx = var(raw_array, ddof=1)
        sec_of_day = 24 * 60 * 60
        nstep = len(time_series[:, 0])  # number of elements in time series
        dtime = time_series[1, 0] - time_series[0, 0]
        delta_min = dtime / 60.0  # time step of time series
        interval = delta_min
        tsfreq = int(ceil(24.0 * 60.0 * 60.0 / dtime))
        ymax = max(raw_array[:, 1])  # maximum and minimum values for
        ymin = min(raw_array[:, 1])  # raw, i.e. not time series data
        daynum = float(nstep) / tsfreq  # number of days in data
        ##################################################################
        if ymin >= 0:  # checking if input data is positiv
            positive = True  # for keeoing threshold posiitve in any case
        else:
            positive = False
        p_value = 0.35  # minimal p.value for periodicity test, if p.value lower than this value then time series is treated as non-periodic
        r_value = 2  # minimal value of difference between upper and lower thresholds, this value mostly used for constant input data
        if r_value > ymax:
            r_value = ymax
        last_time = daytrunc(raw_array[-1, 0])  # timestamp of last day 0:0,
        if dmin is None:  # setting default value for threshold calculation time step
            dmin = 2.0 * delta_min
        else:  # in case when it is not well defined by user correcting it
            if (dmin <= 0) or (dmin > (12.0 * 60.0)) or (dmin < (2 * delta_min)):
                dmin = 2.0 * delta_min
                warnmes = warnmes + "Unexpected value of threshold time step. It has been corrected."
        if (daynum >= 3.0):  # doing trend+season+reminder decomposition using stl method written in Fortran, this requires minimum 3 day data
            x_trend = stl_decompose(time_series[:, 1], period=tsfreq, robust=True)
            x_trend = x_trend['trend']  # taking only trend part of decomposition
        else:
            x_trend = zeros(nstep)  # if no trend data can be extraced then setting it to zeros
        ##################################################################
        rhoval = ARIMAWrapper(time_series[:, 1])  # taking rho value from ARIMA process, it shows if our time series is stationar
        hint = HurstWrapper(time_series[:, 1])  # taking Hurst expoent of time series, it another measure of stationarity
        if (hint > 0.9):  # in case when hint > 0.9, it is very likely that we have non-stationar time series
            hinto = True
        apeak7d1 = NRel(time_series[:, 1], 7 * tsfreq)  # getting p.value for weekly periodicity for original time series
        apeak7d2 = NRel(x_trend, 7 * tsfreq)  # getting p.value for weekly periodicity for trend of original data
        apeak = NRel(time_series[:, 1], tsfreq)  # getting p.value for daily periodicity on original time series
        ksvals = KSValues(time_series[:, 1], tsfreq)  # getting Kolmogorov's lambda values for every 2 * tsfreq values in time series
        ksdata = KSInfo(ksvals)  # getting extra information from Kolmogorov's lambdas, like weekly pattern, stationairyty changes and others
        ##################################################################
        if ksdata["type"] < 3:  # analysing lambda values and getting extra info about data
            ksvals0 = KSValues((time_series[:, 1] - x_trend), tsfreq)
            ksdata0 = KSInfo(ksvals0)
            if ksdata["type"] >= ksdata0["type"]:
                del ksdata0
                del ksvals0
            else:
                ksdata["weekchange"] = copy(ksdata0["weekchange"])
                ksdata["type"] = int(ksdata0["type"])
                ksdata["weekly7"] = bool(ksdata0["weekly7"])
                ksdata["weekly52"] = bool(ksdata0["weekly52"])
                del ksdata0
                del ksvals0
                if (apeak7d2[1] == 0) and (TestS(x_trend) > 0) and (hinto):
                    trendy = True
                    warnmes = warnmes + "Trend in data is detected."
        prob_num = ksdata["nchange"]  # possible number of changes in time series, estimated using lambda values
        apeak7d = array([7 * tsfreq, max(apeak7d1[1], apeak7d2[1])])  # p.value for weekly periodicty
        ##################################################################
        # different and messy logic for time series properties like:
        # stationarity, periodicity, weekly periodicity
        if (apeak7d1[1] > 0.8) and (apeak7d2[1] > 0.8) and (not hinto):
            statio = True
            per7d = True
        if (apeak7d1[1] > 0.7) and (apeak7d2[1] > 0.7) and (apeak[1] > 0.7) and (not hinto):
            statio = True
        if (apeak7d1[1] > 0.8) and (apeak[1] < p_value) and (not hinto):
            statio = True
            per7d = True
        if (apeak7d1[1] > 0.6) and (apeak[1] < p_value) and ((apeak[1] > p_value) or (apeak7d2[1] > p_value)) and (not hinto):
            per7d = True
        if (apeak7d2[1] > 0.6) and (apeak[1] < p_value) and ((apeak[1] > p_value) or (apeak7d1[1] > p_value)) and (not hinto):
            per7d = True
        if (apeak7d1[1] > 0.5) and (apeak7d2[1] > 0.5) and (ksdata["type"] > 0):
            per7d = True
        if (apeak7d1[1] > 0.9) and (apeak7d2[1] > p_value) and (not hinto):
            statio = True
            per7d = True
        if (apeak7d1[1] > 0.6) and (apeak[1] > p_value) and (ksdata["type"] >= 2):
            per7d = True
        if (apeak7d2[1] > 0.9) and (not hinto):
            if apeak[1] > 0:
                statio = True
            else:
                peri = True
            if apeak7d1[1] > 0:
                per7d = True
        if (apeak[1] > 0.8) and (not hinto):
            peri = True
        if (apeak[1] > 0.9) and (not hinto):
            statio = True
        if (apeak7d1[1] > 0.75) and (apeak7d2[1] > 0.75):
            per7d = True
        if (apeak7d1[1] < apeak[1]) and ((0.5 * (apeak7d1[1] + apeak7d2[1])) < apeak[1]):
            per7d = False
        if (rhoval > 0.92) and (hint >= 0.82):
            statio = False
        if ksdata["highly_unstable"]:
            statio = False
        if (not statio) and (apeak7d[1] > 0.6):
            peri = True
        if (not statio) and (apeak[1] > 0.6):
            peri = True
        if prob_num == 0:
            statio = True
        if (apeak7d[1] > 0):
            if ((apeak7d1[1] * apeak7d2[1]) > 0) and (ksdata["type"] > 1):
                per7d = True
            if (min((apeak[1], apeak7d[1])) > 0.6) and (not per7d):
                per7d = (ksdata["weekly7"] or ksdata["weekly52"])
            if (apeak7d1[1] > p_value) and (apeak7d2[1] > 0.7) and (not per7d) and (ksdata["type"] > 0):
                per7d = True
            if (apeak7d2[1] > p_value) and (apeak7d1[1] > 0.7) and (not per7d) and (ksdata["type"] > 0):
                per7d = True
            if (apeak7d[1] > 0.6) and (apeak[1] > 0.7) and (not per7d) and (ksdata["type"] > 0):
                per7d = True
            if (apeak[1] > p_value) and (apeak7d1[1] > 0.7) and (not per7d) and (ksdata["type"] > 0):
                per7d = True
            if (apeak7d[1] > p_value) and (apeak[1] > 0.5) and (not per7d) and (ksdata["type"] > 0):
                per7d = True
        ##################################################################
        #  in case of non-stationary time series (TS) (statio = False) getting#
        #  positions of stationarity changes (indeces), otherwise mimics##
        #  structure of StatLen function return values ###################
        if not statio:
            allchindex = StatLen(time_series[:, 1], tsfreq, hinto, peri, prob_num)
        else:
            allchindex = {"strong": [0, (nstep - 1)], "weak": [0, (nstep - 1)]}
        ##################################################################
        chindex = allchindex['strong']
        tchindex = allchindex['weak']
        tchindex = array(tchindex[1: (-1)])
        chindex0 = array(chindex[1: (-1)])
        tchindex = setdiff1d(tchindex, chindex0)
        ##################################################################
        #  removing more than 2 stationary change points in last day of TS
        if (len(where(chindex0 >= (nstep - tsfreq))[0]) > 2):
            chindex0 = append(chindex0[where(chindex0 < (nstep - tsfreq))[0]], min(chindex0[where(chindex0 >= (nstep - tsfreq))[0]]))
            chindex0 = append(chindex0, max(chindex0[where(chindex0 >= (nstep - tsfreq))[0]]))
        ##################################################################
        #  removing stationary change points that appear in TS due to ####
        #  weekly periodicity ############################################
        if (per7d):
            if (chindex0.size > 0) and (ksdata["weekchange"].size > 0):
                kweekpattern = map(lambda(w): min(abs(ksdata["weekchange"] - w)), chindex0)
                kch7ind = array([w < (0.75 * tsfreq) for w in kweekpattern])
                chindex0 = chindex0[~kch7ind]
                if (chindex0.size >= 2) and (not trendy):
                    weekpattern0 = reshape(tile(chindex0, chindex0.size), (chindex0.size, chindex0.size)).T
                    ident_mat = nmax(weekpattern0) * identity(chindex0.size)
                    weekpattern0 = abs(weekpattern0 - chindex0) + ident_mat
                    if nany(weekpattern0 < tsfreq):
                        weekpattern0 = apply_along_axis(lambda(w): obsicut(w, tsfreq), 1, weekpattern0)
                    weekpattern0 = weekpattern0 - ident_mat
                    weekpattern0 = weekpattern0 / float(tsfreq)
                    weekpattern0 = (weekpattern0 - 3.5) % 7 + 3.5
                    weekpattern0 = weekpattern0 - 8 * identity(chindex0.size)
                    ch7ind0 = where((weekpattern0 > 6.98) & (weekpattern0 < 7.02))
                    ch7ind0 = append(ch7ind0[0], ch7ind0[1])
                    ch7ind0 = sort(unique(ch7ind0))
                    ch7ind0 = setdiff1d(arange(chindex0.size), ch7ind0)
                    if (ch7ind0.size > 0) and (chindex0.size > ch7ind0.size):
                        chindex0 = chindex0[ch7ind0]
                    else:
                        if ch7ind0.size == 0:
                            chindex0 = empty(0, dtype=int)
                        if ch7ind0.size == chindex0.size:
                            per7d = False
        ##################################################################
        #  removing stationary change points that appear in TS due to ####
        #  weekly periodicity, in case when weekly periodicty is hard to##
        #  detect or last part of code fails to remove them completely####
        if ((apeak7d[1] > 0.72) and (not ksdata["highly_unstable"])) or ((per7d) and (ksdata["type"] == 0) and (not ksdata["highly_unstable"])):
            if (chindex0.size > 1) and (ksdata["weekchange"].size == 0):
                weekpattern = reshape(tile(chindex0, chindex0.size), (chindex0.size, chindex0.size)).T
                ident_mat = nmax(weekpattern) * identity(chindex0.size)
                weekpattern = abs(weekpattern - chindex0) + ident_mat
                if nany(weekpattern < tsfreq):
                    weekpattern = apply_along_axis(lambda(w): obsicut(w, tsfreq), 1, weekpattern)
                weekpattern = weekpattern - ident_mat
                weekpattern = weekpattern / float(tsfreq)
                weekpattern = (weekpattern - 3.5) % 7 + 3.5
                weekpattern = weekpattern - 8 * identity(chindex0.size)
                ch7ind = where((weekpattern > 6.95) & (weekpattern < 7.05))
                ch7ind = append(ch7ind[0], ch7ind[1])
                ch7ind = sort(unique(ch7ind))
                ch7ind = setdiff1d(arange(chindex0.size), ch7ind)
                if (ch7ind.size > 0) and (chindex0.size > ch7ind.size):
                    chindex0 = chindex0[ch7ind]
                    per7d = True
                else:
                    if ch7ind.size == 0:
                        chindex0 = empty(0, dtype=int)
        ##################################################################
        #  removing any statioanry change points of last day affected by##
        #  missing data in it, TestLD detects them and then they are removed
        if chindex0.size > 0:
            if (last_time - time_series[chindex0[-1], 0]) < sec_of_day:
                last_chindex = chindex0[(last_time - time_series[chindex0, 0]) < sec_of_day]
                lstest = TestLD(raw_array[:, 0], time_series[last_chindex, 0], tsfreq)
                if last_chindex.size > 1:
                    deltax = max(time_series[last_chindex, 0]) - min(time_series[last_chindex, 0])
                else:
                    deltax = 12.0 * dtime
                if (lstest > 0) and (deltax > (12.0 * dtime)):
                    chindex0 = setdiff1d(chindex0, last_chindex)
        ##################################################################
        #  removing stationary changes and data from both TS and raw data#
        #  in case when time difference between to stationary changes is #
        #  less than one day and before and after these changes TS behaves
        #  similarly######################################################
        bad_data = BadInd(time_series[:, 0], time_series[:, 1], raw_array[:, 0], chindex0, hinto)
        if (len(bad_data['indices']) > 0):
            raw_array = delete(raw_array, bad_data['indices'], 0)
            temp_chindex = setdiff1d(chindex0, bad_data['stat'])
            tchindex = setdiff1d(tchindex, temp_chindex)
            chindex0 = bad_data['stat']
            if max(raw_array[:, 0]) < max(time_series[:, 0]):
                time_series = delete(time_series, where(time_series[:, 0] > max(raw_array[:, 0]))[0], 0)
                nstep = len(time_series[:, 0])
                daynum = float(nstep) / tsfreq
            if min(raw_array[:, 0]) > min(time_series[:, 0]):
                time_series = delete(time_series, where(time_series[:, 0] < min(raw_array[:, 0]))[0], 0)
                nstep = len(time_series[:, 0])
                daynum = float(nstep) / tsfreq
            redefine_metric_values = interp1d(x=raw_array[:, 0], y=raw_array[:, 1], kind='nearest')
            time_series[:, 1] = redefine_metric_values(time_series[:, 0])
            if (len(chindex0) > 0):
                chindex0 = chindex0[where((time_series[chindex0, 0] <= max(raw_array[:, 0])) & (time_series[chindex0, 0] >= min(raw_array[:, 0])))]
        ##################################################################
        #  filtering less possible stationary changes in case when any ###
        #  lambda value greater than 3, i.e. it is very likely than we can
        #  miss some stationary change or it can be better to use TS data#
        #  after that point###############################################
        if nany(ksvals[:, 1] > 3):
            if (per7d):
                if (tchindex.size > 0) and (nany(tchindex < (nstep - 14 * tsfreq))):
                    chindex10 = max(tchindex[tchindex < (nstep - 14 * tsfreq)])
                else:
                    chindex10 = empty(0, dtype=int)
            else:
                if (tchindex.size > 0) and (nany(tchindex < (nstep - 7 * tsfreq))):
                    chindex10 = max(tchindex[tchindex < (nstep - 7 * tsfreq)])
                else:
                    chindex10 = empty(0, dtype=int)  # less possible stationary change point happened 10 days before the last day in data
        else:
            chindex10 = empty(0, dtype=int)  # less possible stationary change point happened 10 days before the last day in data, default value
        ##################################################################
        if chindex0.size > 0:  # getting last stationary change position to cut TS and raw data (RD) after that poisition
            chindex = chindex0[-1]
        else:
            chindex = 0
        chtime = time_series[chindex, 0]
        if chindex0.size > 1:
            chtime0 = time_series[chindex0[: -1], 0]  # last stationary change time
        else:
            chtime0 = empty(0, dtype=float)  # last stationary change time, default value
        ##################################################################
        if nany(ksvals[:, 1] > 3):  # in case when no stationary change exists then using less posible stationary change position to cut TS and RD
            if (nstep > (14 * tsfreq)) and (chindex == 0) and (chindex10.size > 0) and (per7d):
                chindex = chindex10
                chtime = time_series[chindex, 0]
            if (nstep > (7 * tsfreq)) and (chindex == 0) and (chindex10.size > 0) and (not per7d):
                chindex = chindex10
                chtime = time_series[chindex, 0]
        ##################################################################
        if (nstep > (10 * tsfreq)) and (chindex == 0) and (NRel(time_series[:, 1], tsfreq)[1] > p_value) and (chindex10.size > 0):
            chindex = chindex10
            chtime = time_series[chindex, 0]
        ##################################################################
        if (chindex > 0):  # cutting RD and TS data after statioanry change time/position
            time_series = time_series[min((chindex + 1), (nstep - 1)):, :]
            ksvals[:, 0] = ksvals[:, 0] - chindex
            ksvals = ksvals[ksvals[:, 0] >= tsfreq, :]
            ksdata = KSInfo(ksvals)
            apeak = NRel(time_series[:, 1], tsfreq)
            nstep = len(time_series[:, 0])
            daynum = float(nstep) / tsfreq
            if daynum < 14:
                per7d = False
            hint = HurstWrapper(time_series[:, 1])
            if (hint > 0.9):
                hinto = True
            else:
                hinto = False
            rchindex = min(where(raw_array[:, 0] > chtime)[0])
        else:
            nstep = len(time_series[:, 0])
            daynum = float(nstep) / float(tsfreq)
            if (len(bad_data['indices']) > 0):
                apeak = NRel(time_series[:, 1], tsfreq)
            rchindex = 0
        ##################################################################
        #  checking if our data has some trend or any other highly instabilty
        #  for using different algorithm for threshold construction
        if (ksdata["highly_unstable"]) or hinto:
            xstat = TestS(time_series[:, 1])
            if (daynum > 3) and (xstat > 0):
                time_series = time_series[(nstep - 3 * tsfreq):, :]
                daynum = 3
        if (ksdata["highly_unstable"]) and (xstat == 0) and (TestS(time_series[(nstep - tsfreq):, 1]) > 0):
            time_series = time_series[(nstep - tsfreq):, :]
            daynum = 1
        if (ksdata["highly_unstable"]) and (xstat == 0) and (TestS(time_series[(nstep - tsfreq):, 1]) == 0):
            apeak[1] = 0.8
        ##################################################################
        #  when p.value of weekly periodicty is higher then using it #####
        #  instead of daily one###########################################
        ##################################################################
        if (per7d) and (apeak7d[1] > apeak[1]):
            apeak[1] = apeak7d[1]
        ##################################################################
        ##################################################################
        per_reliab = 100.0 * apeak[1]
        y_med = median(raw_array[rchindex:, 1])  # median of RD, rchindex is stationary change index in RD
        varx = var(raw_array[rchindex:], ddof=1)  # variance of RD
        y_len = (len(raw_array[:, 1]) - rchindex)  # length of RD
        mindelta = nabs(diff(raw_array[rchindex:, 1]))  # minimal differnce of consecutive elements in RD
        if nany(mindelta > 0):
            mindelta = min(mindelta[mindelta > 0])  # non-zero value of minimal difference is used for some cases
        else:
            mindelta = r_value  # if all differnces are zero then using r_value, it means input data is constant
        probs = probs + [(100.0 * float(7 * tsfreq - 1)) / float(7 * tsfreq), (100 * float(tsfreq - 1)) / float(tsfreq)]  # probabilities for calculating percentiles
        y_quan = percentile(raw_array[rchindex:, 1], probs)  # calulating 4 percentiles, 2 for dynamic thresholds, 2 for detecting daily peaks
        hard_thresh = y_quan[2: 4]  # percentile thresholds for single peaks
        y_quan = y_quan[0: 2]  # percentile thresholds for dynamic thresholds
        probs = probs[0: 2]  # ommiting additional percents in
        ##################################################################
        #  detecting if any single daily peaks exists in TS for treating
        #  such a TS as a periodic one####################################
        if (apeak[1] >= 0.2) and (apeak[1] <= 0.5) and (daynum >= 3) and nany(time_series[:, 1] > hard_thresh[1]):
            day_spikes = where(time_series[:, 1] > hard_thresh[1])[0]
            dday_spikes = diff(day_spikes) - 1
            dday_spikes = dday_spikes.astype(float)
            ddday_spikes = nabs((dday_spikes - 0.5 * tsfreq) % tsfreq - 0.5 * float(tsfreq))
            if nall(ddday_spikes <= 5.0) and nall(dday_spikes < 3.0 * tsfreq) and nany(dday_spikes > (0.95 * tsfreq)) and (day_spikes[-1] >= (nstep - tsfreq)):
                apeak[1] = max(sum(ddday_spikes == 0) / day_spikes.size, apeak[1])
                per_reliab = 100.0 * apeak[1]
        else:
            if ((apeak7d[1] > 0.1) and (not per7d) and (daynum >= 21) and nany(time_series[:, 1] > hard_thresh[0])):
                week_spikes = where(time_series[:, 1] > hard_thresh[0])[0]
                dweek_spikes = diff(week_spikes) - 1
                dweek_spikes = dweek_spikes.astype(float)
                ddweek_spikes = nabs((dweek_spikes - 3.5 * float(tsfreq)) % (7 * tsfreq) - 3.5 * float(tsfreq))
                if nall(ddweek_spikes <= 3.0) and nall(dweek_spikes < 15.0 * tsfreq) and nany(dweek_spikes > (6.95 * tsfreq)) and (week_spikes[-1] >= (nstep - 7.0 * tsfreq)):
                    per7d = True
                    if (apeak[1] < apeak7d[1]) and (apeak7d[1] >= 0.5):
                        apeak[1] = apeak7d[1]
                        per_reliab = 100 * apeak[1]
        ##################################################################
        #  correcting percentile values in case when our data is highly unstable
        if rhoval > 0.9 and hinto:
            y_quan = sort(y_quan)
            y_dquan = 0.5 * nabs(diff(y_quan))
            if (y_med > ymin) and (y_med < ymax):
                y_quan[0] = max([y_quan[0], y_med - y_dquan])
                y_quan[1] = max([y_quan[1], y_med + y_dquan])
        if y_quan[0] == y_quan[1]:  # correcting percentile values in case when our data is almost constant upper and lower percentiles are equal
            y_quan[0] = y_quan[0] - 0.5 * r_value
            y_quan[1] = y_quan[1] + 0.5 * r_value
        ##################################################################
        if (xstat == 0) and (y_len > 300):  # in case when RD length greater than 300 and data has no trend (xstat == 0)
            # means that doing dynamic threshold calculation using percentiles
            # Divgroups() projects all raw data into one day and then divide it into groups by their appearance in day time, also
            # one should provide time step by which these groups are organized, chindex is stationary change point index,
            # delta_min time difference between groups, overlap means that some values in neighbour groups can be duplicated
            # looped means that last and first groups are the same
            oneday_proj, x_groups, npattern, ndates = DivGroups(time_stamps=raw_array[:, 0], values=raw_array[:, 1],\
                ch_ind=rchindex, delta_min=dmin, overlap=True, looped=True)
            if daynum >= 2:
                if daynum >= 7:  # using percentiles when we have more than 7 days data in RD
                    npout = 2 * int(round(dmin * tsfreq / (24.0 * 60.0)))  # number of points allowed to go out
                    #  CalcOnGroups() calculates func_calc on groups, in case when it impossible
                    #  default function func_def is used, HistQuantile uses percentiles for its calculations
                    dyn_thresh = CalcOnGroups(x_groups, func_calc = lambda(w): HistQuantile(w, probs, y_quan, npout, y_med), func_def=Mean)
                    hist_outliers = HistOutliers(x_groups, dyn_thresh)  # getting distances from thresholds for point that are out of them
                    if (len(hist_outliers) > (0.01 * y_len)):  # checking if we have enough point for further calculation
                        deltas = StepY(hist_outliers, sens)  # calculating corrections for thresholds, idea very similair to 3 sigma
                    else:
                        deltas = zeros(2)  # when there is not enough points, setting corrections to 0
                        if nany(nabs(hist_outliers) > mindelta) and len(hist_outliers) > daynum:
                            deltas = array([-mindelta, mindelta])  # but in some cases mindelta can be used as corrections
                else:
                    # using prediction when we have more than 2 days but less than 7 days data in RD
                    data_predicter = Predicter(time_series[:, 0], time_series[:, 1], is_positive=positive, trend_only=False, psense=sens)
                    pred_reliab = data_predicter["reliability"]  # reliability of prediction model
                    #  doubling timestamp of predictions
                    pdates = tile(data_predicter["thresholds"][:, 0], 2)
                    #  joining upper and lower thresholds of prediction
                    prediction = concatenate((data_predicter["thresholds"][:, 1], data_predicter["thresholds"][:, 2]))
                    #  dividing prediction upper and lower thresholds into groups, idea is the same as for quantiles
                    pred_1dproj, x_pgroups, pred_pattern, ndates = DivGroups(time_stamps=pdates, values=prediction,
                                                                             ch_ind=None, delta_min=dmin, overlap=True, looped=True)
                    dyn_thresh = CalcOnGroups(x_pgroups, func_calc=MinMax, func_def=Median)
                #  calculating minimum and maximum values for groups members, this used later for reliability calculations
                x_minmax = CalcOnGroups(x_groups, func_calc=MinMax, func_def=Median)
                dyn_fthresh = dyn_thresh.copy()
                if (daynum >= 7):  # adding corrections of thresholds to thresholds
                    dyn_thresh[0, :] = dyn_thresh[0, :] - abs(deltas[0])
                    dyn_thresh[1, :] = dyn_thresh[1, :] + abs(deltas[1])
                if (nany(nabs(dyn_thresh[0, :] - dyn_thresh[1, :]) < mindelta)) and (mindelta > 0):
                    #  in case when upper and lower thresholds are the same
                    ind_eq = where(nabs(dyn_thresh[0, :] - dyn_thresh[1, :]) < mindelta)[0]
                    ##################################################################
                    #  adding 0.5*mindelta value to upper and lower threhsolds when ##
                    #  they are equal#################################################
                    dyn_thresh[0, ind_eq] = dyn_thresh[0, ind_eq] - (0.5 * mindelta)
                    dyn_thresh[1, ind_eq] = dyn_thresh[1, ind_eq] + (0.5 * mindelta)
                if varx > 0:
                    #  if data is not constant correcting any threshold values outside [ymin, ymax] interval
                    dyn_thresh[0, dyn_thresh[0, :] < ymin] = ymin
                    dyn_thresh[1, dyn_thresh[1, :] > ymax] = ymax
                    dyn_fthresh[0, dyn_fthresh[0, :] < ymin] = ymin
                    dyn_fthresh[1, dyn_fthresh[1, :] > ymax] = ymax
            ##################################################################
            #  calculating constant dynamic threshold, it used in calculation
            #  of reliability of dynamic thresholds
            dyn_cthresh = zeros((2, len(ndates)))
            dyn_cthresh[0, :] = y_quan[0]  # using just y_quan values
            dyn_cthresh[1, :] = y_quan[1]
            hist_coutliers = HistOutliers(x_groups, dyn_cthresh)  # getting distances from thresholds for point that are out of them
            cquant = 1.0 - float(len(hist_coutliers)) / float(y_len)
            if (len(hist_coutliers) > (0.01 * y_len)):  # the same logic as for non-constant threshold, corrections for consant thresholds
                cdeltas = StepY(hist_coutliers, sens)
            else:
                cdeltas = zeros(2)
                if nany(nabs(hist_coutliers) > mindelta) and len(hist_coutliers) > daynum:
                    cdeltas = array([-mindelta, mindelta])
            dyn_fcthresh = copy(dyn_cthresh)
            dyn_cthresh[0, :] = dyn_cthresh[0, :] - abs(cdeltas[0])
            dyn_cthresh[1, :] = dyn_cthresh[1, :] + abs(cdeltas[1])
            ##################################################################
            #  adding 0.5*mindelta value to upper and lower threhsolds when ##
            #  they are equal#################################################
            if (dyn_cthresh[0, 0] == dyn_cthresh[1, 0]) and (mindelta > 0):
                dyn_cthresh[0, :] = dyn_cthresh[0, :] - (0.5 * mindelta)
                dyn_cthresh[1, :] = dyn_cthresh[1, :] + (0.5 * mindelta)
            if varx > 0:
                #  if data is not constant correcting any threshold values outside [ymin, ymax] interval
                dyn_cthresh[0, dyn_cthresh[0, :] < ymin] = ymin
                dyn_cthresh[1, dyn_cthresh[1, :] > ymax] = ymax
            ##################################################################
            #  calculating reliability of dynamic threshold, it just measures#
            #  how good this threshold covers historical points###############
            if daynum >= 2:
                dth1 = x_minmax[1, :] - dyn_thresh[1, :]
                dth1[(dth1 < 0)] = 0.0
                dth2 = dyn_thresh[0, :] - x_minmax[0, :]
                dth2[(dth2 < 0)] = 0.0
                s0 = trapz(y=(x_minmax[1, :] - x_minmax[0, :]), x=ndates)
                if (isnan(s0)) or (isinf(s0)):
                    s0 = 0.0
                if (s0 == 0.0):
                    s0 = (ndates[-1] - ndates[0]) * r_value
                if daynum >= 7:
                    dthresh_reliab = trapz(y=(dth1 + dth2), x=ndates) / s0
                    thresh_reliab = 1.0 - dthresh_reliab
                else:
                    thresh_reliab = pred_reliab
                ##################################################################
                cdth1 = x_minmax[1, :] - dyn_cthresh[1, :]
                cdth1[(cdth1 < 0)] = 0.0
                cdth2 = dyn_cthresh[0, :] - x_minmax[0, :]
                cdth2[(cdth2 < 0)] = 0.0
                cdth3 = x_minmax[1, :] - dyn_cthresh[1, :]
                cdth3[(cdth1 > 0)] = 0.0
                cdth4 = dyn_cthresh[0, :] - x_minmax[0, :]
                cdth4[(cdth4 > 0)] = 0.0
                cthresh_reliab = cquant * (1.0 - (trapz(y=(cdth1 + cdth2), x=ndates) / s0))
                cthreshin_reliab = trapz(y=(nabs(cdth3) + nabs(cdth4)), x=ndates) / s0
                ##################################################################
                if daynum >= 7:
                    #  making decision on what threshold to use: prediction, constant or percentile
                    if (cthreshin_reliab < dthresh_reliab) and (apeak[1] >= 0.2) and (apeak[1] < 0.75):
                        #  in case when reliability of constant threshold is higher and
                        #  data is not so periodic then use consant threshold
                        th_reliab = thresh_reliab / cthresh_reliab
                        dyn_thresh = dyn_cthresh
                        dyn_fthresh = dyn_fcthresh
                    if (cthreshin_reliab >= dthresh_reliab) and (apeak[1] >= 0.2) and (apeak[1] < 0.75):
                        #  in case when reliability of constant threshold is lower and
                        #  data is not so periodic then use percentile threshold
                        th_reliab = thresh_reliab
                    if apeak[1] >= 0.75:
                        #  in case when data is periodic then use percentile threshold
                        th_reliab = thresh_reliab
                    if (apeak[1] < 0.2) and (cthresh_reliab <= thresh_reliab):
                        #  in case when reliability of constant threshold is higher and
                        #  data is not periodic then use consant threshold
                        th_reliab = cthresh_reliab
                        dyn_thresh = dyn_cthresh
                        dyn_fthresh = dyn_fcthresh
                    if (apeak[1] < 0.2) and (cthresh_reliab > thresh_reliab):
                        #  in case when reliability of constant threshold is higher and
                        #  data is not so periodic then use consant threshold
                        th_reliab = thresh_reliab / cthresh_reliab
                        dyn_thresh = dyn_cthresh
                        dyn_fthresh = dyn_fcthresh
                else:
                    if (thresh_reliab > 0.70) and (apeak[1] > p_value):
                        #  in case when reliability of prediction threshold is higher than 0.7
                        #  and data very periodic then use periodic threshold
                        th_reliab = thresh_reliab
                        warnmes = warnmes + "Last stable period is too short. Prediction based automatic thresholds is used."
                    else:
                        #  otherwise use consant threshold
                        th_reliab = cthresh_reliab
                        dyn_thresh = dyn_cthresh
                        dyn_fthresh = dyn_fcthresh
                ##################################################################
                th_reliab = 100.0 * th_reliab
            else:
                #  in case when only constant threshold can be provided
                #  then use consant threshold
                dyn_thresh = dyn_cthresh
                dyn_fthresh = dyn_fcthresh
                th_reliab = 100.0 * cquant * (dyn_thresh[1, 0] - dyn_thresh[0, 0]) / (ymax - ymin)
            status = 0
        ##################################################################
        #  in case when we have less than 2 days data, it has no trend ###
        #  number of elements in it greater than 10, then just returning##
        #  constant threshold based on percentiles #######################
        if (status == 1) and (daynum > 1) and (xstat == 0) and (y_len > 10):
            ndates = arange(last_time, (last_time + 24.0 * 60.0 * 60.0), dmin * 30)
            ndates = append(ndates, (last_time + 24.0 * 60.0 * 60.0))
            dyn_thresh = zeros((2, len(ndates)))
            dyn_thresh[0, :] = y_quan[0]
            dyn_thresh[1, :] = y_quan[1]
            dyn_fthresh = dyn_thresh
            if ymin != ymax:
                th_reliab = min(((100.0 - 2.0 * pquant) * (y_quan[1] - y_quan[0]) / (ymax - ymin)), (100.0 - 2.0 * pquant))
            else:
                th_reliab = 100.0
            status = 0
        ##################################################################
        ##################################################################
        #  when data has trend (xstat > 0) using trend prediction algorithm
        #  to estimate dynamic thresholds ###############################
        if (status == 1) and (daynum >= 1) and (xstat > 0):
            nstep = len(time_series[:, 1])  # calling Predicter method from predicter.py
            trend_predicter = Predicter(time_series[(nstep - tsfreq):, 0], time_series[(nstep - tsfreq):, 1], is_positive=positive, trend_only=True, psense=sens)
            th_reliab = 100.0 * trend_predicter["reliability"]
            tpdates = tile(trend_predicter["thresholds"][:, 0], 2)
            tprediction = concatenate((trend_predicter["thresholds"][:, 1], trend_predicter["thresholds"][:, 2]))
            pred_1dproj, x_pgroups, pred_pattern, ndates = DivGroups(time_stamps=tpdates, values=tprediction, ch_ind=None, delta_min=dmin, overlap=True, looped=False)
            dyn_thresh = CalcOnGroups(x_pgroups, func_calc=MinMax, func_def=Median)
            dyn_fthresh = dyn_thresh
            warnmes = warnmes + "Data likely is having increasing or decreasing trend. Trend prediction based automatic thresholds is used."
            status = 0
        if th_reliab > 100:
            th_reliab = 98.0
        ##################################################################
        #  ploting analysis result in png file if requested##############
        if plot_in_file:
            oneday_time = HumanTime(mod(raw_array[rchindex:, 0], (24 * 60 * 60)) + last_time)
            if mon_id is None:
                image_name = str(long(1000000 * time())) + ".png"
            else:
                image_name = mon_id + ".png"
            if out_path is not None:
                image_name = out_path + image_name
            plt.subplot(211)
            plt.title('Raw data.', fontsize=10)
            plt.tick_params(labelsize=7)
            plt.ylabel("value", fontsize=7)
            plt.plot(HumanTime(raw_array[:, 0]), raw_array[:, 1], "g-")
            for i in chtime0:
                plt.axvline(x=HumanTime(i), color="#302b2b")
            plt.axvline(x=HumanTime(chtime), color="#de3737")
            plt.subplot(212)
            if status == 0:
                plt.title('Dynamic thresholds.', fontsize=10)
                plt.plot(oneday_time, raw_array[rchindex:, 1], c='#9e9b9b', marker='o', markeredgecolor='None', markersize=2, linestyle='None')
            else:
                plt.title('Last stable period.', fontsize=10)
            plt.tick_params(labelsize=7)
            plt.xlabel("time", fontsize=7)
            plt.ylabel("value", fontsize=7)
            if status == 0:
                hndates = HumanTime(ndates)
                plt.plot(hndates, dyn_thresh[0, :], "r-")
                plt.plot(hndates, dyn_thresh[1, :], "r-")
                plt.plot(hndates, dyn_fthresh[0, :], "r--")
                plt.plot(hndates, dyn_fthresh[1, :], "r--")
            else:
                hrawdates = HumanTime(raw_array[rchindex:, 0])
                plt.plot(hrawdates, raw_array[rchindex:, 1], "g-")
            plt.savefig(image_name, format="png")
            plt.close()
        ##################################################################
        #  generating differnt warning and error mesages for different ###
        #  cases #########################################################
        if per7d:
            warnmes = warnmes + "Seven days periodicity is detected. Please use seven day automatic thresholds."
        if status == 0:  # generating output dictionary with threshold, reliability and etc....
            if ltimestamp > 0:
                ndates = ndates - last_time + currentday(ltimestamp)
            dyn_thresh = vstack((ndates, dyn_thresh)).T
            dyn_thresh = dyn_thresh.tolist()
            for i in dyn_thresh:
                i[0] = long(1000 * i[0])
                i[1] = Signif(i[1])
                i[2] = Signif(i[2])
            outdict = {"data": dyn_thresh, "reliability": Signif(th_reliab), "periodicity": Signif(per_reliab), "sens": sens,
                       "image": image_name, "errMsg": errmess, "warnMsg": warnmes, "status": status, "interval": interval}
        else:  # when no analysis is done due to some reasons generating empty threshold data and error message indicating reason
            if xstat >= 0:
                errmess = errmess + "Impossible to calculate automatic thresholds. At least 2 days of stable data should be supplied."
            if xstat < 0:
                errmess = errmess + "Impossible to calculate automatic thresholds. Data is highly unstable."
            outdict = {"data": array(()).tolist(), "reliability": th_reliab, "periodicity": per_reliab, "sens": sens,
                       "image": image_name, "errMsg": errmess, "warnMsg": warnmes, "status": status, "interval": interval}
    else:
        #  when no analysis can be done crating generating empty threshold data and error message indicating reason
        if synchronisable:
            errmess = errmess + "No input data is provided."
        else:
            errmess = errmess + "Input data is out of date."
        outdict = {"data": array(()).tolist(), "reliability": th_reliab, "periodicity": per_reliab, "sens": sens,
                   "image": image_name, "errMsg": errmess, "warnMsg": warnmes, "status": status, "interval": interval}
    return outdict
