from preprocessor import Preprocessor, SJSONReader, MJSONReader, MSJSONReader
from threshold import dynthresh
from pandas.json import dumps


def HistAvg(json_str, plot_in_file=False, out_path=None,
            dmin=None, sens="low", last_ts=-1):
    """
    Reading JSON file or string and returns quantile based threshold for one day.
    Acceptable JSON format is explained in SJSONReader method of preprocessor module.

    Parameters
    ----------
    json_str: string
        JSON string or filepath

    plot_in_file: boolean
        plot raw data into *.png file or not

    out_path: string
        path where plot file should be created

    dmin: integer
        number of minutes to aggregate raw data

    sens: string
        one of three values string ["low", "medium", "high"]

    last_ts: integer
        timestamp of calculation Unix-time

    Returns
    -------
    outjson_str: string
        dynamic threshold in JSON string format, i.e. [[timestamp, lower, upper]...]
    """
    dynout_pattern = dynthresh()
    outdict = []
    monid = None
    monid, rarray, errm, status = SJSONReader(json_str)
    if status == 0:
        tseries, errm, status = Preprocessor(rarray)
    else:
        outdict = dynout_pattern
        outdict["errMsg"] = errm
    if status == 0:
        outdict = dynthresh(monid, tseries, rarray, plot_in_file, out_path, dmin, sens, last_ts)
    else:
        outdict = dynout_pattern
        outdict["errMsg"] = errm
    if monid is not None:
        outdict["monitor_id"] = monid
    outjson_str = dumps(outdict)
    return outjson_str


def MAHistAvg(json_str, plot_in_file=False,
              out_path=None, dmin=None, last_ts=-1):
    """
    Reading JSON file or string and returns quantile based threshold for one day.
    Acceptable JSON format is explained in MJSONReader method of preprocessor module.

    Parameters
    ----------
    json_str: string
        JSON string or filepath

    plot_in_file: boolean
        plot raw data into *.png file or not

    out_path: string
        path where plot file should be created

    dmin: integer
        number of minutes to aggregate raw data

    last_ts: integer
        timestamp of calculation Unix-time

    Returns
    -------
    outjson_str: string
        dynamic threshold in JSON string format, i.e. [[timestamp, lower, upper]...]
    """
    dynout_pattern = dynthresh()
    outjson_str = ""
    autothresh = []
    monid, sensitivity, rarray, errm1, status = MJSONReader(json_str)
    if status == 0:
        for i in range(len(monid)):
            tseries, errm0, status0 = Preprocessor(rarray[i])
            if status0 == 0:
                autothresh = autothresh + [dynthresh(monid[i], tseries, rarray[i], plot_in_file, out_path, dmin, sensitivity[i], last_ts)]
                autothresh[i]["metric_id"] = monid[i]
            else:
                autothresh = autothresh + [dynout_pattern]
                autothresh[i]["metric_id"] = monid[i]
                autothresh[i]["errMsg"] = errm0
    else:
        autothresh = autothresh + [dynout_pattern]
        autothresh[0]["metric_id"] = ""
        autothresh[0]["errMsg"] = errm1
    outjson_str = dumps(autothresh)
    return outjson_str


def MHistAvg(json_str, plot_in_file=False, out_path=None, dmin=None, last_ts=-1):
    """
    Reading JSON file or string and returns quantile based threshold for one day.
    This synchronous one, i.e. calculation timestamps includes in JSON
    Acceptable JSON format is explained in MSJSONReader method of preprocessor module.

    Parameters
    ----------
    json_str: string
        JSON string or filepath

    plot_in_file: boolean
        plot raw data into *.png file or not

    out_path: string
        path where plot file should be created

    dmin: integer
        number of minutes to aggregate raw data

    last_ts: integer
        timestamp of calculation Unix-time

    Returns
    -------
    outjson_str: string
        dynamic threshold in JSON string format, i.e. [[timestamp, lower, upper]...]
    """
    dynout_pattern = dynthresh()
    outjson_str = ""
    autothresh = []
    monid, sensitivity, calc_ts, rarray, errm1, status = MSJSONReader(json_str)
    if status == 0:
        for i in range(len(monid)):
            tseries, errm0, status0 = Preprocessor(rarray[i])
            if status0 == 0:
                autothresh = autothresh + [dynthresh(monid[i], tseries, rarray[i], plot_in_file, out_path, dmin, sensitivity[i], calc_ts[i])]
                autothresh[i]["metric_id"] = monid[i]
            else:
                autothresh = autothresh + [dynout_pattern]
                autothresh[i]["metric_id"] = monid[i]
                autothresh[i]["errMsg"] = errm0
    else:
        autothresh = autothresh + [dynout_pattern]
        autothresh[0]["metric_id"] = ""
        autothresh[0]["errMsg"] = errm1
    outjson_str = dumps(autothresh)
    return outjson_str
