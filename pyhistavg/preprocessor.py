from pandas.json import loads
from os.path import isdir, isfile
from numpy import diff, array, arange, argsort, std, nan
from numpy import column_stack, diff, median, ceil, isnan
from scipy.interpolate import interp1d
from scipy.stats import mode


def JSONValidator(jstring):
    """
    Checks given jstring compatibilty to JSON format

    Parameters
    ----------
    jstring: string
        possible JSON string to be checked

    Returns
    -------
    json_data: JSON data as dict
        data extracted from JSON
        None if it is not detectable
    """
    # trying to read possible json string, if it impossible then return False
    try:
        json_data = loads(jstring)
    except Exception:
        json_data = None
    return json_data


def FileString(file_path):
    """
    Reading file content and returns if file exists,
    otherwise returns original string or empty string

    Parameters
    ----------
    file_path: string
        possible file_path or string to be read

    Returns
    -------
    out_string: string
        content of the file or just a original string
    """
    # default value setting
    out_string = ""
    # checking if given pah is real file path
    ftype = isfile(file_path)
    # checking if given pah is real directory path
    dtype = isdir(file_path)
    # if it is file ppath and not directory path then read it
    if ftype and not(dtype):
        out_string = open(file_path, "r").read()
    if not(ftype) and not(dtype):
        out_string = file_path
    return out_string


def MJSONReader(json_str):
    """
    Reading JSON file or string and returns monitor IDs, timestamps and values of data

    Parameters
    ----------
    json_str: string/file_path
        file_path or string to be read in JSON format
        format example: [{"metric_id":"454","sens":"low","data":{"timestamp1":value1,...}},{"metric_id":"789","sens":"high","data":{"timestamp1":value1,...}}]

    Returns
    -------
    monitor_id: list of strings
        monitor ID of JSON data converted to string format if exists in data otherwise should be None

    lsens: list of strings
        containing information about sensitivity for every metric

    raw_data: list of 2D arrays
        list of 2D arrays containing:
            first column containing timestamps
             second column containing values

    error_message: string
        possible error that occurs during transformation in string format

    status: integer
        1 shows that something goes wrong
        0 shows that something eveerything is OK
    """
    raw_data = []
    monitor_id = []
    lsens = []
    error_message = ""
    status = 0
    # checking if json_str is valid JSON
    string_data = FileString(json_str)
    input_data = JSONValidator(string_data)
    if input_data is None:
        input_data = {'0': 0}
        error_message = error_message + "Unknown input data type."
        status = 1
    try:
        if type(input_data) is list:
            input_keys = input_data[0].keys()
        if type(input_data) is dict:
            input_keys = input_data.keys()
            input_data = [] + [input_data]
        pred_keys = ['data', 'metric_id', 'sens']
        if sorted(input_keys) == pred_keys:
            for i in range(len(input_data)):
                metric_values = []
                timestamps = []
                monitor_id = monitor_id + [input_data[i][pred_keys[1]]]
                lsens = lsens + [input_data[i][pred_keys[2]]]
                data_part = input_data[i][pred_keys[0]]
                if type(data_part) is dict:
                    metric_values = data_part.values()
                    timestamps = array(map(long, map(float, data_part.keys()))) / 1000
                    time_order = argsort(timestamps)
                    # sorting metric values according to time
                    metric_values = array([metric_values[i] for i in time_order])
                    timestamps = array([timestamps[i] for i in time_order])
                    raw_data = raw_data + [column_stack((timestamps, metric_values))]
                else:
                    error_message = error_message + "Unsupported JSON data type."
                    status = 1
        else:
            error_message = error_message + "Unsupported JSON data type."
            status = 1
    except Exception:
        error_message = error_message + "Unsupported JSON data type."
        status = 1
    return monitor_id, lsens, raw_data, error_message, status


def MSJSONReader(json_str):
    """
    Reading JSON file or string and returns monitor IDs, calculation timestamp, timestamps and values of data (i.e. synchronous one)

    Parameters
    ----------
    json_str: string/file_path
        file_path or string to be read in JSON format
        format example: [{"metric_id":"454","sens":"low","data":{"timestamp1":value1,...}},{"metric_id":"789","sens":"high","data":{"timestamp1":value1,...}}]

    Returns
    -------
    monitor_id: list of strings
        monitor ID of JSON data converted to string format if exists in data otherwise should be None

    lsens: list of strings
        containing information about sensitivity for every metric

    call_ts: integer
        calculation timestamps

    raw_data: list of 2D arrays
        list of 2D arrays containing:
            first column containing timestamps
            second column containing values

    error_message: string
        possible error that occurs during transformation in string format

    status: integer
        1 shows that something goes wrong
        0 shows that something eveerything is OK
    """
    raw_data = []
    monitor_id = []
    lsens = []
    call_ts = []
    error_message = ""
    status = 0
    # checking if json_str is valid JSON
    string_data = FileString(json_str)
    input_data = JSONValidator(string_data)
    if input_data is None:
        input_data = {'0': 0}
        error_message = error_message + "Unknown input data type."
        status = 1
    try:
        if type(input_data) is list:
            input_keys = input_data[0].keys()
        if type(input_data) is dict:
            input_keys = input_data.keys()
            input_data = [] + [input_data]
        pred_keys = ['data', 'metric_id', 'sens', 'timestamp']
        if sorted(input_keys) == pred_keys:
            for i in range(len(input_data)):
                metric_values = []
                timestamps = []
                call_ts = call_ts + [input_data[i][pred_keys[3]]]
                monitor_id = monitor_id + [input_data[i][pred_keys[1]]]
                lsens = lsens + [input_data[i][pred_keys[2]]]
                data_part = input_data[i][pred_keys[0]]
                if type(data_part) is dict:
                    metric_values = data_part.values()
                    timestamps = array(map(long, map(float, data_part.keys()))) / 1000
                    time_order = argsort(timestamps)
                    # sorting metric values according to time
                    metric_values = array([metric_values[i] for i in time_order])
                    timestamps = array([timestamps[i] for i in time_order])
                    raw_data = raw_data + [column_stack((timestamps, metric_values))]
                else:
                    error_message = error_message + "Unsupported JSON data type."
                    status = 1
        else:
            error_message = error_message + "Unsupported JSON data type."
            status = 1
    except Exception:
        error_message = error_message + "Unsupported JSON data type."
        status = 1
    return monitor_id, lsens, call_ts, raw_data, error_message, status


def SJSONReader(json_str):
    """
    Reading JSON file or string and returns monitor ID, timestamps and values of data
    JSON format should be one of these listed below
    a) {"timestamp1":value1, "timestamp2":value2....}
    b) {"monitor_id":{"timestamp1":value1, "timestamp2":value2....}}
    c) [[timestamp1,value1],[timestamp2,value2],....]

    Parameters
    ----------
    json_str: string/file_path
        file_path or string to be read in JSON format

    Returns
    -------
    monitor_id: string
        monitor ID of JSON data converted to string format if exists in data otherwise should be None

    raw_data: 2D array
        first column containing timestamps
        second column containing values

    error_message: string
        possible error that occurs during transformation in string format

    status: integer
        1 shows that something goes wrong
        0 shows that something eveerything is OK
    """
    raw_data = None
    error_message = ""
    status = 0
    # checking if json_str is valid JSON
    string_data = FileString(json_str)
    input_data = JSONValidator(string_data)
    if input_data is None:
        input_data = {'0': 0}
        error_message = error_message + "Unknown input data type."
        status = 1
    # setting monitor_id default value
    monitor_id = None
    try:
        if type(input_data) is dict:
            # reading timestamps from JSON array
            timestamps = array(map(long, map(float, input_data.keys()))) / 1000
            # reading metric values from JSON array
            metric_values = input_data.values()
        else:
            if type(input_data) is list and len(input_data) > 0:
                input_data = sum(input_data, [])
                input_data = sum(input_data, [])
                if len(input_data) > 1:
                    timestamps = array(input_data[0::2], dtype=long) / 1000
                    metric_values = input_data[1::2]
                else:
                    timestamps = [0]
                    metric_values = [0]
                    error_message = error_message + "Unsupported JSON data type."
                    status = 1
            else:
                timestamps = [0]
                metric_values = [0]
                error_message = error_message + "Unsupported JSON data type."
                status = 1
    except Exception:
        timestamps = [0]
        metric_values = [0]
        error_message = error_message + "Unsupported JSON data type."
        status = 1
    # checking if monitor ID exists in JSON and if yes reading it
    if type(metric_values[0]) is dict and len(timestamps) == 1:
        monitor_id = str(input_data.keys()[0])
        timestamps = array(map(long, map(float, (input_data.values()[0]).keys()))) / 1000
        metric_values = (input_data.values()[0]).values()
    # finding time order of timestamps
    time_order = argsort(timestamps)
    # sorting metric values according to time
    metric_values = array([metric_values[i] for i in time_order])
    timestamps = array([timestamps[i] for i in time_order])
    raw_data = column_stack((timestamps, metric_values))
    return monitor_id, raw_data, error_message, status


def Preprocessor(raw_array):
    """
    Reading JSON file or string and returns time series

    Parameters
    ----------
    json_str: string/file_path
        file_path or string to be read in JSON format

    Returns
    -------

    time_series: ndarray
        two column matrix of timestamps and values of time series

    error_message: string
        possible error that occurs during transformation in string format

    status: integer
        1 shows that something goes wrong
        0 shows that something eveerything is OK
    """
    timestamps = raw_array[:, 0]
    metric_values = raw_array[:, 1]
    time_series = None
    error_message = ""
    # checking if json_str is valid JSON
    status = 0
    if len(timestamps) > 1:
        std_data = std(metric_values, ddof=1)
        if isnan(std_data):
            std_data = 0
        diff_timestamps = diff(timestamps)
        dtime = float(mode(diff_timestamps)[0][0])
        dtime0 = float(median(diff_timestamps))
        alpha = dtime / dtime0
        obs = ceil(24.0 * 60.0 * 60.0 / dtime)
        nobs = int(ceil(float(timestamps[-1] - timestamps[0]) / float(dtime)))
        intervals = ceil(float(nobs) / float(obs))
        data_quality = float(len(timestamps)) / float(nobs)
    else:
        std_data = -1
        dtime = 0
        dtime0 = 0
        alpha = 0
        obs = 0
        nobs = 1
        intervals = 1
        data_quality = 0
    if (data_quality <= 0.25) or (len(timestamps) < 3) or (intervals <= 1) or (dtime >= (6 * 60 * 60)) or (alpha < 0.5) or (std_data <= 0):
        if (alpha >= 0.5) and (status == 0) and (len(timestamps) < 3):
            error_message = error_message + "Not enough data. At least 3 data point should be supplied."
            status = 1
        if (alpha >= 0.5) and (status == 0) and (intervals <= 1):
            error_message = error_message + "Not enough data. At least 2 days of data should be supplied."
            status = 1
        if (alpha >= 0.5) and (status == 0) and (data_quality <= 0.25):
            error_message = error_message + "Not enough data. At least 25% of data points should be supplied."
            status = 1
        if (alpha >= 0.5) and (status == 0) and (dtime >= (6 * 60 * 60)):
            error_message = error_message + "Time steps between timestamps should be less than 6 hours."
            status = 1
        if (alpha < 0.5) and (status == 0):
            error_message = error_message + "Time step between timestamps are not fixed."
            status = 1
        if (std_data <= 0):
            error_message = error_message + "Impossible to calculate thresholds for constant data."
            status = 1
    if (data_quality > 0.25) and (len(timestamps) > 2) and (intervals > 1) and (dtime < (6 * 60 * 60)) and (alpha >= 0.5) and (std_data > 0):
        timestamp_fixed_step = arange(timestamps[0], timestamps[-1], dtime)
        metric_values_on_nodes = interp1d(x=timestamps, y=metric_values, kind='zero', bounds_error=False, fill_value=nan)
        if timestamp_fixed_step[-1] == timestamps[-1]:
            metric_values_on_nodes[-1] = metric_values[-1]
        normalized_metric_values = metric_values_on_nodes(timestamp_fixed_step)
        time_series = column_stack((timestamp_fixed_step, normalized_metric_values))
    return time_series, error_message, status
