from numpy import trunc, mod, append, ceil, argsort, repeat, where, take, isnan 
from numpy import nanmean, apply_along_axis, arange, zeros, empty, nan, nanstd 
from numpy import percentile, shape, array, tile
from numpy import any as nany
from helper import Median

def DivGroups(time_stamps, values, delta_min, ch_ind = None, overlap = True, looped = True):
    """
    Projects on one day and divides data points pairs into groups by having fallen in [t - 0.5 *delta_min, t + 0.5 *delta_min) criteria
    one day.
    
    Parameters
    ----------
    time_stamps: 1D numpy array
        timestamps of observed values

    values: 1D numpy array
        observed values of monitor
    
    delta_min: integer 
        number of minutes to aggregate raw data
    
    ch_ind: integer
        index of stationarity change poistion in data
    
    overlap: boolean
        indicates if groups can be overlapped or no, i.e. True/False
        i.e. if some elements of one group can exists in
        in neighbour groups

    looped: boolean
        (default True) indicates if last and first groups will be mirrored or not
    
    Returns
    -------
    one_day_tstp: 1D numpy array
        projection of all timestamps to one day, i.e. last day
    
    data_matrix: ndarray
        matrix of values of raw data projected onto one day
    
    day_pattern:
        number of day from 01 January 1970 given for each row of data_matrix
    
    nodes: ndarray
        vector of timestamps of nodes around which threshold will be calculated
    
    error_message: string
        possible error that occurs during transformation in string format
    """
    # initializing default values of returning parameters
    if ch_ind != None and len(time_stamps) > 1:
        time_stamps = time_stamps[ch_ind:]
        values = values[ch_ind:]
    # predefined step given in minutes, transforming it ot days
    delta_seconds = delta_min * 60
    # number of seconds in a day
    day_in_seconds = 24 * 60 * 60
    # most recent day timestamp
    last_day_tstp = day_in_seconds * (max(time_stamps) // day_in_seconds)
    # projecting all timestamps to one day, i.e. 01 January 1970
    one_day_tstp = time_stamps % day_in_seconds
    # day information to store separately
    day_info = time_stamps // day_in_seconds
    # getting day_info to 0, max(number of days) range and changing type
    min_day = min(day_info)
    day_info = day_info - min_day
    day_info = day_info.astype("int")
    # maximal day number
    day_max = max(day_info)
    # getting group number for all data points by their appearance in day
    if overlap:
        group_ind = one_day_tstp // delta_seconds
        group_ind = 2 * group_ind + 1
        group_ind_even = trunc((one_day_tstp + (0.5 * delta_seconds)) / delta_seconds)
        group_ind_even = 2 * group_ind_even
        group_ind = append(group_ind, group_ind_even)       
        # timestamps of nodes around which threshold will be calculated        
        nodes = arange(0, day_in_seconds, 0.5 * delta_seconds)
        nodes = nodes + last_day_tstp
        nodes = append(nodes, [last_day_tstp + 24 * 60 * 60])
        # number of groups
        group_num = int(ceil(2 * day_in_seconds / delta_seconds) + 1)
        values = tile(values, 2)
        day_info = tile(day_info, 2)
    else:
        group_ind = trunc((one_day_tstp + (0.5 * delta_seconds)) / delta_seconds)
        # timestamps of nodes around which threshold will be calculated
        nodes = arange(0, day_in_seconds, delta_seconds)
        nodes = nodes + last_day_tstp
        nodes = append(nodes, [last_day_tstp + 24 * 60 * 60])
        # number of groups
        group_num = int(ceil(day_in_seconds / delta_seconds) + 1)
    # list of values
    data_groups = [None] * group_num
    # day information of groups
    dayinf_groups = [None] * group_num
    # number of rows and columns in data matrix
    mat_nrow = 0
    mat_ncol = group_num
    freq_matrix = zeros([group_num, (day_max + 1)], dtype = "int")
    # splitting data into a list of numpy arrays by their appearance date in a day
    for i in range(group_num):
        # getting indices of preferred time in day
        group_indeces = where(group_ind == i)[0]
        # if it is not empty then filling it into a list of data and day information
        if len(group_indeces) > 0:
            data_groups[i] = take(values, group_indeces)
            dayinf_groups[i] = take(day_info, group_indeces)
            dayinf_groups[i] = dayinf_groups[i].astype("int")
            if (i != 0) and (i != (group_num - 1)):
                freq_matrix[i,:] = map(lambda(w): len(where(dayinf_groups[i] == w)[0]), arange(day_max + 1))
    if overlap: 
        values = values[:(len(values) / 2)]
        day_info = day_info[:(len(day_info) / 2)]
    if looped:
        if (data_groups[0] != None) and (data_groups[-1] != None):
            data_groups[0] = append(data_groups[0], data_groups[-1])
            data_groups[-1] = data_groups[0]
            dayinf_groups[0] = append(dayinf_groups[0], dayinf_groups[-1])
            dayinf_groups[-1] = dayinf_groups[0]
            first_sec_ind = argsort(dayinf_groups[0])
            dayinf_groups[0] = dayinf_groups[0][first_sec_ind]
            dayinf_groups[-1] = dayinf_groups[0]
            data_groups[0] = data_groups[0][first_sec_ind]
            data_groups[-1] = data_groups[0]
        else:
            if (data_groups[0] == None) and (data_groups[-1] != None):
                data_groups[0] = data_groups[-1]
                dayinf_groups[0] = dayinf_groups[-1]
            if (data_groups[0] != None) and (data_groups[-1] == None):
                data_groups[-1] = data_groups[0]
                dayinf_groups[-1] = dayinf_groups[0]
    freq_matrix[0, :] = map(lambda(w): len(where(dayinf_groups[0] == w)[0]), arange(day_max + 1))
    # calculating number of rows in matrix
    if looped:
        freq_matrix[-1, :] = freq_matrix[0,:]
    else:
        freq_matrix[-1, :] = map(lambda(w): len(where(dayinf_groups[-1] == w)[0]), arange(day_max + 1))
    freq_pattern = apply_along_axis(max, 0, freq_matrix)
    day_pattern = apply_along_axis(lambda(w): repeat(w, freq_pattern), 0, arange(day_max + 1))
    mat_nrow = sum(freq_pattern)
    # creating data matrix with predefined number of columns and rows
    data_matrix = empty([mat_nrow, mat_ncol], dtype = float)
    # filling it in with nan-s
    data_matrix[:] = nan
    # filling raw data matrix from list
    miss_nan = repeat(nan, mat_nrow)
    for i in range(mat_ncol):
        temp_data = data_groups[i]
        if (temp_data != None):
            if len(temp_data) < mat_nrow: 
                temp_day_inf =  dayinf_groups[i]
                miss_pattern = freq_pattern - freq_matrix[i, :]
                miss_pattern = apply_along_axis(lambda(w): repeat(w, miss_pattern), 0, arange(day_max + 1))
                miss_pattern = miss_pattern + 0.1
                temp_data = append(temp_data, miss_nan[ :len(miss_pattern)])
                temp_day_inf = append(temp_day_inf, miss_pattern)
                temp_ind = argsort(temp_day_inf)
                temp_data = temp_data[temp_ind]
            data_matrix[:, i] = temp_data
    ## recovering original range of timestamps
    day_pattern = day_pattern + min_day
    if len(one_day_tstp) > 0: 
        one_day_tstp = one_day_tstp + last_day_tstp
    return one_day_tstp, data_matrix, day_pattern, nodes

def HistQuantile(x, hprobs, limits, nout, def_val):
    """
    Estimates upper and lower bounds for array x using percentiles 
    percents given in hprobs.
    
    Parameters
    ----------
    x: 1D array 
        one dimensional array
    hprobs: 2 element 1D array
        upper and lower percentiles percents values for threshold
    limits: 2 element 1D array
        upper and lower bounds above which should be certain number of elements
        that upper and lower threshold goes that way
    nout: integer 
        number of elements allowed to go out from bounds limits
    def_val:
        default value of thresholds in case when calculation of 
        quantiles are impossible 
        
    Returns
    -------
    qdown, qup: float 
        lower and upper bounds estimated using percentile
    """
    qdown = def_val## initialization
    qup = def_val## initial values
    x = x[~isnan(x)]
    if (nanstd(x) > 0) and (len(x) > 1):## when x is constant it is useless to use percentile to get thresholds
        hquant = percentile(x,hprobs)## calculating percentiles
        qupper = max(hquant)##robust usage of quantile in case when we have
        qlower = min(hquant)##x with median is equal to zero then this approch works more stable
        quplow =  [max(x[where(x <= qupper)[0]]), min(x[where(x >= qlower)[0]])]
        qup = max(quplow)
        qdown = min(quplow)
        if (qup > limits[1]) and (len(where( (x > limits[1]) & (x < qup) )[0]) < nout):
            qup = limits[1]## in case when number of outliers is more than nout then
        if (qdown < limits[0]) and (len(where( (x < limits[0]) & (x > qdown) )[0]) < nout):
            qdown = limits[0]## qup become limit[1] and qdown limit[0]
    else:## in case when x is constant then we return that constant value as upper and lower
        if len(x) > 1:
            qup = nanmean(x)## thresholds
            qdown =  nanmean(x)
    return qdown,qup

def HistOutliers(data_mat, limits):
    """
    Returns historical outliers of data_mat from given limits 
    
    Parameters
    ----------
    data_mat: 2D array 
        two dimensional array of data
    limits: 2D array
        upper and lower limits estimated from HistQuantile mwthod
        
    Returns
    -------
    outliers: 1D array
        outliers of data_mat, given in relative distance from upper and lower bounds
    """
    n_groups = shape(data_mat)[1]
    upper = array([])
    lower = array([])
    for i in arange(n_groups):
        temp_up = where(data_mat[:, i] > limits[1, i])[0]
        temp_down = where(data_mat[:, i] < limits[0, i])[0]
        if len(temp_up) > 0:
            temp_data = data_mat[temp_up, i] - limits[1, i]
            upper = append(upper, temp_data)
        if len(temp_down) > 0:
            temp_data = data_mat[temp_down, i] - limits[0, i]
            lower = append(lower, temp_data)
    outliers = append(upper, lower)
    return outliers

def CalcOnGroups(div_groups, func_calc, func_def = Median):
    """
    Gives results of calculation of func_calc on divided groups of data
    
    Parameters
    ----------
    div_groups: 2D array 
        two dimensional array of data
    func_calc: function name
        name of function to be used to calculate on div_groups rows
    func_def: function name
        name of function to be used to calculate on div_groups rows in case when func_calc is failed
        
    Returns
    -------
    der_vals: 1D array
        derived values from div_groups using func_calc
    """
    default_value = func_def(div_groups)
    der_vals = apply_along_axis(func_calc, 0, div_groups)
    if (nany(isnan(der_vals))):
        der_vals[where(isnan(der_vals))[0]] = default_value
    return der_vals