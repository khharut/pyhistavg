from numpy import array, tile, ceil, zeros, column_stack, arange
from pandas import DataFrame
from pystl import stl


def nextodd(x):
    """
    Gives next odd integer of given integer

    Parameters
    ----------
    x: integer
        odd or even integer value

    Returns
    -------
    y: integer
        next odd integer of x or x if it odd
    """
    y = round(x)  # making x integer by rounding if it not integer
    if y % 2 == 0:
        y = y + 1  # if y is even then adding unity to make it odd
    y = int(y)  # casting y to int
    return y


def deg_check(deg):
    """
    Checks if deg can be converted into integer with value 0 or 1

    Parameters
    ----------
    deg: any object convertable to integer
        any object to be checked if possible to get integer value 0 or 1

    Returns
    -------
    deg0: integer
        if possible integer value from deg otherwise ValueError raised
    """
    deg0 = -1
    try:
        deg0 = str(deg)
    except Exception:
        pass
    try:
        deg0 = int(deg0)
    except Exception:
        pass
    if deg0 < 0 or deg0 > 1:
        raise ValueError
    return deg0


def stl_decompose(x, period=None, s_window=None, s_degree=0, t_window=None,
                  t_degree=1, l_window=None, l_degree=None, s_jump=None,
                  t_jump=None, l_jump=None, robust=False,
                  inner=None, outer=None):
    """
    Decompose a time series into seasonal, trend and irregular
    components using loess, acronym STL. Popular decomposition
    algorithm is written by

    R.B. Cleveland, W.S.Cleveland, J.E. McRae, and I. Terpenning,
    STL: A Seasonal-Trend Decomposition Procedure Based on Loess,
    Statistics Research Report, AT&T Bell Laboratories.

    Parameters
    ----------
    x: array
        one dimensional array of time series

    period: integer
        decomposition window length

    s_window: integer
        overall decompostion window length

    s_degree,t_window,t_degree,l_window,l_degree,
    s_jump,t_jump,l_jump,robust,inner,outer: integer
        different integer parameters from original fortran code


    Returns
    -------
    z: pandas DataFrame
        returns DataFrame containing decomposed
        time series
        first column is trend
        second coulmn is seasonal
        third column is remainder
    """
    # converting original x array to numpy array
    y = array(x)
    # length of array
    n = len(y)
    # setting default value of s_window and s_degree according to their values
    if s_window is None:
        s_window = 10 * n + 1
        s_degree = 0
    # setting default value of period
    if period is None:
        period = round(float(n) / 3.0)
    # setting default value of l_window
    if l_window is None:
        l_window = nextodd(period)
    # setting default value of l_degree
    if l_degree is None:
        l_degree = t_degree
    # checking if s_degree, t_degree and l_degree are integer in interval (0,1)
    s_degree = deg_check(s_degree)
    t_degree = deg_check(t_degree)
    l_degree = deg_check(l_degree)
    # setting default value of t_window
    if t_window is None:
        t_window = nextodd(ceil(1.5 * float(period) / (1.0 - 1.5 / float(s_window))))
    # setting default value of s_jump,t_jump,l_jump,inner,outer
    if s_jump is None:
        s_jump = ceil(float(s_window) / 10.0)
    if t_jump is None:
        t_jump = ceil(float(t_window) / 10.0)
    if l_jump is None:
        l_jump = ceil(float(l_window) / 10.0)
    if inner is None:
        inner = 1 if robust else 2
    if outer is None:
        outer = 15 if robust else 0
    # creating seasonal vector to get seasonal data from fortran function
    seasonal = zeros(n, dtype=float)
    # creating trend vector to get trend data from fortran function
    trend = zeros(n, dtype=float)
    # creating weights vector to get weights data from fortran function
    weights = zeros(n, dtype=float)
    # calling original stl() fortran function
    z = stl(y=y, n=int(n), np=int(period), ns=int(s_window), nt=int(t_window),
            nl=int(l_window), isdeg=int(s_degree), itdeg=int(t_degree),
            ildeg=int(l_degree), nsjump=int(s_jump), ntjump=int(t_jump),
            nljump=int(l_jump), ni=int(inner), no=int(outer), rw=weights,
            season=seasonal, trend=trend, work=zeros(((n + 2 * period), 5),
            dtype=float))
    # creating cycle 1,2,3...period,1,2,3..period for averaging seasonal data
    which_cycle = tile(arange(period), ceil(float(n) / float(period)))[:n]
    # averaging seasonal data according to cycle
    true_season = DataFrame(column_stack((seasonal, which_cycle)),
                            columns=['data', 'seasonal']).groupby('seasonal').mean()
    true_season = array(true_season['data'])
    # replicating true season to get length of n
    seasonal = tile(true_season, ceil(float(n) / float(period)))[:n]
    # correcting remainder according to true season and trend
    remainder = y - trend - seasonal
    # creating return pandas dataframe
    z = DataFrame(column_stack((trend, seasonal, remainder)), columns=['trend', 'seasonal', 'remainder'])
    return z
