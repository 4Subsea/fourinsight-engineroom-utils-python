import pandas as pd


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    closed=None,
    **kwargs,
):
    """
    Return lists of start/end pairs. Wrapper around ``pandas.date_range``.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.
    end : str or datetime-like, optional
        Right bound for generating dates.
    periods : int, optional
        Number of periods to generate.
    freq : str or DateOffset, default 'D'
        Frequency.
    closed : {None, 'left', 'right'}, optional
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None, the default).
    **kwargs :
        Additional keyword arguments that will be passed on to ``pandas.date_range()``.

    Returns
    -------
    start : list
        Sequence of start values as `pandas.Timestamp`.
    end : list
        Sequence of end values as `pandas.Timestamp`.
    """
    start_end = pd.date_range(
        start=start, end=end, periods=periods, freq=freq, closed=closed, **kwargs
    )
    return list(start_end[:-1]), list(start_end[1:])
