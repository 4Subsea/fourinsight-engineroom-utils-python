import pandas as pd


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    inclusive=None,
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
    inclusive : {"both", "neither", "left", "right"}, default "both"
        Include boundaries; whether to set each bound as closed or open.
    **kwargs :
        Additional keyword arguments that will be passed on to ``pandas.date_range()``.

    Returns
    -------
    start : list
        Sequence of start values as `pandas.Timestamp`.
    end : list
        Sequence of end values as `pandas.Timestamp`.
    """
    if periods:
        periods += 1

    start_end = pd.date_range(
        start=start, end=end, periods=periods, freq=freq, inclusive=inclusive, **kwargs
    )
    return list(start_end[:-1]), list(start_end[1:])
