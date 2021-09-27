import warnings
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd


class BaseDataSource(ABC):
    """
    Abstract class for data sources.

    Parameters
    ----------
    index_sync : bool, optional
        If the index should be synced. If True, a valid tolerance must be given.
    tolerance : int, float or pandas.Timedelta
        Tolerance limit for syncing (see Notes). If ``index_sync`` is set to True,
        datapoints that are closer than the tolerance are merged so that they
        share a common index. The common index will be the first index of the
        neighboring datapoints.

    Notes
    -----
    The tolerance must be of a type that is comparable to the data index. E.g.
    if the data has a ``DatetimeIndex``, the tolerance should be of type
    ``pandas.Timestamp``. And if the data has a ``Int64Index``, the tolerance
    should be an integer.
    """

    def __init__(self, index_sync=False, tolerance=None):
        self._index_sync = index_sync
        self._tolerance = tolerance

    @abstractproperty
    def labels(self):
        """Data source labels."""
        raise NotImplementedError()

    @abstractmethod
    def _get(self, start, end):
        """
        Get data from source.

        Parameters
        ----------
        start :
            Start index of the data.
        end :
            End index of the data.

        Returns
        -------
        dict
            Label and data as key/value pairs. The data is returned as ``pandas.Series``
            objects.
        """
        raise NotImplementedError()

    def get(self, start, end):
        """
        Get data from source.

        Parameters
        ----------
        start :
            Start index of the data. Will be passed on to the ``_get`` method.
        end :
            End index of the data. Will be passed on to the ``_get`` method.

        Returns
        -------
        pandas.DataFrame
            Source data.
        """

        data = self._get(start, end)
        if not self._index_sync:
            return pd.DataFrame(data)
        else:
            if not self._tolerance:
                raise ValueError("No tolerance given.")
            return self._sync_data(data, self._tolerance)

    @staticmethod
    def _sync_data(data, tolerance):
        """
        Sync data index.

        Datapoints that are closer than the tolerance are merged so that they share
        a common index. The common index will be the first index of the neighboring
        datapoints.

        Parameters
        ----------
        data : dict
            Label and data as key/value pairs. The data must be represented as
            ``pandas.Series`` objects.
        tolerance : int, float or pandas.Timedelta
            Tolerance limit for syncing. The tolerance must be of a type that is
            comparable to the data index. E.g. if the data has a ``DatetimeIndex``,
            the tolerance should be of type ``pandas.Timestamp``. And if the data
            has a ``Int64Index``, the tolerance should be an integer.

        Returns
        -------
        pandas.DataFrame
            Synchronized data.
        """
        index_common = np.sort(
            np.unique(np.concatenate([series.index for series in data.values()]))
        )
        index_keep = np.r_[True, (np.diff(index_common) > tolerance)]
        index_common = index_common[index_keep]
        df_synced = pd.DataFrame(index=index_common)

        for key, series in data.items():
            if isinstance(series, pd.Series):
                series.name = key

            if tolerance >= np.median(np.diff(series.index)):
                warnings.warn(
                    f"Tolerance is greater than the median sampling frequency of '{key}'. "
                    "This may lead to significant loss of data."
                )

            df_synced = pd.merge_asof(
                df_synced,
                series,
                left_index=True,
                right_index=True,
                tolerance=tolerance,
                direction="nearest",
            )

        return df_synced

    def iter(self, start_list, end_list, indexing_mode="start"):
        """
        Data generator yielding tuples of index and data for every start/end pair.

        Parameters
        ----------
        start_list : array-like
            Start indexes. 'start' and 'end' must have the same length.
        end_list : array-like
            End indexes. 'start' and 'end' must have the same length.
        indexing_mode : str, optional
            Indexing mode. Must be 'start', 'end' or 'mid'.
        """
        start = np.asarray_chkfinite(start_list)
        end = np.asarray_chkfinite(end_list)

        if not len(start) == len(end):
            raise ValueError("'start' and 'end' does not have the same length.")

        if indexing_mode == "start":
            index = start
        elif indexing_mode == "end":
            index = end
        elif indexing_mode == "mid":
            index = start + (end - start) / 2.0
        else:
            raise ValueError("'indexing_mode' must be 'start', 'end' or 'mid'.")

        return (
            (index, self.get(start, end))
            for index, start, end in zip(index, start, end)
        )


class DrioDataSource(BaseDataSource):
    """
    DataReservoir.io data source.

    Parameters
    ----------
    drio_client : obj
        DataReservoir.io client.
    lables : dict
        Labels and timeseries IDs as key/value pairs.
    index_sync : bool, optional
        If the index should be synced. If True, a valid tolerance must be given.
    tolerance : int, float or pandas.Timedelta
        Tolerance limit for syncing (see Notes). If ``index_sync`` is set to True,
        datapoints that are closer than the tolerance are merged so that they
        share a common index. The common index will be the first index of the
        neighboring datapoints.
    convert_date : bool
        If True (default), the index is converted to DatetimeIndex.
        If False, index is returned as ascending integers. Will be passed on to
        the ``drio_client._get`` method.
    raise_empty : bool
        If True, raise ValueError if no data exist in the provided
        interval. Otherwise, return an empty pandas.Series (default). Will be passed
        on to the ``drio_client.get`` method.

    Notes
    -----
    The tolerance must be of a type that is comparable to the data index. E.g.
    if the data has a ``DatetimeIndex``, the tolerance should be of type
    ``pandas.Timestamp``. And if the data has a ``Int64Index``, the tolerance
    should be an integer.
    """

    def __init__(
        self,
        drio_client,
        labels,
        index_sync=False,
        tolerance=None,
        convert_date=True,
        raise_empty=False,
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._convert_date = convert_date
        self._raise_empty = raise_empty
        super().__init__(index_sync=index_sync, tolerance=tolerance)

    def _get(self, start, end):
        """
        Get data from the DataReservoir.io.

        Parameters
        ----------
        start :
            Start time of the data. Will be passed on to the ``drio_client.get`` method.
        end :
            End time of the data. Will be passed on to the ``drio_client.get`` method.

        Returns
        -------
        dict
            Label and data as key/value pairs. The data is returned as ``pandas.Series``
            objects.
        """
        data = {}
        for label in self.labels:
            data[label] = self._drio_client.get(
                self._labels[label],
                start=start,
                end=end,
                convert_date=self._convert_date,
                raise_empty=self._raise_empty,
            )
        return data

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels.keys())


def date_range_iter(
    source,
    start=None,
    end=None,
    periods=None,
    freq=None,
    closed=None,
    indexing_mode="start",
):
    """
    Return a fixed frequency data generator.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates. Will be passed on to ``pandas.date_range()``.
    end : str or datetime-like, optional
        Right bound for generating dates. Will be passed on to ``pandas.date_range()``.
    periods : int, optional
        Number of periods to generate. Will be passed on to ``pandas.date_range()``.
    freq : str or DateOffset, default 'D'
        Frequency. Will be passed on to ``pandas.date_range()``.
    closed : {None, 'left', 'right'}, optional
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None, the default).
    indexing_mode : str, optional
        Indexing mode. Must be 'start', 'end' or 'mid'.
    """
    start_end = pd.date_range(start=start, end=end, periods=periods, freq=freq, closed=closed)
    return source.iter(start_end[:-1], start_end[1:], indexing_mode=indexing_mode)


# class DataIterator:
#     """
#     Data iterator.

#     Parameters
#     ----------
#     source : obj
#         Data source object.
#     start : list
#         Start indexes. Will be passed on to source.get().
#     end : list
#         End indexes. Will be passed on to source.get().
#     indexing_mode : str
#         Indexing mode. Must be 'start', 'end' or 'mid'.
#     **kwargs : optional
#         Keyword arguments passed on to ``source.get()``.
#     """

#     def __init__(self, source, start, end, indexing_mode="start"):
#         if not isinstance(source, BaseDataSource):
#             raise TypeError("'source' must be instance of ``BaseDataSource.``")

#         if not len(start) == len(end):
#             raise ValueError("Start and end must have the same length.")

#         self._source = source
#         self._start = np.asarray_chkfinite(start)
#         self._end = np.asarray_chkfinite(end)

#         if indexing_mode == "start":
#             self._index = self._start
#         elif indexing_mode == "end":
#             self._index = self._end
#         elif indexing_mode == "mid":
#             self._index = start + (end - start) / 2.0

#         self._index_start_end_iter = zip(self._index, self._start, self._end)

#     def __next__(self):
#         start, end = next(self._start_end_iter)
#         if self._indexing_mode == "start":
#             index = start
#         elif self._indexing_mode == "end":
#             index = end
#         elif self._indexing_mode == "mid":
#             index = start + (end - start) / 2
#         else:
#             raise ValueError("indexing_mode must be 'start', 'end' or 'mid'.")
#         return index, self._source.get(start, end, **self._kwargs)

#     def __iter__(self):
#         return self


# class DateRangeIterMixin:
#     def date_range_iter(
#         self,
#         start=None,
#         end=None,
#         periods=None,
#         freq=None,
#         closed=None,
#         indexing_mode="start",
#         **kwargs,
#     ):
#         """
#         Return a fixed frequency DataIterator.

#         Parameters
#         ----------
#         start : str or datetime-like, optional
#             Left bound for generating dates. Will be passed on to ``pandas.date_range()``.
#         end : str or datetime-like, optional
#             Right bound for generating dates. Will be passed on to ``pandas.date_range()``.
#         periods : int, optional
#             Number of periods to generate. Will be passed on to ``pandas.date_range()``.
#         freq : str or DateOffset, default 'D'
#             Frequency. Will be passed on to ``pandas.date_range()``.
#         closed : {None, 'left', 'right'}, optional
#             Make the interval closed with respect to the given frequency to
#             the 'left', 'right', or both sides (None, the default).
#         **kwargs : optional
#             Keyword arguments passed on to ``source.get()``.
#         """
#         date_range = pd.date_range(
#             start=start, end=end, periods=periods, freq=freq, closed=closed
#         )
#         return DataIterator(
#             self, date_range[:-1], date_range[1:], indexing_mode=indexing_mode, **kwargs
#         )
