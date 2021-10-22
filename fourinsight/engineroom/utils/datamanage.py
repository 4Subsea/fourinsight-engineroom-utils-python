import warnings
from abc import ABC, abstractmethod, abstractproperty
from itertools import chain

# 'pairwise' is introduced in Python 3.10.
try:
    from itertools import pairwise
except ImportError:
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)


import numpy as np
import pandas as pd


def universal_datetime_index(index):
    """
    Convert datetime-like index to universal type.

    Parameters
    ----------
    index : single value or array-like
        Datetime-like index. Will be passed on to :func:`pandas.to_datetime`.

    Returns
    int or array of int
        Index converted to epoch.
    """
    index = np.asarray_chkfinite(index).flatten()
    index = np.int64(pd.to_datetime(index, utc=True).values)
    if len(index) == 1:
        return index[0]
    else:
        return index


def universal_integer_index(index):
    """
    Convert integer-like index to universal type.

    Parameters
    ----------
    index : single value or array-like
        Integer-like index.

    Returns
    -------
    int or array of int
        Index converted to ``int`` type.
    """
    return np.int64(np.asarray_chkfinite(index))


class BaseDataSource(ABC):
    """
    Abstract class for data sources.

    Parameters
    ----------
    index_type : str or callable
        Index type (see Notes). Should be 'datetime', 'integer' or a callable.
    index_sync : bool, optional
        If the index should be synced. If True, a valid tolerance must be given.
    tolerance : int, float or pandas.Timedelta
        Tolerance limit for syncing (see Notes). If ``index_sync`` is set to True,
        datapoints that are closer than the tolerance are merged so that they
        share a common index. The common index will be the first index of the
        neighboring datapoints.

    Notes
    -----
    - The `tolerance` must be of a type that is comparable to the data index. E.g.
      if the data has a ``DatetimeIndex``, the tolerance should be of type
      ``pandas.Timestamp``. And if the data has a ``Int64Index``, the tolerance
      should be an integer.
    - The `index_type` describes how to interpret the data index and convert it
      to a universal type. By universal type, we mean a data type that is common
      for all indexes related to the data.

      If ``index_type='datetime'``, a datetime-like index is expected. Indices are
      passed on to :func:`universal_datetime_index` to create a universal index type.

      If ``index_type='integer'``, an integer-like index is expected. Indices are
      passed on to :func:`universal_integer_index` to create a universal index type.

      Custom index types are provided through giving a ``callable`` as the `index_type`.
      The callable should then take a single value or array-like, and convert it
      to a universal type.
    """

    def __init__(self, index_type, index_sync=False, tolerance=None):
        self._index_type = index_type
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
            Start index of the data. Will be passed on to the :meth:`_get` method.
        end :
            End index of the data. Will be passed on to the :meth:`_get` method.

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

        for key_i, series_i in data.items():
            if isinstance(series_i, pd.Series):
                series_i.name = key_i

            if tolerance >= np.median(np.diff(series_i.index)):
                warnings.warn(
                    f"Tolerance is greater than the median sampling frequency of '{key_i}'. "
                    "This may lead to significant loss of data."
                )

            df_synced = pd.merge_asof(
                df_synced,
                series_i,
                left_index=True,
                right_index=True,
                tolerance=tolerance,
                direction="nearest",
            )

        return df_synced

    def iter(self, start, end, index_mode="start"):
        """
        Iterate over source data as (index, data) pairs.

        Parameters
        ----------
        start : array-like
            Sequence of start indexes.
        end : array-like
            Sequence of end indexes.
        index_mode : str, optional
            How to index/label the data. Must be 'start', 'end' or 'mid'. If 'start',
            start is used as index. If 'end', end is used as index. If 'mid', the
            index is set to ``start + (end - start) / 2.0``. Then, the start and end
            objects must be of such type that this operation is possible.

        Yields
        ------
        index : label
            The index/label.
        data : pandas.DataFrame
            The source data.

        See Also
        --------
        fourinsight.engineroom.utils.iter_index :
            Convenience functions for generating 'start' and 'end' index lists.
        """
        start = np.asarray_chkfinite(start)
        end = np.asarray_chkfinite(end)

        if not len(start) == len(end):
            raise ValueError("'start' and 'end' must have the same length.")

        if index_mode == "start":
            index = start
        elif index_mode == "end":
            index = end
        elif index_mode == "mid":
            index = start + (end - start) / 2.0
        else:
            raise ValueError("'index_mode' must be 'start', 'end' or 'mid'.")

        return (
            (index_i, self.get(start_i, end_i))
            for index_i, start_i, end_i in zip(index, start, end)
        )

    def _index_universal(self, index):
        """
        Convert index to universal type.

        Parameters
        ----------
        index : single value or array-like
            Index value.
        """
        if callable(self._index_type):
            return self._index_type(index)
        elif self._index_type == "datetime":
            return universal_datetime_index(index)
        elif self._index_type == "integer":
            return universal_integer_index(index)


class DrioDataSource(BaseDataSource):
    """
    DataReservoir.io data source.

    Parameters
    ----------
    drio_client : obj
        DataReservoir.io client.
    lables : dict
        Labels and timeseries IDs as key/value pairs.
    index_type : str or callable
        Index type (see Notes). Should be 'datetime', 'integer' or a callable.
    index_sync : bool, optional
        If the index should be synced. If True, a valid tolerance must be given.
    tolerance : int, float or pandas.Timedelta
        Tolerance limit for syncing (see Notes). If ``index_sync`` is set to True,
        datapoints that are closer than the tolerance are merged so that they
        share a common index. The common index will be the first index of the
        neighboring datapoints.
    **get_kwargs : optional
        Keyword arguments that will be passed on to the ``drio_client.get`` method.

    Notes
    -----
    - The `tolerance` must be of a type that is comparable to the data index. E.g.
      if the data has a ``DatetimeIndex``, the tolerance should be of type
      ``pandas.Timestamp``. And if the data has a ``Int64Index``, the tolerance
      should be an integer.
    - The `index_type` describes how to interpret the data index and convert it
      to a universal type. By universal type, we mean a data type that is common
      for all indexes related to the data.

      If ``index_type='datetime'``, a datetime-like index is expected. Indices are
      passed on to ``pd.to_datetime`` and then converted to integer with ``np.int64``
      to create a universal index type.

      If ``index_type='integer'``, an integer-like index is expected. Indices are
      passed on to ``np.int64`` to create a universal index.

      Custom index types are provided through giving a ``callable`` as the `index_type`.
      The callable should then take a single value or list-like, and convert it
      to a universal type.
    """

    def __init__(
        self,
        drio_client,
        labels,
        index_type="datetime",
        index_sync=False,
        tolerance=None,
        **get_kwargs,
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._get_kwargs = get_kwargs
        super().__init__(index_type, index_sync=index_sync, tolerance=tolerance)

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
        return {
            label: self._drio_client.get(
                ts_id, start=start, end=end, **self._get_kwargs
            )
            for label, ts_id in self._labels.items()
        }

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels.keys())


class NullDataSource(BaseDataSource):
    """
    Will return empty data.

    Parameters
    ----------
    labels : list-like
        Data labels.
    index_type : str or callable
        Index type. Should be 'datetime', 'integer' or a callable.
    """

    def __init__(self, labels=None, index_type="datetime"):
        self._labels = tuple(labels) if labels else ()
        super().__init__(index_type, index_sync=False)

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels)

    def _get(self, start, end):
        return {label: pd.Series([], dtype="object") for label in self._labels}


class CompositeDataSource(BaseDataSource):
    """
    Handles data from a sequence of data sources.

    During download, the class will switch between different data sources based on
    the index.

    Parameters
    ----------
    index_source : list-like
        Sequence of (index, source) tuples. The `index` value determines which index
        a `source` is valid from. The source will then be valid until the next item
        in the sequence (see Example).

    Examples
    --------
    This example shows how to set up a composite of three data sources. During data
    download, data is retrieved from ``source_a`` between '2020-01-01 00:00' and
    '2020-01-02 00:00', from ``source_b`` between '2020-01-02 00:00' and '2020-01-03 00:00',
    and from ``source_c`` between '2020-01-03 00:00' and the 'end'.

    >>> index_source = [
            ('2020-01-01 00:00', source_a),
            ('2020-01-02 00:00', source_b),
            ('2020-01-03 00:00', source_c),
        ]
    >>> source = CompositeDataSource(index_source)
    >>> data = source.get('2020-01-01 00:00', '2020-01-05 00:00')
    """

    def __init__(self, index_source):
        index_source_flat = list(chain.from_iterable(index_source))
        self._sources = index_source_flat[1::2]

        index_type_set = set([source._index_type for source in self._sources if source])
        if len(index_type_set) != 1:
            raise ValueError("The data sources does not share the same 'index_type'.")
        index_type = index_type_set.pop()

        labels_set = set(
            [tuple(sorted(source.labels)) for source in self._sources if source]
        )
        if len(labels_set) != 1:
            raise ValueError("The data sources does not share the same 'labels'.")
        self._labels = labels_set.pop()

        super().__init__(index_type, index_sync=False)

        self._sources = [
            source if source else NullDataSource(self._labels, index_type)
            for source in self._sources
        ]

        self._index = {
            index: self._index_universal(index) for index in index_source_flat[::2]
        }
        if list(self._index.values()) != sorted(self._index.values()):
            raise ValueError(
                "indecies in 'index_source' must be in strictly increasing order."
            )

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels)

    def _get(self, *args, **kwargs):
        return NotImplemented

    def get(self, start, end):
        """
        Get data from source.

        Parameters
        ----------
        start :
            Start index of the data. Will be passed on to the :meth:`get` method
            of each individual data source.
        end :
            End index of the data. Will be passed on to the :meth:`get` method
            of each individual data source.

        Returns
        -------
        pandas.DataFrame
            Source data.
        """

        if (start is None) or (end is None):
            raise ValueError("'start' and 'end' can not be NoneType.")

        start_uni = self._index_universal(start)
        end_uni = self._index_universal(end)

        index_list = list(self._index.keys())
        index_uni_list = list(self._index.values())
        # shallow copy, source objects are not copied.
        sources_list = self._sources.copy()

        first_source = None
        while index_uni_list and index_uni_list[0] <= start_uni:
            index_uni_list.pop(0)
            index_list.pop(0)
            first_source = sources_list.pop(0)
        else:
            index_list.insert(0, start)
            sources_list.insert(0, first_source or NullDataSource(self._labels))

        while index_uni_list and index_uni_list[-1] >= end_uni:
            index_uni_list.pop()
            index_list.pop()
            sources_list.pop()
        else:
            index_list.append(end)

        data_list = [
            source_i.get(start_i, end_i)
            for (start_i, end_i), source_i in zip(pairwise(index_list), sources_list)
        ]
        return pd.concat(data_list)
