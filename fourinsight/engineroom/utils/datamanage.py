import warnings
from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from hashlib import md5

import numpy as np
import pandas as pd


# def universal_datetime_index(index):
#     """
#     Convert datetime-like index to universal type.

#     Parameters
#     ----------
#     index : single value or array-like
#         Datetime-like index. Will be passed on to :func:`pandas.to_datetime`.

#     Returns
#     int or array of int
#         Index converted to epoch.
#     """
#     index = np.asarray_chkfinite(index).flatten()
#     index = np.int64(pd.to_datetime(index, utc=True).values)
#     if len(index) == 1:
#         return index[0]
#     else:
#         return index


# def universal_integer_index(index):
#     """
#     Convert integer-like index to universal type.

#     Parameters
#     ----------
#     index : single value or array-like
#         Integer-like index.

#     Returns
#     -------
#     int or array of int
#         Index converted to ``int`` type.
#     """
#     return np.int64(np.asarray_chkfinite(index))


class BaseIndexConverter:
    @abstractmethod
    def universal_index(self, index):
        raise NotImplementedError()

    @abstractmethod
    def universal_partition(self, partition):
        raise NotImplementedError()

    @abstractproperty
    def reference(self):
        raise NotImplementedError()

    def universal_chunk_iter(self, start, end, partition):
        start = self.universal_index(start)
        end = self.universal_index(end)
        partition = self.universal_partition(partition)
        ref = self.universal_index(self.reference)

        start_part = ref + ((start - ref) // partition) * partition
        end_part = ref + ((end - ref) // partition) * partition
        if end_part <= end:
            end_part += partition

        index_chunks = np.arange(start_part, end_part, partition)
        return zip(index_chunks[:-1], index_chunks[1:])


class DatetimeIndexConverter(BaseIndexConverter):
    def universal_index(self, index):
        index = np.asarray_chkfinite(index).flatten()
        index = pd.to_datetime(index, utc=True).values
        if len(index) == 1:
            return index[0]
        else:
            return index

    def universal_partition(self, partition):
        return pd.to_timedelta(partition)

    @property
    def reference(self):
        return pd.to_datetime(0, utc=True)


class IntegerIndexConverter(BaseIndexConverter):
    def universal_index(self, index):
        return np.int64(np.asarray_chkfinite(index))

    @abstractmethod
    def universal_partition(self, partition):
        return int(partition)

    @abstractproperty
    def reference(self):
        return 0


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

    def __init__(self, index_type, index_sync=False, tolerance=None, cache=None):
        self._index_type = index_type
        self._index_sync = index_sync
        self._tolerance = tolerance

        if isinstance(self._index_type, BaseIndexConverter):
            self._index_converter = self._index_type
        elif self._index_type == "datetime":
            self._index_converter = DatetimeIndexConverter()
        elif self._index_type == "integer":
            self._index_converter = IntegerIndexConverter()
        else:
            raise ValueError(
                "The 'index_type' is not valid. "
                "Should be 'datetime', 'integer' or an instance of 'BaseIndexConverter'"
            )

        self._fingerprint = self._md5()
        self._cache = Path(cache) if cache else None
        self._cache_index = self._cache / f"{self._fingerprint}.index" if self._cache else None

        if self._cache:
            if not self._cache.exists():
                self._cache.mkdir()

            if not self._cache_index.exists():
                open(self._cache_index, mode="w").close()

    def _md5(self):
        return md5("TODO: Not implemented yet.".encode()).hexdigest()

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

    def get_cached(self, start, end):
        pass

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

    # def _index_universal(self, index):
    #     """
    #     Convert index to universal type.

    #     Parameters
    #     ----------
    #     index : single value or array-like
    #         Index value.
    #     """
    #     if callable(self._index_type):
    #         return self._index_type(index)
    #     elif self._index_type == "datetime":
    #         return universal_datetime_index(index)
    #     elif self._index_type == "integer":
    #         return universal_integer_index(index)


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
        cache=None,
        **get_kwargs,
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._get_kwargs = get_kwargs
        super().__init__(index_type, index_sync=index_sync, tolerance=tolerance, cache=cache)

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

    def __init__(self, index_source, index_sync=False, tolerance=None):
        self._index_attached, self._sources = np.asarray(index_source).T

        index_type_set = set([source._index_type for source in self._sources if source])
        if len(index_type_set) == 1:
            index_type = list(index_type_set)[0]
        else:
            raise ValueError("The data sources does not share the same 'index_type'.")

        labels_set = set(
            [tuple(sorted(source.labels)) for source in self._sources if source]
        )
        if len(labels_set) == 1:
            self._labels = list(labels_set)[0]
        else:
            raise ValueError("The data sources does not share the same 'labels'.")

        self._sources = np.array(
            [
                source if source else NullDataSource(self._labels, index_type)
                for source in self._sources
            ]
        )

        super().__init__(index_type, index_sync=index_sync, tolerance=tolerance)

        sorted_args = np.argsort(self._index_converter.universal_index(self._index_attached))
        self._sources = self._sources[sorted_args]
        self._index_attached = self._index_attached[sorted_args]

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels)

    def _get(self, start, end):

        if (start is None) or (end is None):
            raise ValueError("'start' and 'end' can not be NoneType.")

        attached_after_start = self._index_converter.universal_index(
            self._index_attached
        ) > self._index_converter.universal_index(start)
        attached_before_end = self._index_converter.universal_index(
            self._index_attached
        ) < self._index_converter.universal_index(end)
        attached_between_start_end = attached_after_start & attached_before_end

        if sum(~attached_after_start) == 0:
            first_source = NullDataSource(self._labels)
        else:
            first_source = self._sources[~attached_after_start][-1]

        start_list = np.r_[[start], self._index_attached[attached_between_start_end]]
        end_list = np.r_[self._index_attached[attached_between_start_end], [end]]
        source_list = np.r_[first_source, self._sources[attached_between_start_end]]

        data_list = [
            source_i._get(start_i, end_i)
            for start_i, end_i, source_i in zip(start_list, end_list, source_list)
        ]

        return self._concat_data(data_list)

    def _concat_data(self, data_list):
        """Concatenate list of data dicts."""
        return {
            key: pd.concat([data_i[key] for data_i in data_list])
            for key in self._labels
        }
