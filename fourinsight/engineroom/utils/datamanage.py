import warnings
from abc import ABC, abstractmethod, abstractproperty
from hashlib import md5
from itertools import chain
from pathlib import Path

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


class BaseIndexConverter(ABC):
    @abstractmethod
    def to_universal_index(self, index):
        """Convert index to universal type"""
        raise NotImplementedError()

    @abstractmethod
    def to_universal_delta(self, delta):
        """Convert index partition to universal type"""
        raise NotImplementedError()

    def to_native_index(self, index):
        """Convert index to native type"""
        return self.to_universal_index(index)

    @abstractproperty
    def reference(self):
        """Index reference"""
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @classmethod
    def __eq__(cls, other):
        return isinstance(other, cls)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())


class DatetimeIndexConverter(BaseIndexConverter):
    def to_universal_index(self, index):
        """Convert index to universal type"""
        return pd.to_datetime(index, utc=True)

    def to_universal_delta(self, delta):
        """Convert index partition to universal type"""
        return pd.to_timedelta(delta)

    @property
    def reference(self):
        """Index reference"""
        return pd.to_datetime(0, utc=True)

    def __repr__(self):
        return "DatetimeIndexConverter"


class IntegerIndexConverter(BaseIndexConverter):
    def to_universal_index(self, index):
        """Convert index to universal type"""
        return np.int64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to universal type"""
        return np.int64(delta)

    @property
    def reference(self):
        """Index reference"""
        return 0

    def __repr__(self):
        return "IntegerIndexConverter"


class FloatIndexConverter(BaseIndexConverter):
    def to_universal_index(self, index):
        """Convert index to universal type"""
        return np.float64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to universal type"""
        return np.float64(delta)

    @property
    def reference(self):
        """Index reference"""
        return 0.0

    def __repr__(self):
        return "FloatIndexConverter"


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

    def __init__(
        self,
        index_converter,
        index_sync=False,
        tolerance=None,
        cache=None,
        cache_size=None,
    ):
        self._index_converter = index_converter
        self._index_sync = index_sync
        self._tolerance = tolerance
        self._cache = Path(cache) if cache else None
        self._cache_size = cache_size
        self._memory_cache = {}

        if self._cache and self._cache_size is None:
            raise ValueError("No 'cache_size' provided.")

        if not isinstance(self._index_converter, BaseIndexConverter):
            raise ValueError(
                "'index_converter' should be of type 'BaseIndexConverter'. "
                f"'{type(self._index_converter)}' given."
            )

        if self._cache and not self._cache.exists():
            self._cache.mkdir()

    def _md5hash(self, *args):
        return md5("_".join(map(lambda x: str(x), args)).encode()).hexdigest()

    @abstractproperty
    def _fingerprint(self):
        raise NotImplementedError()

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

    def get(self, start, end, refresh_cache=False):
        if not self._cache:
            return self._source_get(start, end)
        else:
            return self._cache_source_get(start, end, refresh_cache=refresh_cache)

    def _source_get(self, start, end):
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
    def _partition_start_end(start, end, partition, reference):
        start_part = reference + ((start - reference) // partition) * partition
        end_part = reference + ((end - reference) // partition) * partition
        if end_part == end:
            end_part += partition
        elif end_part < end:
            end_part += 2.0 * partition

        num_partitions = (end_part - start_part) // partition
        index_chunks = start_part + partition * np.arange(
            0, num_partitions, 1, dtype="int64"
        )
        return zip(index_chunks[:-1], index_chunks[1:])

    def _is_cached(self, id_):
        return (self._cache / id_).exists()

    def _cache_read(self, id_):
        dataframe = pd.read_feather(self._cache / id_).set_index(id_)
        dataframe.index.name = None
        return dataframe

    def _cache_write(self, id_, dataframe):
        dataframe.index.name = id_
        dataframe = dataframe.reset_index()
        dataframe.to_feather(self._cache / id_)

    def _cache_source_get(self, start, end, refresh_cache=False):
        """Get data from cache. Fall back to source if not available in cache."""

        chunks_universal = self._partition_start_end(
            self._index_converter.to_universal_index(start),
            self._index_converter.to_universal_index(end),
            self._index_converter.to_universal_delta(self._cache_size),
            self._index_converter.to_universal_index(self._index_converter.reference),
        )

        df_list = []
        memory_cache_update = {}
        for start_universal_i, end_universal_i in chunks_universal:
            chunk_id = self._md5hash(
                self._fingerprint, start_universal_i, end_universal_i
            )

            print(start_universal_i, end_universal_i)
            if not refresh_cache and chunk_id in self._memory_cache.keys():
                print("Get from memory")
                df_i = self._memory_cache[chunk_id]
            elif not refresh_cache and self._is_cached(chunk_id):
                print("Get from cache")
                df_i = self._cache_read(chunk_id)
            else:
                print("Get from source")
                df_i = self._source_get(
                    self._index_converter.to_native_index(start_universal_i),
                    self._index_converter.to_native_index(end_universal_i),
                )
                df_i = df_i.loc[start_universal_i:end_universal_i]
                self._cache_write(chunk_id, df_i.copy(deep=True))
            df_list.append(df_i)
            memory_cache_update[chunk_id] = df_i

        self._memory_cache = memory_cache_update
        start_universal = self._index_converter.to_universal_index(start)
        end_universal = self._index_converter.to_universal_index(end)
        return pd.concat(df_list).loc[start_universal:end_universal]

    def _sync_data(self, data, tolerance):
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

        tolerance = self._index_converter.to_universal_delta(tolerance)
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

    def iter(self, start, end, index_mode="start", refresh_cache=False):
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
            (index_i, self.get(start_i, end_i, refresh_cache=refresh_cache))
            for index_i, start_i, end_i in zip(index, start, end)
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
        cache_size=None,
        **get_kwargs,
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._get_kwargs = get_kwargs

        if index_type == "datetime":
            index_converter = DatetimeIndexConverter()
            cache_size = cache_size or "24H"
        elif index_type == "integer":
            index_converter = IntegerIndexConverter()
            cache_size = cache_size or 8.64e13
        else:
            raise ValueError("'index_type' should be 'datetime' or 'integer'.")

        super().__init__(
            index_converter,
            index_sync=index_sync,
            tolerance=tolerance,
            cache=cache,
            cache_size=cache_size,
        )

    @property
    def _fingerprint(self):
        return self._md5hash(
            self._labels,
            self._get_kwargs,
            self._index_converter,
            self._index_sync,
            self._tolerance,
        )

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


class _NullDataSource(BaseDataSource):
    """
    Will return empty data.

    Parameters
    ----------
    labels : list-like
        Data labels.
    index_type : str or callable
        Index type. Should be 'datetime', 'integer' or a callable.
    """

    def __init__(self, index_converter, labels=None):
        self._labels = tuple(labels) if labels else ()
        super().__init__(index_converter, index_sync=False, cache=None)

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels)

    @property
    def _fingerprint(self):
        return self._md5hash(self._index_converter, self._labels)

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

        index_conv_set = set(
            [source._index_converter for source in self._sources if source]
        )
        if len(index_conv_set) != 1:
            raise ValueError("The data sources does not share the same 'index_type'.")
        index_converter = index_conv_set.pop()

        labels_set = set(
            [tuple(sorted(source.labels)) for source in self._sources if source]
        )
        if len(labels_set) != 1:
            raise ValueError("The data sources does not share the same 'labels'.")
        self._labels = labels_set.pop()

        super().__init__(index_converter, index_sync=False, cache=None)

        self._sources = [
            source if source else _NullDataSource(index_converter, self._labels)
            for source in self._sources
        ]

        self._index = {
            index: self._index_converter.to_universal_index(index)
            for index in index_source_flat[::2]
        }
        if list(self._index.values()) != sorted(self._index.values()):
            raise ValueError(
                "indecies in 'index_source' must be in strictly increasing order."
            )

    def _fingerprint(self):
        fingerprint_list = [source._fingerprint for source in self._sources]
        return self._md5hash(*fingerprint_list)

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels)

    def _get(self, *args, **kwargs):
        return NotImplemented

    def get(self, start, end, refresh_cache=False):
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

        start_uni = self._index_converter.to_universal_index(start)
        end_uni = self._index_converter.to_universal_index(end)

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
            sources_list.insert(
                0, first_source or _NullDataSource(self._index_converter, self._labels)
            )

        while index_uni_list and index_uni_list[-1] >= end_uni:
            index_uni_list.pop()
            index_list.pop()
            sources_list.pop()
        else:
            index_list.append(end)

        data_list = [
            source_i.get(start_i, end_i, refresh_cache=refresh_cache)
            for (start_i, end_i), source_i in zip(pairwise(index_list), sources_list)
        ]
        return pd.concat(data_list)
