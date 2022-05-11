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
    """
    Abstract class for 'index converters'.
    """

    @abstractmethod
    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        raise NotImplementedError()

    @abstractmethod
    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        raise NotImplementedError()

    def to_native_index(self, index):
        """Convert index to 'native' type"""
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
    """
    Datetime index converter.

    Index values will be passed on to meth:`pandas.to_datetime` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return pd.to_datetime(index, utc=True)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        return pd.to_timedelta(delta)

    @property
    def reference(self):
        """Index reference"""
        return pd.to_datetime(0, utc=True)

    def __repr__(self):
        return "DatetimeIndexConverter"


class IntegerIndexConverter(BaseIndexConverter):
    """
    Integer index converter.

    Index values will be passed on to meth:`numpy.int64` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return np.int64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        return np.int64(delta)

    @property
    def reference(self):
        """Index reference"""
        return 0

    def __repr__(self):
        return "IntegerIndexConverter"


class FloatIndexConverter(BaseIndexConverter):
    """
    Float index converter.

    Index values will be passed on to meth:`numpy.float64` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return np.float64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
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
    index_converter : obj
        Index converter (see Notes).
    index_sync : bool, optional
        If the index should be synced. If ``True``, a valid `tolerance` must be given.
    tolerance :
        Tolerance limit for syncing. Should be given in anything that
        :meth:`index_converter.to_universal_delta` can parse. If `index_sync` is
        set to ``True``, datapoints that are closer than the `tolerance` are merged
        so that they share a common index. The common index will be the first index
        of the neighboring datapoints.
    cache : str, optional
        Cache folder. If ``None`` (default), caching is disabled.
    cache_size :
        Cache size as an index partition (see Notes).

    Notes
    -----
    - The `index_converter` is used to convert index values to a universal type.
      For datetime-like indices, use a :class:`DatetimeIndexConverter`. For integer-like
      indices, use a :class:`IntegerIndexConverter`. Other index converters can
      be set up by inheriting from :class:`BaseIndexConverter`.

    - Caching will speed-up the data downloading, if the same data is requested multiple
      times. First time some data is retrieved from the source, it will be split
      up in 'chunks' and stored in a local folder. Then, the data is more readily
      available next time it is requested.

      The `cache_size` determines how to partition the data in chunks. It describes
      the size of each cache chunk by providing a index span. The `cache_size` should
      be given as a dtype that the :meth:`index_converter.to_universal_delta` can
      parse.

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
        self._tolerance = (
            self._index_converter.to_universal_delta(tolerance) if tolerance else None
        )
        self._cache = Path(cache) if cache else None
        self._cache_size = (
            self._index_converter.to_universal_delta(cache_size) if cache_size else None
        )
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
        """
        Returns a hash from the list of input values.
        """
        return md5("_".join(map(lambda x: str(x), args)).encode()).hexdigest()

    @abstractproperty
    def _fingerprint(self):
        """Fingerprint that uniquely identifies the configuration of the data source."""
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
        """
        Get data from source.

        Parameters
        ----------
        start :
            Start index of the data. Will be passed on to the :meth:`_get` method.
        end :
            End index of the data. Will be passed on to the :meth:`_get` method.
        refresh_cache : bool, optional
            Refresh cache data.

        Returns
        -------
        pandas.DataFrame
            Source data.
        """
        if not self._cache:
            return self._source_get(start, end)
        elif refresh_cache:
            self._build_cache(start, end)
        return self._cache_get(start, end)

    def _build_cache(self, start, end):
        """
        Build/refresh cache.

        Parameters
        ----------
        start :
            Start index.
        end :
            End index.
        """
        start_end_uni = self._partition_start_end(
            self._index_converter.to_universal_index(start),
            self._index_converter.to_universal_index(end),
            self._index_converter.to_universal_delta(self._cache_size),
            self._index_converter.to_universal_index(self._index_converter.reference),
        )
        dataframe = self._source_get(
            self._index_converter.to_native_index(start_end_uni[0]),
            self._index_converter.to_native_index(start_end_uni[-1]),
        )
        memory_cache_update = {}
        for start_uni_i, end_uni_i in zip(start_end_uni[:-1], start_end_uni[1:]):
            chunk_id = self._md5hash(self._fingerprint, start_uni_i, end_uni_i)
            df_i = self._slice(dataframe, start_uni_i, end_uni_i)
            self._cache_write(chunk_id, df_i.copy(deep=True))
            memory_cache_update[chunk_id] = df_i
        self._memory_cache = memory_cache_update

    def _cache_get(self, start, end):
        """
        Get data from cache. Data is retrieved from source if a partition is not
        available in cache.

        Parameters
        ----------
        start :
            Start index of the data.
        end :
            End index of the data.
        """
        start_end_uni = self._partition_start_end(
            self._index_converter.to_universal_index(start),
            self._index_converter.to_universal_index(end),
            self._index_converter.to_universal_delta(self._cache_size),
            self._index_converter.to_universal_index(self._index_converter.reference),
        )

        df_list = []
        memory_cache_update = {}
        for start_uni_i, end_uni_i in zip(start_end_uni[:-1], start_end_uni[1:]):
            chunk_id = self._md5hash(self._fingerprint, start_uni_i, end_uni_i)
            if chunk_id in self._memory_cache.keys():
                df_i = self._memory_cache[chunk_id]
            elif self._is_cached(chunk_id):
                df_i = self._cache_read(chunk_id)
            else:
                df_i = self._source_get(
                    self._index_converter.to_native_index(start_uni_i),
                    self._index_converter.to_native_index(end_uni_i),
                )
                df_i = self._slice(df_i, start_uni_i, end_uni_i)
                self._cache_write(chunk_id, df_i.copy(deep=True))
            df_list.append(df_i)
            memory_cache_update[chunk_id] = df_i

        self._memory_cache = memory_cache_update
        start_uni = self._index_converter.to_universal_index(start)
        end_uni = self._index_converter.to_universal_index(end)
        return pd.concat(df_list).loc[start_uni:end_uni]

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
        if end_part < end:
            end_part += partition

        num_partitions = (end_part - start_part) // partition
        if (end_part - start_part) % partition:
            num_partitions += 1
        return start_part + partition * np.arange(
            0, num_partitions + 1, 1, dtype="int64"
        )

    def _is_cached(self, id_):
        """Check if data is cached."""
        return (self._cache / id_).exists()

    def _cache_read(self, id_):
        """Read data from cache file."""
        dataframe = pd.read_feather(self._cache / id_).set_index(id_)
        dataframe.index.name = None
        return dataframe

    def _cache_write(self, id_, dataframe):
        """Write dataframe to cache file."""
        dataframe.index.name = id_
        dataframe = dataframe.reset_index()
        dataframe.to_feather(self._cache / id_)

    @staticmethod
    def _slice(df, start, end):
        """Slice dataframe and exclude endpoint"""
        df = df.loc[start:end]
        if not df.empty and (df.index[-1] == end):
            df = df.iloc[:-1]
        return df

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
    index_type : str or obj
        Index type (see Notes). Should be 'datetime', 'integer' or an `index converter`
        object.
    index_sync : bool, optional
        If the index should be synced. If ``True``, a valid tolerance must be given.
    tolerance :
        Tolerance limit for syncing. Should be given in anything that
        :meth:`index_converter.to_universal_delta` can parse. If `index_sync` is
        set to ``True``, datapoints that are closer than the `tolerance` are merged
        so that they share a common index. The common index will be the first index
        of the neighboring datapoints.
    cache : str, optional
        Cache folder. If ``None`` (default), caching is disabled.
    cache_size :
        Cache size as an index partition (see Notes). Defaults to ``'24H'`` if the
        `index_type` is 'datetime', otherwise ``None`` is default.
    **get_kwargs : optional
        Keyword arguments that will be passed on to the ``drio_client.get`` method.

    Notes
    -----
    - The `index_type` describes how to interpret the data index and convert it
      to a 'universal type'. By universal type, we mean a data type that is common
      for all indices related to the data. An internal ``index_converter`` object
      is set based on the `index_type`.

      If `index_type` is set to 'datetime', a datetime-like index is expected. Indices
      are then converted to a universal type using the :class:`DatetimeIndexConverter`.

      If `index_type` is set to 'integer', an integer-like index is expected. Indices
      are then converted to a universal type using the :class:`IntegerIndexConverter`.

      Custom index types are provided through giving an 'index converter' object as
      the `index_type`. The index converter should inherit from the abstract class,
      :class:`BaseIndexConverter`.

    - Caching will speed-up the data downloading, if the same data is requested multiple
      times. First time some data is retrieved from the source, it will be split
      up in 'chunks' and stored in a local folder. Then, the data is more readily
      available next time it is requested.

      The `cache_size` determines how to partition the data in chunks. It describes
      the size of each cache chunk by providing a index span. The `cache_size` should
      be given as a dtype that the :meth:`index_converter.to_universal_delta` can
      parse.

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
        self._get_kwargs = get_kwargs

        self._labels = {lab: id.strip() for lab, id in labels.items()}

        if index_type == "datetime":
            index_converter = DatetimeIndexConverter()
            cache_size = cache_size or "24H"
        elif index_type == "integer":
            index_converter = IntegerIndexConverter()
        elif isinstance(index_type, BaseIndexConverter):
            index_converter = index_type
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
        """Fingerprint that uniquely identifies the configuration of the data source."""
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
    index_converter : obj
        Index converter.
    labels : list-like
        Data labels.
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
        """Fingerprint that uniquely identifies the configuration of the data source."""
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
        in the sequence (see Example). If a source is set to ``None``, empty data will
        be returned for that period.

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

    @property
    def _fingerprint(self):
        """Fingerprint that uniquely identifies the configuration of the data source."""
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
        refresh_cache : bool
            Refresh cache if True.

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
        return pd.concat(data_list).infer_objects()
