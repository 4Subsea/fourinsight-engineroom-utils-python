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
    **get_kwargs : optional
        Keyword arguments that will be passed on to the ``drio_client.get`` method.

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
        **get_kwargs,
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._get_kwargs = get_kwargs
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
    NullDataSource.
    """

    def __init__(self, labels=None):
        self._labels = labels if labels else ()
        self._index_sync = False
        self._tolerance = None

    @property
    def labels(self):
        return tuple(self._labels)

    def _get(self, start, end):
        return {label: pd.Series([], dtype="object") for label in self._labels}


class CompositeDataSource(BaseDataSource):
    def __init__(self, index_source, index_sync=False, tolerance=None):
        self._index_attached, self._sources = np.asarray(index_source).T
        self._index_sync = index_sync
        self._tolerance = tolerance

        self._labels = self._verify_labels(self._sources)
        self._sources = np.where(
            np.equal(self._sources, None), NullDataSource(self._labels), self._sources
        )

        self._index_attached_universal = self._universal_index_converter(
            self._index_attached
        )

    @staticmethod
    def _universal_index_converter(index):
        """
        Convert index to universal type that is comparable.
        """
        return pd.to_datetime(index, utc=True)

    @property
    def labels(self):
        return tuple(self._labels)

    def _get(self, start, end):
        start_universal = self._universal_index_converter(start)
        end_universal = self._universal_index_converter(end)

        attached_after_start = self._index_attached_universal > start_universal
        attached_before_end = self._index_attached_universal < end_universal
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

    @staticmethod
    def _concat_data(data_list):
        data = {key: pd.Series([], dtype="object") for key in data_list[0].keys()}
        for data_i in data_list:
            for key, series in data_i.items():
                data[key] = pd.concat([data[key], series])
        return data

    @staticmethod
    def _verify_labels(source_list):
        """
        Check that all sources have the same labels.

        Returns
        -------
        list-like
            Labels.
        """
        labels = set([source.labels for source in source_list if source is not None])
        if len(labels) == 0:
            return None
        elif len(labels) == 1:
            return list(labels)[0]
        else:
            raise ValueError("All source labels are not equal.")
