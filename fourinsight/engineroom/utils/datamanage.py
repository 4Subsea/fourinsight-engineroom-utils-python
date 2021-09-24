import warnings
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd


class BaseDataSource(ABC):
    """
    Abstract class for data sources.
    """

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

    def get(self, start, end, index_sync=True, tolerance=None):
        """
        Get data from source, and perform syncing of the data index (optional).

        Parameters
        ----------
        start :
            Start index of the data. Will be passed on to the ``_get`` method.
        end :
            End index of the data. Will be passed on to the ``_get`` method.
        index_sync : bool, optional
            If the index should be synced. If True, a valid tolerance must be given.
        tolerance : int, float or pandas.Timedelta
            Tolerance limit for syncing (see Notes). If ``index_sync`` is set to True,
            datapoints that are closer that the tolerance are merged so that they
            share a common index. The common index will be the first index of the
            neighboring datapoints.

        Returns
        -------
        pandas.DataFrame
            Source data.

        Notes
        -----
        The tolerance must be of a type that is comparable to the data index. E.g.
        if the data has a ``DatetimeIndex``, the tolerance should be of type
        ``pandas.Timestamp``. And if the data has a ``Int64Index``, the tolerance
        should be an integer.

        """

        data = self._get(start, end)
        if not index_sync:
            return pd.DataFrame(data)
        else:
            if not tolerance:
                raise ValueError("No tolerance given.")
            return self._sync_data(data, tolerance)

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
                    f"Tolerance is greater than the mean sampling frequency of '{key}'."
                    + " This may lead to loss of data."
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


class DrioDataSource(BaseDataSource):
    """
    DataReservoir.io data source.

    Parameters
    ----------
    drio_client : obj
        DataReservoir.io client.
    lables : dict
        Labels and timeseries IDs as key/value pairs.
    get_kwargs : dict, optional
        Keyword arguments that will be passed on to the ``drio_client.get`` method.
        See datareservoirio documentation for details.
    """

    def __init__(
        self,
        drio_client,
        labels,
        get_kwargs={"convert_date": True, "raise_empty": False},
    ):
        self._drio_client = drio_client
        self._labels = labels
        self._get_opt = get_kwargs

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
                self._labels[label], start=start, end=end, **self._get_opt
            )
        return data

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels.keys())
