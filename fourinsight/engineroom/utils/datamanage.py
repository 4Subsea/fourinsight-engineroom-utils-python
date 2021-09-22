from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd


class BaseDataSource(ABC):
    @abstractproperty
    def labels(self):
        """Data source labels."""
        raise NotImplementedError()

    @abstractmethod
    def _get(self, start, end):
        """
        Get data fro source.

        Parameters
        ----------
        start : str or datetime-like
            Start time (inclusive) of the data, given as anything pandas.to_datetime
            is able to parse.
        end : str or datetime-like
            Stop time (inclusive) of the data, given as anything pandas.to_datetime
            is able to parse.

        Returns
        -------
        dict
            Label and data as key/value pairs. The data is returned as type
            ``pandas.Series``.
        """
        raise NotImplementedError()

    def get(self, start, end, index_sync=True, tolerance=None):
        """
        Get source data.

        Parameters
        ----------
        start : str or datetime-like
            Start time (inclusive) of the data, given as anything pandas.to_datetime
            is able to parse.
        end : str or datetime-like
            Stop time (inclusive) of the data, given as anything pandas.to_datetime
            is able to parse.
        index_sync : bool, optional
            Controls if index is synced. If ``True``, a valid ``tolerance`` value
            must be given.
        tolerance : int, float or pandas.Timedelta
            Tolerance limit for syncing (see Notes). If ``index_sync`` is ``True``,
            data points that are closer that the tolerance are merged so that they
            share a common index. The common index will be the first index of the
            neighboring data points.

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

        index_common = np.sort(
            np.unique(np.concatenate([series.index for series in data.values()]))
        )
        index_keep = np.r_[True, (np.diff(index_common) > tolerance)]
        index_common = index_common[index_keep]
        df_synced = pd.DataFrame(index=index_common)

        for key, series in data.items():
            if isinstance(series, pd.Series):
                series.name = key

            df_synced = pd.merge_asof(
                df_synced,
                series,
                left_index=True,
                right_index=True,
                tolerance=tolerance,
                direction="nearest",
            )

        return df_synced

    # @staticmethod
    # def _sync_series(series_a, series_b, tolerance):
    #     """
    #     Merge series so that they share a common index.

    #     Parameters
    #     ----------
    #     series_a : pandas.Series
    #         Data series.
    #     series_b : pandas.Series
    #         Data series.
    #     tolerance : int, float or pandas.Timedelta
    #         Tolerance limit for syncing.
    #     """
    #     merge_a = pd.merge_asof(
    #         series_a,
    #         series_b,
    #         left_index=True,
    #         right_index=True,
    #         tolerance=tolerance,
    #         direction="nearest",
    #     )
    #     merge_b = pd.merge_asof(
    #         series_b,
    #         series_a,
    #         left_index=True,
    #         right_index=True,
    #         tolerance=tolerance,
    #         direction="nearest",
    #     )
    #     df_synced = pd.concat([merge_a, merge_b]).sort_index()
    #     idx_keep = np.r_[True, (np.diff(df_synced.index) > tolerance)]
    #     return df_synced[idx_keep]

    # def _synchronize(self, data, tolerance):
    #     """
    #     Synchronize data series.

    #     Parameters
    #     ----------
    #     data : dict
    #         Label and data as key/value pairs. The data should be of type ``pandas.Series``.
    #     tolerance : int, float or pandas.Timedelta
    #         Tolerance limit for syncing.
    #     """
    #     for i, (key, series_i) in enumerate(data.items()):
    #         if isinstance(series_i, pd.Series):
    #             series_i.name = key
    #         if i == 0:
    #             df_synced = pd.DataFrame(series_i)
    #         else:
    #             df_synced = self._sync_series(df_synced, series_i, tolerance)
    #     return df_synced


class DrioDataSource(BaseDataSource):
    def __init__(self, drio_client, **labels):
        self._drio_client = drio_client
        self._labels = labels

    def _get(self, start, end):
        data = {}
        for label in self.labels():
            data[label] = self._drio_client.get(
                self._labels[label], start=start, end=end
            )
        return data

    @property
    def labels(self):
        """Data source labels."""
        return tuple(self._labels.keys())
