from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseDataSource(ABC):
    @abstractmethod
    def labels(self):
        """Data source labels."""
        raise NotImplementedError()

    @abstractmethod
    def _get(self, start, end):
        """
        Get data.

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
            Label and data as key/value pairs. The data must be of type ``pandas.Series``.
        """
        raise NotImplementedError()

    def get(self, start, end, index_sync=True, tolerance=None):
        data = self._get(start, end)
        if not index_sync:
            return pd.DataFrame(data)
        else:
            if not tolerance:
                raise ValueError("No tolerance given.")
            return self._synchronize(data, tolerance)

    @staticmethod
    def _sync_series(series_a, series_b, tolerance):
        merge_a = pd.merge_asof(
            series_a,
            series_b,
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            direction="nearest",
        )
        merge_b = pd.merge_asof(
            series_b,
            series_a,
            left_index=True,
            right_index=True,
            tolerance=tolerance,
            direction="nearest",
        )
        df_synced = pd.concat([merge_a, merge_b]).sort_index()
        idx_keep = np.r_[True, (np.diff(df_synced.index) > tolerance)]
        return df_synced[idx_keep]

    def _synchronize(self, data, tolerance):
        for i, (key, series_i) in enumerate(data.items()):
            if isinstance(series_i, pd.Series):
                series_i.name = key
            if i == 0:
                df_synced = pd.DataFrame(series_i)
            else:
                df_synced = self._sync_series(df_synced, series_i, tolerance)
        return df_synced


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

    def labels(self):
        """Data source labels."""
        return tuple(self._labels.keys())
