from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseDataSource(ABC):
    def _get(self, start, end):
        data = {}
        for label in self.labels():
            data[label] = self.get_label(label, start, end)
        return data

    def get(self, start, end, synchronization=True, tolerance=None):
        data = self._get(start, end)
        if not synchronization:
            return pd.DataFrame(data)
        else:
            return self._synchronize(list(data.values()), tolerance)

    @abstractmethod
    def get_label(self, label, start, end):
        raise NotImplementedError()

    @abstractmethod
    def labels(self):
        raise NotImplementedError()

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

    def _synchronize(self, data_list, tolerance):
        df_synced = data_list.pop(0)
        for s_i in data_list:
            df_synced = self._sync_series(df_synced, s_i, tolerance)
        return df_synced


class DrioDataSource(BaseDataSource):
    def __init__(self, drio_client, **labels):
        self._drio_client = drio_client
        self._labels = labels

    def get_label(self, label, start, end):
        series = self._drio_client.get(self._labels[label], start=start, end=end)
        series.name = label
        return series

    def labels(self):
        return tuple(self._labels.keys())
