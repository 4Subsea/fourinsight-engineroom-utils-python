from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fourinsight.engineroom.utils import DrioDataSource
from fourinsight.engineroom.utils.datamanage import BaseDataSource


class BaseDataSourceForTesting(BaseDataSource):
    @property
    def labels(self):
        super().labels

    def _get(self, start, end):
        super()._get(start, end)


class Test_BaseDataSource:

    def test_labels(self):
        source = BaseDataSourceForTesting()
        with pytest.raises(NotImplementedError):
            source.labels

    def test__get(self):
        source = BaseDataSourceForTesting()
        with pytest.raises(NotImplementedError):
            source._get("2020-01-01", "2021-01-01")

    def test__sync_data_datetimeindex(self):
        index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="5s")
        values_a = np.random.random(len(index_a))
        series_a = pd.Series(data=values_a, index=index_a)

        noise_b = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_b = index_a + noise_b
        values_b = np.random.random(len(index_b))
        series_b = pd.Series(data=values_b, index=index_b)

        noise_c = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_c = index_a - noise_c
        values_c = np.random.random(len(index_c))
        series_c = pd.Series(data=values_c, index=index_c)

        noise_d = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_d = index_a + noise_d
        values_d = np.random.random(len(index_d))
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, pd.to_timedelta("2s"))

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_data_intindex(self):
        index_a = np.arange(0, 1000, 10)
        values_a = np.random.random(len(index_a))
        series_a = pd.Series(data=values_a, index=index_a)

        noise_b = np.random.randint(0, 2, len(index_a))
        index_b = index_a + noise_b
        values_b = np.random.random(len(index_b))
        series_b = pd.Series(data=values_b, index=index_b)

        noise_c = np.random.randint(0, 2, len(index_a))
        index_c = index_a - noise_c
        values_c = np.random.random(len(index_c))
        series_c = pd.Series(data=values_c, index=index_c)

        noise_d = np.random.randint(0, 2, len(index_a))
        index_d = index_a + noise_d
        values_d = np.random.random(len(index_d))
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, 4)

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_data_floatindex(self):
        index_a = np.arange(0, 1000, 0.1)
        values_a = np.random.random(len(index_a))
        series_a = pd.Series(data=values_a, index=index_a)

        noise_b = np.random.random(len(index_a)) / 100.
        index_b = index_a + noise_b
        values_b = np.random.random(len(index_b))
        series_b = pd.Series(data=values_b, index=index_b)

        noise_c = np.random.random(len(index_a)) / 100.
        index_c = index_a - noise_c
        values_c = np.random.random(len(index_c))
        series_c = pd.Series(data=values_c, index=index_c)

        noise_d = np.random.random(len(index_a)) / 100.
        index_d = index_a + noise_d
        values_d = np.random.random(len(index_d))
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, 2./100.)

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_missing_values_and_other_dtypes(self):
        index_a = [1., 2., 3.]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, 0.2)

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0]
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_dataframe(self):
        index_a = [1., 2., 3.]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.DataFrame(data={"a": values_a}, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.DataFrame(data={"b": values_b}, index=index_b)

        index_c = [2.09, 5.]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.DataFrame(data={"c": values_c}, index=index_c)

        index_d = [0.95, 5.]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.DataFrame(data={"d": values_d}, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, 0.2)

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0]
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_small_tolerance_yields_no_syncing(self):
        index_a = [1., 2., 3.]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        df_out = BaseDataSource._sync_data(data, 0.0001)   # small tolerance yields no syncing

        df_expect = pd.DataFrame(
            data={
                "a": [np.nan, "value_a1", np.nan, np.nan, 2.0, np.nan, "value_a3", np.nan],
                "b": [np.nan, np.nan, "value_b1", 1, np.nan, np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan, np.nan, np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.0, 1.01, 1.91, 2.0, 2.09, 3.0, 5.0]
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_get_nosync(self, mock_get):
        index_a = [1., 2., 3.]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        mock_get.return_value = data

        source = BaseDataSourceForTesting()
        df_out = source.get("<start-time>", "<end-time>", index_sync=False)

        df_expect = pd.DataFrame(
            data={
                "a": [np.nan, "value_a1", np.nan, np.nan, 2.0, np.nan, "value_a3", np.nan],
                "b": [np.nan, np.nan, "value_b1", 1, np.nan, np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan, np.nan, np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.0, 1.01, 1.91, 2.0, 2.09, 3.0, 5.0]
        )

        pd.testing.assert_frame_equal(df_out, df_expect)
        mock_get.assert_called_once_with("<start-time>", "<end-time>")

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_get_sync(self, mock_get):
        index_a = [1., 2., 3.]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        mock_get.return_value = data

        source = BaseDataSourceForTesting()
        df_out = source.get("<start-time>", "<end-time>", index_sync=True, tolerance=0.2)

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0]
        )

        pd.testing.assert_frame_equal(df_out, df_expect)
        mock_get.assert_called_once_with("<start-time>", "<end-time>")

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_get_raises_no_tolerance(self, mock_get):
        source = BaseDataSourceForTesting()

        with pytest.raises(ValueError):
            source.get("<start-time>", "<end-time>", index_sync=True, tolerance=None)

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_get_sync_datetimeindex(self, mock_get):
        index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="5s")
        values_a = np.random.random(len(index_a))
        series_a = pd.Series(data=values_a, index=index_a)

        noise_b = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_b = index_a + noise_b
        values_b = np.random.random(len(index_b))
        series_b = pd.Series(data=values_b, index=index_b)

        noise_c = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_c = index_a - noise_c
        values_c = np.random.random(len(index_c))
        series_c = pd.Series(data=values_c, index=index_c)

        noise_d = pd.to_timedelta(np.random.randint(0, 999, len(index_a)), "ms")
        index_d = index_a + noise_d
        values_d = np.random.random(len(index_d))
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        mock_get.return_value = data

        source = BaseDataSourceForTesting()
        df_out = source.get(
            "<start-time>", "<end-time>", index_sync=True, tolerance=pd.to_timedelta("2s")
        )

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c
        )

        pd.testing.assert_frame_equal(df_out, df_expect)
