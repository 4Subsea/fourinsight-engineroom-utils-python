from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fourinsight.engineroom.utils import DrioDataSource
from fourinsight.engineroom.utils.datamanage import BaseDataSource


class BaseDataSourceForTesting(BaseDataSource):
    def labels(self):
        super().labels()

    def _get(self, start, end):
        super()._get(start, end)


# class Test_BaseDataSource:
#     def test_labels(self):
#         source = BaseDataSourceForTesting()
#         with pytest.raises(NotImplementedError):
#             source.labels()

#     def test_get(self):
#         source = BaseDataSourceForTesting()
#         with pytest.raises(NotImplementedError):
#             source._get("2020-01-01", "2021-01-01")

#     def test_sync_series_int_index(self):
#         index_a = range(0, 1000, 10)
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a, name="a")

#         index_b = index_a + np.random.randint(0, 4, len(index_a))
#         values_b = [2.0] * len(index_a)
#         series_b = pd.Series(data=values_b, index=index_b, name="b")

#         index_expect = index_a
#         values_a_expect = values_a
#         values_b_expect = values_b
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
#         )

#         df_out = BaseDataSource._sync_series(series_a, series_b, 5)

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     def test_sync_series_float_index(self):
#         series_a = pd.Series(
#             data=[1, 2, 3, 4, 5], index=[1.0, 2.0, 3.0, 4.0, 5.0], name="a"
#         )

#         series_b = pd.Series(
#             data=[10, 20, 30, 40], index=[0.95, 2.09999, 3.0, 4.2], name="b"
#         )

#         df_expect = pd.DataFrame(
#             data={"a": [1, 2, 3, 4, np.nan, 5], "b": [10, 20, 30, np.nan, 40, np.nan]},
#             index=[0.95, 2.0, 3.0, 4.0, 4.2, 5.0],
#         )

#         df_out = BaseDataSource._sync_series(series_a, series_b, 0.1)
#         pd.testing.assert_frame_equal(df_out, df_expect)

#     def test_sync_series_timestamp_index(self):
#         index_a = pd.DatetimeIndex(
#             [
#                 "2020-01-01 00:00:00.1",
#                 "2020-01-01 00:00:00.2",
#                 "2020-01-01 00:00:00.3",
#                 "2020-01-01 00:00:00.4",
#             ],
#             tz="utc",
#         )
#         series_a = pd.Series(data=[1.0, 2.0, 3.0, 4.0], index=index_a, name="a")

#         index_b = pd.DatetimeIndex(
#             [
#                 "2020-01-01 00:00:00.109",
#                 "2020-01-01 00:00:00.19999",
#                 "2020-01-01 00:00:00.3",
#                 "2020-01-01 00:00:00.5",
#             ],
#             tz="utc",
#         )
#         series_b = pd.Series(data=[1.0, 2.0, 3.0, 4.0], index=index_b, name="b")

#         df_out = BaseDataSource._sync_series(
#             series_a, series_b, pd.to_timedelta(0.01, "s")
#         )

#         index_expect = pd.DatetimeIndex(
#             [
#                 "2020-01-01 00:00:00.1",
#                 "2020-01-01 00:00:00.19999",
#                 "2020-01-01 00:00:00.3",
#                 "2020-01-01 00:00:00.4",
#                 "2020-01-01 00:00:00.5",
#             ],
#             tz="utc",
#         )
#         df_expect = pd.DataFrame(
#             data={"a": [1.0, 2.0, 3.0, 4.0, np.nan], "b": [1.0, 2.0, 3.0, np.nan, 4.0]},
#             index=index_expect,
#         )
#         pd.testing.assert_frame_equal(df_out, df_expect)

#     def test_sync_series_timestamp_index2(self):
#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a, name="a")

#         index_b = index_a + pd.to_timedelta(
#             np.random.randint(0, 999, len(index_a)), "ms"
#         )
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b, name="b")

#         df_out = BaseDataSource._sync_series(
#             series_a, series_b, pd.to_timedelta(1, "s")
#         )

#         index_expect = index_a
#         values_a_expect = values_a
#         values_b_expect = values_b
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect, check_freq=False)

#     def test_sync_series_timestamp_index3(self):
#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a, name="a")

#         index_b = index_a - pd.to_timedelta(
#             np.random.randint(0, 999, len(index_a)), "ms"
#         )
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b, name="b")

#         df_out = BaseDataSource._sync_series(
#             series_a, series_b, pd.to_timedelta(1, "s")
#         )

#         index_expect = index_b
#         values_a_expect = values_a
#         values_b_expect = values_b
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect, check_freq=False)

#     def test__synchronize(self):
#         source = BaseDataSourceForTesting()

#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a)

#         index_b = index_a - pd.to_timedelta(
#             np.random.randint(0, 499, len(index_a)), "ms"
#         )
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b)

#         index_c = index_a + pd.to_timedelta(
#             np.random.randint(0, 499, len(index_a)), "ms"
#         )
#         values_c = [3.0] * len(index_c)
#         series_c = pd.Series(data=values_c, index=index_c)

#         data = {"a": series_a, "b": series_b, "c": series_c}

#         df_out = source._synchronize(data, pd.to_timedelta("1s"))

#         index_expect = index_b
#         values_a_expect = values_a
#         values_b_expect = values_b
#         values_c_expect = values_c
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect, "c": values_c_expect},
#             index=index_expect,
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     def test__synchronize_empty_series(self):
#         source = BaseDataSourceForTesting()

#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a)

#         index_b = index_a - pd.to_timedelta(
#             np.random.randint(0, 499, len(index_a)), "ms"
#         )
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b)

#         index_c = pd.DatetimeIndex([])
#         values_c = []
#         series_c = pd.Series(data=values_c, index=index_c, dtype="float64")

#         data = {"a": series_a, "b": series_b, "c": series_c}

#         df_out = source._synchronize(data, pd.to_timedelta("1s"))

#         index_expect = index_b
#         values_a_expect = values_a
#         values_b_expect = values_b
#         values_c_expect = [np.nan] * len(index_expect)
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect, "c": values_c_expect},
#             index=index_expect,
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     def test__synchronize_single_series(self):
#         source = BaseDataSourceForTesting()

#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a)

#         data = {"a": series_a}

#         df_out = source._synchronize(data, pd.to_timedelta("1s"))

#         index_expect = index_a
#         values_a_expect = values_a
#         df_expect = pd.DataFrame(data={"a": values_a_expect}, index=index_expect)

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     @patch.object(BaseDataSourceForTesting, "_get")
#     def test_get_nosync(self, mock_get):
#         source = BaseDataSourceForTesting()

#         index_a = [1, 2, 3, 4]
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a)

#         index_b = [1, 10, 100, 1000]
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b)

#         index_c = [1, 2, 7]
#         values_c = [3.0] * len(index_c)
#         series_c = pd.Series(data=values_c, index=index_c)

#         data = {"a": series_a, "b": series_b, "c": series_c}

#         mock_get.return_value = data

#         df_out = source.get("2020-01-01 00:00", "2020-01-02 00:00", index_sync=False)

#         df_expect = pd.DataFrame(
#             data={
#                 "a": [1.0, 1.0, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan],
#                 "b": [2.0, np.nan, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0],
#                 "c": [3.0, 3.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan],
#             },
#             index=[1, 2, 3, 4, 7, 10, 100, 1000],
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     @patch.object(BaseDataSourceForTesting, "_get")
#     def test_get_sync(self, mock_get):
#         source = BaseDataSourceForTesting()

#         index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
#         values_a = [1.0] * len(index_a)
#         series_a = pd.Series(data=values_a, index=index_a)

#         index_b = index_a - pd.to_timedelta(
#             np.random.randint(0, 499, len(index_a)), "ms"
#         )
#         values_b = [2.0] * len(index_b)
#         series_b = pd.Series(data=values_b, index=index_b)

#         index_c = index_a + pd.to_timedelta(
#             np.random.randint(0, 499, len(index_a)), "ms"
#         )
#         values_c = [3.0] * len(index_c)
#         series_c = pd.Series(data=values_c, index=index_c)

#         data = {"a": series_a, "b": series_b, "c": series_c}

#         mock_get.return_value = data

#         df_out = source.get(
#             "2020-01-01 00:00",
#             "2020-01-02 00:00",
#             index_sync=True,
#             tolerance=pd.to_timedelta("1s"),
#         )

#         index_expect = index_b
#         values_a_expect = values_a
#         values_b_expect = values_b
#         values_c_expect = values_c
#         df_expect = pd.DataFrame(
#             data={"a": values_a_expect, "b": values_b_expect, "c": values_c_expect},
#             index=index_expect,
#         )

#         pd.testing.assert_frame_equal(df_out, df_expect)

#     @patch.object(BaseDataSourceForTesting, "_get")
#     def test_get_raises(self, mock_get):
#         source = BaseDataSourceForTesting()

#         with pytest.raises(ValueError):
#             source.get(
#                 "2020-01-01 00:00", "2020-01-02 00:00", index_sync=True, tolerance=None
#             )


class Test_BaseDataSource:

    def test_labels(self):
        source = BaseDataSourceForTesting()
        with pytest.raises(NotImplementedError):
            source.labels()

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
