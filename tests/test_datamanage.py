import numpy as np
import pandas as pd

from fourinsight.engineroom.utils import DrioDataSource
from fourinsight.engineroom.utils.datamanage import BaseDataSource


class Test_BaseDataSource:

    def test_sync_series_int_index(self):
        index_a = range(0, 1000, 10)
        values_a = [1.0] * len(index_a)
        series_a = pd.Series(data=values_a, index=index_a, name="a")

        index_b = index_a + np.random.randint(0, 4, len(index_a))
        values_b = [2.0] * len(index_a)
        series_b = pd.Series(data=values_b, index=index_b, name="b")

        index_expect = index_a
        values_a_expect = values_a
        values_b_expect = values_b
        df_expect = pd.DataFrame(
            data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
        )

        df_out = BaseDataSource._sync_series(
            series_a, series_b, 5
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_sync_series_float_index(self):
        series_a = pd.Series(
            data=[1, 2, 3, 4, 5], index=[1.0, 2.0, 3.0, 4.0, 5.0], name="a"
        )

        series_b = pd.Series(
            data=[10, 20, 30, 40], index=[0.95, 2.09999, 3.0, 4.2], name="b"
        )

        df_expect = pd.DataFrame(
            data={"a": [1, 2, 3, 4, np.nan, 5], "b": [10, 20, 30, np.nan, 40, np.nan]},
            index=[0.95, 2.0, 3.0, 4.0, 4.2, 5.0],
        )

        df_out = BaseDataSource._sync_series(series_a, series_b, 0.1)
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_sync_series_timestamp_index(self):
        index_a = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00.1",
                "2020-01-01 00:00:00.2",
                "2020-01-01 00:00:00.3",
                "2020-01-01 00:00:00.4",
            ],
            tz="utc",
        )
        series_a = pd.Series(data=[1.0, 2.0, 3.0, 4.0], index=index_a, name="a")

        index_b = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00.109",
                "2020-01-01 00:00:00.19999",
                "2020-01-01 00:00:00.3",
                "2020-01-01 00:00:00.5",
            ],
            tz="utc",
        )
        series_b = pd.Series(data=[1.0, 2.0, 3.0, 4.0], index=index_b, name="b")

        df_out = BaseDataSource._sync_series(
            series_a, series_b, pd.to_timedelta(0.01, "s")
        )

        index_expect = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00.1",
                "2020-01-01 00:00:00.19999",
                "2020-01-01 00:00:00.3",
                "2020-01-01 00:00:00.4",
                "2020-01-01 00:00:00.5",
            ],
            tz="utc",
        )
        df_expect = pd.DataFrame(
            data={"a": [1.0, 2.0, 3.0, 4.0, np.nan], "b": [1.0, 2.0, 3.0, np.nan, 4.0]},
            index=index_expect,
        )
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_sync_series_timestamp_index2(self):
        index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
        values_a = [1.0] * len(index_a)
        series_a = pd.Series(data=values_a, index=index_a, name="a")

        index_b = index_a + pd.to_timedelta(
            np.random.randint(0, 999, len(index_a)), "ms"
        )
        values_b = [2.0] * len(index_b)
        series_b = pd.Series(data=values_b, index=index_b, name="b")

        df_out = BaseDataSource._sync_series(
            series_a, series_b, pd.to_timedelta(1, "s")
        )

        index_expect = index_a
        values_a_expect = values_a
        values_b_expect = values_b
        df_expect = pd.DataFrame(
            data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
        )

        pd.testing.assert_frame_equal(df_out, df_expect, check_freq=False)

    def test_sync_series_timestamp_index3(self):
        index_a = pd.date_range("2020-01-01 00:00", "2020-02-01 00:00", freq="2s")
        values_a = [1.0] * len(index_a)
        series_a = pd.Series(data=values_a, index=index_a, name="a")

        index_b = index_a - pd.to_timedelta(
            np.random.randint(0, 999, len(index_a)), "ms"
        )
        values_b = [2.0] * len(index_b)
        series_b = pd.Series(data=values_b, index=index_b, name="b")

        df_out = BaseDataSource._sync_series(
            series_a, series_b, pd.to_timedelta(1, "s")
        )

        index_expect = index_b
        values_a_expect = values_a
        values_b_expect = values_b
        df_expect = pd.DataFrame(
            data={"a": values_a_expect, "b": values_b_expect}, index=index_expect
        )

        pd.testing.assert_frame_equal(df_out, df_expect, check_freq=False)
