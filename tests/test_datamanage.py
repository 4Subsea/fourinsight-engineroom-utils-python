import numpy as np
import pandas as pd

from fourinsight.engineroom.utils import DrioDataSource
from fourinsight.engineroom.utils.datamanage import BaseDataSource


class Test_BaseDataSource:
    def test_sync_series_float_index(self):
        series_a = pd.Series(
            data=[1, 2, 3, 4, 5], index=[1.0, 2.0, 3.0, 4.0, 5.0], name="a"
        )
        series_b = pd.Series(
            data=[10, 20, 30, 40], index=[0.95, 2.09999, 3.0, 4.2], name="b"
        )
        df_out = BaseDataSource._sync_series(series_a, series_b, 0.1)
        df_expect = pd.DataFrame(
            data={"a": [1, 2, 3, 4, np.nan, 5], "b": [10, 20, 30, np.nan, 40, np.nan]},
            index=[0.95, 2.0, 3.0, 4.0, 4.2, 5.0],
        )
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
            index=index_expect
        )
        pd.testing.assert_frame_equal(df_out, df_expect)
