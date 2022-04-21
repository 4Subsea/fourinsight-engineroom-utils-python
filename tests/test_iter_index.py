import pandas as pd
from fourinsight.engineroom.utils import iter_index


class Test_date_range:
    def test_start_end_freq(self):
        start = "2020-01-01 00:00"
        end = "2020-01-01 05:00"
        freq = "1H"
        start_out, end_out = iter_index.date_range(start=start, end=end, freq=freq)

        date_range = pd.date_range(start=start, end=end, freq=freq)
        start_expect, end_expect = date_range[:-1], date_range[1:]

        assert (start_out == start_expect).all()
        assert (end_out == end_expect).all()

    def test_start_end_period(self):
        start = "2020-01-01 00:00"
        end = "2020-01-01 05:00"
        periods = 2
        start_out, end_out = iter_index.date_range(
            start=start, end=end, periods=periods
        )

        date_range = pd.date_range(start=start, end=end, periods=periods)
        start_expect, end_expect = date_range[:-1], date_range[1:]

        assert (start_out == start_expect).all()
        assert (end_out == end_expect).all()
