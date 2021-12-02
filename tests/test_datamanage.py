import os
import types
from hashlib import md5
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from fourinsight.engineroom.utils import CompositeDataSource, DrioDataSource
from fourinsight.engineroom.utils.datamanage import (
    BaseDataSource,
    DatetimeIndexConverter,
    FloatIndexConverter,
    IntegerIndexConverter,
    _NullDataSource,
)


class BaseDataSourceForTesting(BaseDataSource):
    @property
    def labels(self):
        super().labels

    @property
    def _fingerprint(self):
        super()._fingerprint

    def _get(self, start, end):
        super()._get(start, end)


class Test_BaseDataSource:
    def test__init__default(self):
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(),
        )
        assert isinstance(source._index_converter, DatetimeIndexConverter)
        assert source._index_sync is False
        assert source._tolerance is None
        assert source._cache is None
        assert source._cache_size is None
        assert source._memory_cache == {}

    def test__init__cache(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            IntegerIndexConverter(),
            index_sync=True,
            tolerance=1,
            cache=cache_dir,
            cache_size=1,
        )
        assert isinstance(source._index_converter, IntegerIndexConverter)
        assert source._index_sync is True
        assert source._tolerance == 1
        assert source._cache == Path(cache_dir)
        assert source._cache_size == 1
        assert source._memory_cache == {}
        assert cache_dir.exists()

    def test__init__raises_converter(self):
        with pytest.raises(ValueError):
            BaseDataSourceForTesting(Mock())

    def test__init__raises_cache_size(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        with pytest.raises(ValueError):
            BaseDataSourceForTesting(
                IntegerIndexConverter(), cache=cache_dir, cache_size=None
            )

    def test__md5hash(self):
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(),
        )

        out = source._md5hash("test", 1, 2.0)
        expect = md5("test_1_2.0".encode()).hexdigest()
        assert out == expect

    def test__fingerprint(self):
        source = BaseDataSourceForTesting(DatetimeIndexConverter())
        with pytest.raises(NotImplementedError):
            source._fingerprint

    def test_labels(self):
        source = BaseDataSourceForTesting(DatetimeIndexConverter())
        with pytest.raises(NotImplementedError):
            source.labels

    def test__get(self):
        source = BaseDataSourceForTesting(DatetimeIndexConverter())
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

        source = BaseDataSourceForTesting(DatetimeIndexConverter())
        df_out = source._sync_data(data, pd.to_timedelta("2s"))

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c,
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_data_datetimeindex2(self):
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

        source = BaseDataSourceForTesting(DatetimeIndexConverter())
        df_out = source._sync_data(data, "2s")

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c,
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

        source = BaseDataSourceForTesting(IntegerIndexConverter())
        df_out = source._sync_data(data, 4)

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c,
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_data_floatindex(self):
        index_a = np.arange(0, 1000, 0.1)
        values_a = np.random.random(len(index_a))
        series_a = pd.Series(data=values_a, index=index_a)

        noise_b = np.random.random(len(index_a)) / 100.0
        index_b = index_a + noise_b
        values_b = np.random.random(len(index_b))
        series_b = pd.Series(data=values_b, index=index_b)

        noise_c = np.random.random(len(index_a)) / 100.0
        index_c = index_a - noise_c
        values_c = np.random.random(len(index_c))
        series_c = pd.Series(data=values_c, index=index_c)

        noise_d = np.random.random(len(index_a)) / 100.0
        index_d = index_a + noise_d
        values_d = np.random.random(len(index_d))
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        source = BaseDataSourceForTesting(FloatIndexConverter())
        df_out = source._sync_data(data, 2.0 / 100.0)

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c,
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_missing_values_and_other_dtypes(self):
        index_a = [1.0, 2.0, 3.0]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.0]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.0]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        source = BaseDataSourceForTesting(FloatIndexConverter())
        df_out = source._sync_data(data, 0.2)

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0],
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_dataframe(self):
        index_a = [1.0, 2.0, 3.0]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.DataFrame(data={"a": values_a}, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.DataFrame(data={"b": values_b}, index=index_b)

        index_c = [2.09, 5.0]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.DataFrame(data={"c": values_c}, index=index_c)

        index_d = [0.95, 5.0]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.DataFrame(data={"d": values_d}, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        source = BaseDataSourceForTesting(FloatIndexConverter())
        df_out = source._sync_data(data, 0.2)

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0],
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__sync_small_tolerance_yields_no_syncing(self):
        index_a = [1.0, 2.0, 3.0]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.0]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.0]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        source = BaseDataSourceForTesting(FloatIndexConverter())
        df_out = source._sync_data(data, 0.0001)  # small tolerance yields no syncing

        df_expect = pd.DataFrame(
            data={
                "a": [
                    np.nan,
                    "value_a1",
                    np.nan,
                    np.nan,
                    2.0,
                    np.nan,
                    "value_a3",
                    np.nan,
                ],
                "b": [np.nan, np.nan, "value_b1", 1, np.nan, np.nan, np.nan, np.nan],
                "c": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "value_c1",
                    np.nan,
                    "value_c2",
                ],
                "d": [
                    "value_d1",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "value_d2",
                ],
            },
            index=[0.95, 1.0, 1.01, 1.91, 2.0, 2.09, 3.0, 5.0],
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    @patch.object(BaseDataSourceForTesting, "_get")
    def test__source_get_nosync(self, mock_get):
        index_a = [1.0, 2.0, 3.0]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.0]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.0]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        mock_get.return_value = data

        source = BaseDataSourceForTesting(DatetimeIndexConverter(), index_sync=False)
        df_out = source._source_get("<start-time>", "<end-time>")

        df_expect = pd.DataFrame(
            data={
                "a": [
                    np.nan,
                    "value_a1",
                    np.nan,
                    np.nan,
                    2.0,
                    np.nan,
                    "value_a3",
                    np.nan,
                ],
                "b": [np.nan, np.nan, "value_b1", 1, np.nan, np.nan, np.nan, np.nan],
                "c": [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "value_c1",
                    np.nan,
                    "value_c2",
                ],
                "d": [
                    "value_d1",
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "value_d2",
                ],
            },
            index=[0.95, 1.0, 1.01, 1.91, 2.0, 2.09, 3.0, 5.0],
        )

        pd.testing.assert_frame_equal(df_out, df_expect)
        mock_get.assert_called_once_with("<start-time>", "<end-time>")

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_source_get_sync(self, mock_get):
        index_a = [1.0, 2.0, 3.0]
        values_a = ["value_a1", 2.0, "value_a3"]
        series_a = pd.Series(data=values_a, index=index_a)

        index_b = [1.01, 1.91]
        values_b = ["value_b1", 1]
        series_b = pd.Series(data=values_b, index=index_b)

        index_c = [2.09, 5.0]
        values_c = ["value_c1", "value_c2"]
        series_c = pd.Series(data=values_c, index=index_c)

        index_d = [0.95, 5.0]
        values_d = ["value_d1", "value_d2"]
        series_d = pd.Series(data=values_d, index=index_d)

        data = {
            "a": series_a,
            "b": series_b,
            "c": series_c,
            "d": series_d,
        }

        mock_get.return_value = data

        source = BaseDataSourceForTesting(
            FloatIndexConverter(), index_sync=True, tolerance=0.2
        )
        df_out = source._source_get("<start-time>", "<end-time>")

        df_expect = pd.DataFrame(
            data={
                "a": ["value_a1", 2.0, "value_a3", np.nan],
                "b": ["value_b1", 1, np.nan, np.nan],
                "c": [np.nan, "value_c1", np.nan, "value_c2"],
                "d": ["value_d1", np.nan, np.nan, "value_d2"],
            },
            index=[0.95, 1.91, 3.0, 5.0],
        )

        pd.testing.assert_frame_equal(df_out, df_expect)
        mock_get.assert_called_once_with("<start-time>", "<end-time>")

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_source_get_raises_no_tolerance(self, mock_get):
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), index_sync=True, tolerance=None
        )

        with pytest.raises(ValueError):
            source._source_get("<start-time>", "<end-time>")

    @patch.object(BaseDataSourceForTesting, "_get")
    def test_source_get_sync_datetimeindex(self, mock_get):
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

        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), index_sync=True, tolerance=pd.to_timedelta("2s")
        )
        df_out = source._source_get("<start-time>", "<end-time>")

        df_expect = pd.DataFrame(
            data={
                "a": values_a,
                "b": values_b,
                "c": values_c,
                "d": values_d,
            },
            index=index_c,
        )

        pd.testing.assert_frame_equal(df_out, df_expect)

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_index_mode_start(self, mock_get):
        mock_get.side_effect = lambda start, end, **kwargs: (start, end)

        source = BaseDataSourceForTesting(DatetimeIndexConverter())

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="start", refresh_cache=False)

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == start[i]
            assert data_i == (start[i], end[i])
            mock_get.assert_called_with(start[i], end[i], refresh_cache=False)

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_index_mode_end(self, mock_get):
        mock_get.side_effect = lambda start, end, **kwargs: (start, end)

        source = BaseDataSourceForTesting(DatetimeIndexConverter())

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="end", refresh_cache=True)

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == end[i]
            assert data_i == (start[i], end[i])
            mock_get.assert_called_with(start[i], end[i], refresh_cache=True)

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_index_mode_mid(self, mock_get):
        mock_get.side_effect = lambda start, end, **kwargs: (start, end)

        source = BaseDataSourceForTesting(DatetimeIndexConverter())

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="mid")

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == start[i] + (end[i] - start[i]) / 2.0
            assert data_i == (start[i], end[i])
            mock_get.assert_called_with(start[i], end[i], refresh_cache=False)

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_raises_length(self, mock_get):
        mock_get.side_effect = lambda start, end, **kwargs: (start, end)

        source = BaseDataSourceForTesting(DatetimeIndexConverter())

        start = [1, 2, 3]
        end = [2, 3]
        with pytest.raises(ValueError):
            source.iter(start, end, index_mode="start")

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_raises_index_mode(self, mock_get):
        mock_get.side_effect = lambda start, end, **kwargs: (start, end)

        source = BaseDataSourceForTesting(DatetimeIndexConverter())

        start = [1, 2, 3]
        end = [2, 3, 4]
        with pytest.raises(ValueError):
            source.iter(start, end, index_mode="invalide-mode")

    @patch.object(BaseDataSourceForTesting, "_source_get")
    def test_get_nocache(self, mock_source_get):
        source = BaseDataSourceForTesting(DatetimeIndexConverter(), cache=None)

        out = source.get("<start>", "<end>")
        assert out == mock_source_get.return_value
        mock_source_get.assert_called_once_with("<start>", "<end>")

    @patch.object(BaseDataSourceForTesting, "_cache_source_get")
    def test_get_cache(self, mock_cache_source_get, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )

        out = source.get("<start>", "<end>")
        assert out == mock_cache_source_get.return_value
        mock_cache_source_get.assert_called_once_with(
            "<start>", "<end>", refresh_cache=False
        )

    @patch.object(BaseDataSourceForTesting, "_cache_source_get")
    def test_get_cache_refresh(self, mock_cache_source_get, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )

        out = source.get("<start>", "<end>", refresh_cache=True)
        assert out == mock_cache_source_get.return_value
        mock_cache_source_get.assert_called_once_with(
            "<start>", "<end>", refresh_cache=True
        )

    def test_partition_start_end_int(self):
        out = BaseDataSourceForTesting._partition_start_end(3, 9, 3, 2)
        expect = [(2, 5), (5, 8), (8, 11)]
        assert list(out) == expect

    def test_partition_start_end_float(self):
        out = BaseDataSourceForTesting._partition_start_end(3.0, 9.0, 3.2, 2.1)
        expect = [(2.1, 5.3), (5.3, 8.5), (8.5, 11.7)]
        np.testing.assert_almost_equal(list(out), expect)

    def test_partition_start_end_datetime(self):
        start = pd.to_datetime("2020-01-01 03:00", utc=True)
        end = pd.to_datetime("2020-01-01 09:00", utc=True)
        partition = pd.to_timedelta("3H")
        reference = pd.to_datetime("2020-01-01 02:00", utc=True)
        out = BaseDataSourceForTesting._partition_start_end(
            start, end, partition, reference
        )
        expect = [
            (
                pd.to_datetime("2020-01-01 02:00", utc=True),
                pd.to_datetime("2020-01-01 05:00", utc=True),
            ),
            (
                pd.to_datetime("2020-01-01 05:00", utc=True),
                pd.to_datetime("2020-01-01 08:00", utc=True),
            ),
            (
                pd.to_datetime("2020-01-01 08:00", utc=True),
                pd.to_datetime("2020-01-01 11:00", utc=True),
            ),
        ]
        assert list(out) == expect

    def test__is_cached_false(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )
        assert cache_dir.exists()
        assert source._is_cached("filename") is False

    def test__is_cached_true(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )
        assert cache_dir.exists()
        (cache_dir / "filename").touch()
        assert source._is_cached("filename") is True

    def test__cache_read(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )

        df = pd.DataFrame(data={"filename": [2, 4, 6], "a": [1, 2, 3]})
        df.to_feather(cache_dir / "filename")

        df_out = source._cache_read("filename")
        df_expect = pd.DataFrame(data={"a": [1, 2, 3]}, index=[2, 4, 6])
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test__cache_write(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        source = BaseDataSourceForTesting(
            DatetimeIndexConverter(), cache=cache_dir, cache_size="1H"
        )

        df = pd.DataFrame(data={"a": [1, 2, 3]}, index=[2, 4, 6])
        source._cache_write("filename", df)

        df_out = pd.read_feather(cache_dir / "filename")
        df_expect = pd.DataFrame(data={"filename": [2, 4, 6], "a": [1, 2, 3]})
        pd.testing.assert_frame_equal(df_out, df_expect)

    @patch.object(BaseDataSourceForTesting, "_get")
    def test__cache_source_get(self, mock_get, tmp_path):
        def _get_side_effect(start, end):
            index = np.arange(start, end + 1, dtype="int64")
            values_a = [1.0] * len(index)
            values_b = [2.1] * len(index)
            df = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)
            return df

        mock_get.side_effect = _get_side_effect

        class DataSource(BaseDataSourceForTesting):
            @property
            def _fingerprint(self):
                return "1234"

        cache_dir = tmp_path / ".cache"
        source = DataSource(IntegerIndexConverter(), cache=cache_dir, cache_size=10)

        out_source = source._cache_source_get(5, 105, refresh_cache=False)
        out_memory_cache = source._cache_source_get(5, 105, refresh_cache=False)
        source = DataSource(IntegerIndexConverter(), cache=cache_dir, cache_size=10)
        out_file_cache = source._cache_source_get(5, 105, refresh_cache=False)

        index = np.arange(5, 105 + 1, dtype="int64")
        values_a = [1.0] * len(index)
        values_b = [2.1] * len(index)
        expect = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)

        pd.testing.assert_frame_equal(out_source, expect)
        pd.testing.assert_frame_equal(out_memory_cache, expect)
        pd.testing.assert_frame_equal(out_file_cache, expect)

        mock_get.assert_has_calls(
            [
                call(0, 10),
                call(10, 20),
                call(20, 30),
                call(30, 40),
                call(40, 50),
                call(50, 60),
                call(60, 70),
                call(70, 80),
                call(80, 90),
                call(90, 100),
                call(100, 110),
            ]
        )

        # memory cache
        assert len(source._memory_cache) == 11
        pd.testing.assert_frame_equal(
            pd.concat(source._memory_cache.values()).sort_index().loc[5:105], expect
        )

        # file cache
        df_list = []
        for fname in os.listdir(cache_dir):
            df_i = source._cache_read(fname)
            df_list.append(df_i)

        assert len(os.listdir(cache_dir)) == 11
        pd.testing.assert_frame_equal(
            pd.concat(df_list).sort_index().loc[5:105], expect
        )

    @patch.object(BaseDataSourceForTesting, "_get")
    def test__cache_source_get_refresh(self, mock_get, tmp_path):
        def _get_side_effect(start, end):
            index = np.arange(start, end + 1, dtype="int64")
            values_a = [1.0] * len(index)
            values_b = [2.1] * len(index)
            df = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)
            return df

        mock_get.side_effect = _get_side_effect

        class DataSource(BaseDataSourceForTesting):
            @property
            def _fingerprint(self):
                return "1234"

        cache_dir = tmp_path / ".cache"
        source = DataSource(IntegerIndexConverter(), cache=cache_dir, cache_size=10)

        source._cache_source_get(5, 105, refresh_cache=True)
        source._cache_source_get(5, 105, refresh_cache=True)

        assert len(source._memory_cache) == 11
        assert len(os.listdir(cache_dir)) == 11

        mock_get.assert_has_calls(
            [
                call(0, 10),
                call(10, 20),
                call(20, 30),
                call(30, 40),
                call(40, 50),
                call(50, 60),
                call(60, 70),
                call(70, 80),
                call(80, 90),
                call(90, 100),
                call(100, 110),
                call(0, 10),
                call(10, 20),
                call(20, 30),
                call(30, 40),
                call(40, 50),
                call(50, 60),
                call(60, 70),
                call(70, 80),
                call(80, 90),
                call(90, 100),
                call(100, 110),
            ]
        )

    @patch.object(BaseDataSourceForTesting, "_get")
    def test__cache_source_get_2(self, mock_get, tmp_path):
        def _get_side_effect(start, end):
            index = np.arange(start, end + 1, dtype="int64")
            values_a = [1.0] * len(index)
            values_b = [2.1] * len(index)
            df = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)
            return df

        mock_get.side_effect = _get_side_effect

        class DataSource(BaseDataSourceForTesting):
            @property
            def _fingerprint(self):
                return "1234"

        cache_dir = tmp_path / ".cache"
        source_a = DataSource(IntegerIndexConverter(), cache=cache_dir, cache_size=10)
        source_b = DataSource(IntegerIndexConverter(), cache=cache_dir, cache_size=10)

        out_a = source_a._cache_source_get(5, 45, refresh_cache=True)
        out_b = source_b._cache_source_get(5, 105, refresh_cache=True)

        mock_get.assert_has_calls(
            [
                call(0, 10),
                call(10, 20),
                call(20, 30),
                call(30, 40),
                call(40, 50),
                call(50, 60),
                call(60, 70),
                call(70, 80),
                call(80, 90),
                call(90, 100),
                call(100, 110),
            ]
        )

        index = np.arange(5, 45 + 1, dtype="int64")
        values_a = [1.0] * len(index)
        values_b = [2.1] * len(index)
        expect_a = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)

        index = np.arange(5, 105 + 1, dtype="int64")
        values_a = [1.0] * len(index)
        values_b = [2.1] * len(index)
        expect_b = pd.DataFrame(data={"a": values_a, "b": values_b}, index=index)

        pd.testing.assert_frame_equal(out_a, expect_a)
        pd.testing.assert_frame_equal(out_b, expect_b)


class Test_DrioDataSource:
    def test__init__(self, tmp_path):
        cache_dir = tmp_path / ".cache"
        drio_client = Mock()
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(
            drio_client,
            labels,
            index_type="datetime",
            index_sync=True,
            tolerance=1,
            cache=cache_dir,
            cache_size=None,
            convert_date=False,
            raise_empty=True,
        )

        assert isinstance(source, BaseDataSource)
        assert source._drio_client == drio_client
        assert source._labels == labels
        assert source._index_sync is True
        assert source._tolerance == 1
        assert source._get_kwargs == {"convert_date": False, "raise_empty": True}
        assert isinstance(source._index_converter, DatetimeIndexConverter)
        assert source._cache == Path(cache_dir)
        assert source._cache_size == "24H"

    def test__init__integer(self):
        drio_client = Mock()
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(
            drio_client,
            labels,
            index_type="integer",
        )

        assert isinstance(source._index_converter, IntegerIndexConverter)
        assert source._cache_size == 8.64e13

    def test__init__index_type_raises(self):
        drio_client = Mock()
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }

        with pytest.raises(ValueError):
            DrioDataSource(
                drio_client,
                labels,
                index_type="invalid index type",
            )

    def test__fingerprint(self):
        drio_client = Mock()
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(
            drio_client,
            labels,
            index_type="integer",
            index_sync=False,
            tolerance=1,
            cache=None,
            cache_size=None,
            convert_date=False,
            raise_empty=True,
        )
        out = source._fingerprint
        get_kwargs = {"convert_date": False, "raise_empty": True}
        expect = md5(
            f"{labels}_{get_kwargs}_IntegerIndexConverter_False_1".encode()
        ).hexdigest()
        assert out == expect

    def test__labels(self):
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(Mock(), labels)
        assert set(source.labels) == set(["a", "b", "c"])

    def test__get(self):
        drio_client = Mock()
        drio_client.get.return_value = pd.Series(
            data=[1.1, 1.2, 1.3], index=[1.0, 2.0, 3.0]
        )

        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(
            drio_client, labels, convert_date=True, raise_empty=False
        )

        data_out = source._get("<start-time>", "<end-time>")

        data_expect = {
            "a": pd.Series(data=[1.1, 1.2, 1.3], index=[1.0, 2.0, 3.0]),
            "b": pd.Series(data=[1.1, 1.2, 1.3], index=[1.0, 2.0, 3.0]),
            "c": pd.Series(data=[1.1, 1.2, 1.3], index=[1.0, 2.0, 3.0]),
        }

        assert data_out.keys() == data_expect.keys()
        for key, series_expect in data_expect.items():
            pd.testing.assert_series_equal(series_expect, data_out[key])

        drio_client.get.assert_has_calls(
            [
                call(
                    "timeseriesid-a",
                    start="<start-time>",
                    end="<end-time>",
                    convert_date=True,
                    raise_empty=False,
                ),
                call(
                    "timeseriesid-b",
                    start="<start-time>",
                    end="<end-time>",
                    convert_date=True,
                    raise_empty=False,
                ),
                call(
                    "timeseriesid-c",
                    start="<start-time>",
                    end="<end-time>",
                    convert_date=True,
                    raise_empty=False,
                ),
            ]
        )


class Test_NullDataSource:
    def test__init__(self):
        source = _NullDataSource(DatetimeIndexConverter(), labels=["a", "b", "c"])
        assert isinstance(source, BaseDataSource)
        assert source._labels == ("a", "b", "c")
        assert isinstance(source._index_converter, DatetimeIndexConverter)
        assert source._index_sync is False

    def test_labels(self):
        source = _NullDataSource(DatetimeIndexConverter(), labels=["a", "b", "c"])
        assert source.labels == ("a", "b", "c")

    def test_labels_none(self):
        source = _NullDataSource(DatetimeIndexConverter(), labels=None)
        assert source.labels == ()

    def test__fingerprint(self):
        source = _NullDataSource(DatetimeIndexConverter(), labels=["a", "b", "c"])
        out = source._fingerprint
        expect = md5(
            ("DatetimeIndexConverter_" + str(("a", "b", "c"))).encode()
        ).hexdigest()
        assert out == expect

    def test__get(self):
        source = _NullDataSource(DatetimeIndexConverter(), labels=["a", "b", "c"])
        data_out = source._get("2020-01-01 00:00", "2021-01-01 00:00")

        data_expect = {
            "a": pd.Series([], dtype="object"),
            "b": pd.Series([], dtype="object"),
            "c": pd.Series([], dtype="object"),
        }

        pd.testing.assert_frame_equal(pd.DataFrame(data_out), pd.DataFrame(data_expect))


class Test_CompositeDataSource:
    def test__init__(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = _NullDataSource(DatetimeIndexConverter(), labels=["C", "A", "B"])
        index_source = [
            ("1999-01-01 00:00", None),
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]
        source = CompositeDataSource(index_source)

        assert isinstance(source, BaseDataSource)
        assert source._labels == ("A", "B", "C")
        assert source._index_converter == DatetimeIndexConverter()
        np.testing.assert_array_equal(
            list(source._index.values()),
            pd.to_datetime(
                [
                    "1999-01-01 00:00",
                    "2020-01-01 00:00",
                    "2020-01-01 02:00",
                    "2020-01-01 04:00",
                    "2020-01-01 06:00",
                ],
                utc=True,
            ),
        )
        np.testing.assert_array_equal(
            list(source._index.keys()),
            [
                "1999-01-01 00:00",
                "2020-01-01 00:00",
                "2020-01-01 02:00",
                "2020-01-01 04:00",
                "2020-01-01 06:00",
            ],
        )
        assert isinstance(source._sources[0], _NullDataSource)
        assert isinstance(source._sources[1], DrioDataSource)
        assert isinstance(source._sources[2], _NullDataSource)
        assert isinstance(source._sources[3], DrioDataSource)
        assert isinstance(source._sources[4], _NullDataSource)

    def test__init__raise_wrong_order(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = _NullDataSource(DatetimeIndexConverter(), labels=["C", "A", "B"])
        index_source = [
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("1999-01-01 00:00", None),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]

        with pytest.raises(ValueError):
            CompositeDataSource(index_source)

    def test__init__raises_labels(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = _NullDataSource(DatetimeIndexConverter(), labels=["D", "A", "B"])
        index_source = [
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]
        with pytest.raises(ValueError):
            CompositeDataSource(index_source)

    def test__init__raises_index_converter(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = _NullDataSource(IntegerIndexConverter(), labels=["C", "A", "B"])
        index_source = [
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]
        with pytest.raises(ValueError):
            CompositeDataSource(index_source)

    def test_labels(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = _NullDataSource(DatetimeIndexConverter(), labels=["C", "A", "B"])
        index_source = [
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]
        source = CompositeDataSource(index_source)
        assert source.labels == ("A", "B", "C")

    def test_get(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        source1 = DrioDataSource(drio_client, labels, index_type="datetime")
        source2 = _NullDataSource(DatetimeIndexConverter(), labels=labels.keys())
        source3 = DrioDataSource(drio_client, labels, index_type="datetime")

        # Ugly formatting by 'black'.
        # Use 'with' statement with () when Python 3.9 becomes default for testing.
        with patch.object(source1, "_get") as mock_get1, patch.object(
            source3, "_get"
        ) as mock_get3:
            mock_idx1 = pd.DatetimeIndex(
                ["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:02"],
                tz="utc",
            )
            mock_get1.return_value = {
                "A": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "B": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "C": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
            }

            mock_idx3 = pd.DatetimeIndex(
                ["2020-01-01 04:00", "2020-01-01 04:01", "2020-01-01 04:02"],
                tz="utc",
            )
            mock_get3.return_value = {
                "A": pd.Series([1.0, 2.0, 3.0], index=mock_idx3),
                "B": pd.Series([1.0, 2.0, 3.0], index=mock_idx3),
                "C": pd.Series([1.0, 2.0, 3.0], index=mock_idx3),
            }

            index_source = [
                ("2020-01-01 00:00", source1),
                ("2020-01-01 02:00", source2),
                ("2020-01-01 04:00", source3),
                ("2020-01-01 06:00", None),
            ]
            source = CompositeDataSource(index_source)

            data_out = source.get("2019-01-01 00:00", "2021-01-01 00:00")

            idx_expect = pd.DatetimeIndex(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 00:01",
                    "2020-01-01 00:02",
                    "2020-01-01 04:00",
                    "2020-01-01 04:01",
                    "2020-01-01 04:02",
                ],
                tz="utc",
            )
            data_expect = {
                "A": pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=idx_expect),
                "B": pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=idx_expect),
                "C": pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], index=idx_expect),
            }

            pd.testing.assert_frame_equal(data_out, pd.DataFrame(data_expect))

            mock_get1.assert_called_once_with("2020-01-01 00:00", "2020-01-01 02:00")
            mock_get3.assert_called_once_with("2020-01-01 04:00", "2020-01-01 06:00")

    def test_get_single_source(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        source1 = DrioDataSource(drio_client, labels, index_type="datetime")

        with patch.object(source1, "_get") as mock_get1:
            mock_idx1 = pd.DatetimeIndex(
                ["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:02"],
                tz="utc",
            )
            mock_get1.return_value = {
                "A": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "B": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "C": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
            }

            index_source = [
                ("2020-01-01 00:00", source1),
            ]
            source = CompositeDataSource(index_source)

            data_out = source.get("2019-01-01 00:00", "2021-01-01 00:00")

            idx_expect = pd.DatetimeIndex(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 00:01",
                    "2020-01-01 00:02",
                ],
                tz="utc",
            )
            data_expect = {
                "A": pd.Series([1.0, 2.0, 3.0], index=idx_expect),
                "B": pd.Series([1.0, 2.0, 3.0], index=idx_expect),
                "C": pd.Series([1.0, 2.0, 3.0], index=idx_expect),
            }

            pd.testing.assert_frame_equal(data_out, pd.DataFrame(data_expect))

            mock_get1.assert_called_once_with("2020-01-01 00:00", "2021-01-01 00:00")

    def test_get_out_of_range(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        source1 = DrioDataSource(drio_client, labels, index_type="datetime")

        with patch.object(source1, "_get") as mock_get1:
            mock_idx1 = pd.DatetimeIndex(
                ["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:02"],
                tz="utc",
            )
            mock_get1.return_value = {
                "A": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "B": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
                "C": pd.Series([1.0, 2.0, 3.0], index=mock_idx1),
            }

            index_source = [("2020-01-01 00:00", source1), ("2021-01-01 00:00", None)]
            source = CompositeDataSource(index_source)

            data_out_left = source.get("2018-01-01 00:00", "2019-01-01 00:00")
            data_out_right = source.get("2021-01-01 00:00", "2022-01-01 00:00")

            data_expect = pd.DataFrame(
                {
                    "A": pd.Series([], dtype="object"),
                    "B": pd.Series([], dtype="object"),
                    "C": pd.Series([], dtype="object"),
                }
            )
            pd.testing.assert_frame_equal(data_out_left, data_expect)
            pd.testing.assert_frame_equal(data_out_right, data_expect)

    @patch.object(_NullDataSource, "_get")
    def test_get_integer(self, mock_get):
        mock_get.return_value = {
            "A": pd.Series([], dtype="object"),
            "B": pd.Series([], dtype="object"),
            "C": pd.Series([], dtype="object"),
        }
        index_source = [
            (10, _NullDataSource(IntegerIndexConverter(), labels=["A", "B", "C"])),
            (20, _NullDataSource(IntegerIndexConverter(), labels=["A", "B", "C"])),
            (30, _NullDataSource(IntegerIndexConverter(), labels=["A", "B", "C"])),
            (40, _NullDataSource(IntegerIndexConverter(), labels=["A", "B", "C"])),
        ]
        source = CompositeDataSource(index_source)

        data_out = source.get(0, 100)

        data_expect = {
            "A": pd.Series([], dtype="object"),
            "B": pd.Series([], dtype="object"),
            "C": pd.Series([], dtype="object"),
        }

        pd.testing.assert_frame_equal(data_out, pd.DataFrame(data_expect))
        mock_get.assert_has_calls(
            [call(0, 10), call(10, 20), call(20, 30), call(30, 40), call(40, 100)]
        )


class Test_DatetimeIndexConverter:
    def test_to_universal_index(self):
        out = DatetimeIndexConverter().to_universal_index("2020-01-01 00:00")
        expect = pd.to_datetime("2020-01-01 00:00", utc=True)
        assert out == expect

    def test_to_universal_index_arraylike(self):
        out = DatetimeIndexConverter().to_universal_index(
            [
                "2020-01-01 00:00",
                "2020-01-01 01:00",
                pd.to_datetime("2020-01-01 02:00").tz_localize(tz="Europe/Stockholm"),
            ]
        )
        expect = pd.to_datetime(
            [
                "2020-01-01 00:00",
                "2020-01-01 01:00",
                pd.to_datetime("2020-01-01 02:00").tz_localize(tz="Europe/Stockholm"),
            ],
            utc=True,
        )
        np.testing.assert_array_equal(out, expect)

    def test_to_universal_delta(self):
        out = DatetimeIndexConverter().to_universal_delta("3H")
        expect = pd.to_timedelta("3H")
        assert out == expect

    def test_to_universal_delta_arraylike(self):
        out = DatetimeIndexConverter().to_universal_delta(
            ["3H", "24H", pd.to_timedelta("2D")]
        )
        expect = pd.to_timedelta(["3H", "24H", pd.to_timedelta("2D")])
        np.testing.assert_array_equal(out, expect)

    def test_to_native_index(self):
        out = DatetimeIndexConverter().to_native_index(
            DatetimeIndexConverter().to_universal_index("2020-01-01 00:00")
        )
        expect = pd.to_datetime("2020-01-01 00:00", utc=True)
        assert out == expect

    def test_reference(self):
        out = DatetimeIndexConverter().reference
        expect = pd.to_datetime(0, utc=True)
        assert out == expect

    def test__repr__(self):
        out = str(DatetimeIndexConverter())
        expect = "DatetimeIndexConverter"
        assert out == expect


class Test_IntegerIndexConverter:
    def test_to_universal_index_int(self):
        out = IntegerIndexConverter().to_universal_index(3)
        expect = 3
        assert out == expect

    def test_to_universal_index_arraylike(self):
        out = IntegerIndexConverter().to_universal_index([2, 4.0, "3"])
        expect = [2, 4, 3]
        np.testing.assert_array_equal(out, expect)

    def test_to_universal_delta(self):
        out = IntegerIndexConverter().to_universal_delta(3)
        expect = 3
        assert out == expect

    def test_to_universal_delta_arraylike(self):
        out = IntegerIndexConverter().to_universal_delta([2, 4.0, "3"])
        expect = [2, 4, 3]
        np.testing.assert_array_equal(out, expect)

    def test_to_native_index(self):
        out = IntegerIndexConverter().to_native_index(
            IntegerIndexConverter().to_universal_index(3)
        )
        expect = 3
        assert out == expect

    def test_reference(self):
        out = IntegerIndexConverter().reference
        expect = 0
        assert out == expect

    def test__repr__(self):
        out = str(IntegerIndexConverter())
        expect = "IntegerIndexConverter"
        assert out == expect


class Test_FloatIndexConverter:
    def test_to_universal_index_int(self):
        out = FloatIndexConverter().to_universal_index(3)
        expect = 3.0
        assert out == pytest.approx(expect)

    def test_to_universal_index_arraylike(self):
        out = FloatIndexConverter().to_universal_index([2, 4.0, "3"])
        expect = [2.0, 4.0, 3.0]
        np.testing.assert_array_almost_equal(out, expect)

    def test_to_universal_delta(self):
        out = FloatIndexConverter().to_universal_delta(3)
        expect = 3.0
        assert out == pytest.approx(expect)

    def test_to_universal_delta_arraylike(self):
        out = FloatIndexConverter().to_universal_delta([2, 4.0, "3"])
        expect = [2.0, 4.0, 3.0]
        np.testing.assert_array_almost_equal(out, expect)

    def test_to_native_index(self):
        out = FloatIndexConverter().to_native_index(
            IntegerIndexConverter().to_universal_index(3)
        )
        expect = 3.0
        assert out == pytest.approx(expect)

    def test_reference(self):
        out = FloatIndexConverter().reference
        expect = 0.0
        assert out == pytest.approx(expect)

    def test__repr__(self):
        out = str(FloatIndexConverter())
        expect = "FloatIndexConverter"
        assert out == expect
