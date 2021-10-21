import types
from unittest.mock import Base, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from fourinsight.engineroom.utils import (
    CompositeDataSource,
    DrioDataSource,
    NullDataSource,
)
from fourinsight.engineroom.utils.datamanage import BaseDataSource


class BaseDataSourceForTesting(BaseDataSource):
    @property
    def labels(self):
        super().labels

    def _get(self, start, end):
        super()._get(start, end)


class Test_BaseDataSource:
    def test__init__(self):
        source = BaseDataSourceForTesting("datetime", index_sync=True, tolerance=1)
        assert source._index_type == "datetime"
        assert source._index_sync is True
        assert source._tolerance == 1

    def test_labels(self):
        source = BaseDataSourceForTesting("datetime")
        with pytest.raises(NotImplementedError):
            source.labels

    def test__get(self):
        source = BaseDataSourceForTesting("datetime")
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

        df_out = BaseDataSource._sync_data(data, 4)

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

        df_out = BaseDataSource._sync_data(data, 2.0 / 100.0)

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

        df_out = BaseDataSource._sync_data(data, 0.2)

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

        df_out = BaseDataSource._sync_data(data, 0.2)

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

        df_out = BaseDataSource._sync_data(
            data, 0.0001
        )  # small tolerance yields no syncing

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
    def test_get_nosync(self, mock_get):
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

        source = BaseDataSourceForTesting("datetime", index_sync=False)
        df_out = source.get("<start-time>", "<end-time>")

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
    def test_get_sync(self, mock_get):
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

        source = BaseDataSourceForTesting("datetime", index_sync=True, tolerance=0.2)
        df_out = source.get("<start-time>", "<end-time>")

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
    def test_get_raises_no_tolerance(self, mock_get):
        source = BaseDataSourceForTesting("datetime", index_sync=True, tolerance=None)

        with pytest.raises(ValueError):
            source.get("<start-time>", "<end-time>")

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

        source = BaseDataSourceForTesting(
            "datetime", index_sync=True, tolerance=pd.to_timedelta("2s")
        )
        df_out = source.get("<start-time>", "<end-time>")

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
        mock_get.side_effect = lambda start, end: (start, end)

        source = BaseDataSourceForTesting("datetime")

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="start")

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == start[i]
            assert data_i == (start[i], end[i])

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_index_mode_end(self, mock_get):
        mock_get.side_effect = lambda start, end: (start, end)

        source = BaseDataSourceForTesting("datetime")

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="end")

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == end[i]
            assert data_i == (start[i], end[i])

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_index_mode_mid(self, mock_get):
        mock_get.side_effect = lambda start, end: (start, end)

        source = BaseDataSourceForTesting("datetime")

        start = [1, 2, 3]
        end = [2, 3, 4]
        data_iter = source.iter(start, end, index_mode="mid")

        assert isinstance(data_iter, types.GeneratorType)
        for i, (index_i, data_i) in enumerate(data_iter):
            assert index_i == start[i] + (end[i] - start[i]) / 2.0
            assert data_i == (start[i], end[i])

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_raises_length(self, mock_get):
        mock_get.side_effect = lambda start, end: (start, end)

        source = BaseDataSourceForTesting("datetime")

        start = [1, 2, 3]
        end = [2, 3]
        with pytest.raises(ValueError):
            source.iter(start, end, index_mode="start")

    @patch.object(BaseDataSourceForTesting, "get")
    def test_iter_raises_index_mode(self, mock_get):
        mock_get.side_effect = lambda start, end: (start, end)

        source = BaseDataSourceForTesting("datetime")

        start = [1, 2, 3]
        end = [2, 3, 4]
        with pytest.raises(ValueError):
            source.iter(start, end, index_mode="invalide-mode")

    def test__index_universal_callable(self):
        index_type = Mock()
        source = BaseDataSourceForTesting(index_type)

        assert source._index_universal("2020-01-01 00:00") == index_type.return_value
        index_type.assert_called_once_with("2020-01-01 00:00")

    def test__index_universal_datetime(self):
        source = BaseDataSourceForTesting("datetime")

        index_out = source._index_universal("2020-01-01 00:00")
        index_expect = 1577836800000000000
        assert index_out == index_expect

    def test__index_universal_datetime_list(self):
        source = BaseDataSourceForTesting("datetime")

        index_out = source._index_universal(
            ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
        )
        index_expect = np.array(
            [1577836800000000000, 1577840400000000000, 1577844000000000000]
        )
        np.testing.assert_array_equal(index_out, index_expect)

    def test__index_universal_integer(self):
        source = BaseDataSourceForTesting("integer")

        index_out = source._index_universal(1)
        index_expect = 1
        assert index_out == index_expect

    def test__index_universal_integer_list(self):
        source = BaseDataSourceForTesting("integer")

        index_out = source._index_universal([1, 2, 3])
        index_expect = np.array([1, 2, 3])
        np.testing.assert_array_equal(index_out, index_expect)


class Test_DrioDataSource:
    def test__init__(self):
        drio_client = Mock()
        labels = {
            "a": "timeseriesid-a",
            "b": "timeseriesid-b",
            "c": "timeseriesid-c",
        }
        source = DrioDataSource(
            drio_client,
            labels,
            index_sync=True,
            tolerance=1,
            convert_date=False,
            raise_empty=True,
        )

        assert isinstance(source, BaseDataSource)
        assert source._drio_client == drio_client
        assert source._labels == labels
        assert source._index_sync is True
        assert source._tolerance == 1
        assert source._get_kwargs == {"convert_date": False, "raise_empty": True}

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
        source = NullDataSource(labels=["a", "b", "c"], index_type="datetime")
        assert isinstance(source, BaseDataSource)
        assert source._labels == ("a", "b", "c")
        assert source._index_type == "datetime"
        assert source._index_sync is False

    def test_labels(self):
        source = NullDataSource(labels=["a", "b", "c"])
        assert source.labels == ("a", "b", "c")

    def test_labels_none(self):
        source = NullDataSource(labels=None)
        assert source.labels == ()

    def test__get(self):
        source = NullDataSource(labels=["a", "b", "c"])
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
        null_source = NullDataSource(labels=["C", "A", "B"], index_type="datetime")
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
        assert source._index_type == "datetime"
        np.testing.assert_array_equal(
            source._index_attached,
            pd.to_datetime(
                [
                    "1999-01-01 00:00",
                    "2020-01-01 00:00",
                    "2020-01-01 02:00",
                    "2020-01-01 04:00",
                    "2020-01-01 06:00",
                ],
                utc=True
            ).values.astype("int64"),
        )
        assert isinstance(source._sources[0], NullDataSource)
        assert isinstance(source._sources[1], DrioDataSource)
        assert isinstance(source._sources[2], NullDataSource)
        assert isinstance(source._sources[3], DrioDataSource)
        assert isinstance(source._sources[4], NullDataSource)

    def test__init__raise_wrong_order(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = NullDataSource(labels=["C", "A", "B"], index_type="datetime")
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
        null_source = NullDataSource(labels=["D", "A", "B"], index_type="datetime")
        index_source = [
            ("2020-01-01 00:00", drio_source),
            ("2020-01-01 02:00", null_source),
            ("2020-01-01 04:00", drio_source),
            ("2020-01-01 06:00", None),
        ]
        with pytest.raises(ValueError):
            CompositeDataSource(index_source)

    def test__init__raises_index_type(self):
        drio_client = Mock()
        labels = {
            "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
            "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2",
            "C": "d40fcb53-cce8-4f1a-9772-c5640db29c18",
        }
        drio_source = DrioDataSource(drio_client, labels, index_type="datetime")
        null_source = NullDataSource(labels=["C", "A", "B"], index_type="integer")
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
        null_source = NullDataSource(labels=["C", "A", "B"], index_type="datetime")
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
        source2 = NullDataSource(labels=labels.keys(), index_type="datetime")
        source3 = DrioDataSource(drio_client, labels, index_type="datetime")

# multiline with statement with parantheses supported only in Python 3.9 or newer.
# Refactor when possible.
# fmt: off
        with patch.object(source1, "_get") as mock_get1, patch.object(source3, "_get") as mock_get3:
# fmt: on
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

            mock_get1.assert_called_once_with(*pd.to_datetime(["2020-01-01 00:00", "2020-01-01 02:00"], utc=True).values.astype("int64"))
            mock_get3.assert_called_once_with(*pd.to_datetime(["2020-01-01 04:00", "2020-01-01 06:00"], utc=True).values.astype("int64"))

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

            mock_get1.assert_called_once_with(*pd.to_datetime(["2020-01-01 00:00", "2021-01-01 00:00"], utc=True).values.astype("int64"))

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

            index_source = [
                ("2020-01-01 00:00", source1),
            ]
            source = CompositeDataSource(index_source)

            data_out_left = source.get("2018-01-01 00:00", "2019-01-01 00:00")
            data_out_right = source.get("2021-01-01 00:00", "2022-01-01 00:00")

            data_expect = pd.DataFrame(
                {
                    "A": pd.Series([], dtype="object"),
                    "B": pd.Series([], dtype="object"),
                    "C": pd.Series([], dtype="object")
                }
            )
            pd.testing.assert_frame_equal(data_out_left, data_expect)
            pd.testing.assert_frame_equal(data_out_right, data_expect)

    @patch.object(NullDataSource, "_get")
    def test_get_integer(self, mock_get):
        mock_get.return_value = {
            "A": pd.Series([], dtype="object"),
            "B": pd.Series([], dtype="object"),
            "C": pd.Series([], dtype="object"),
        }
        index_source = [
            (10, NullDataSource(labels=["A", "B", "C"], index_type="integer")),
            (20, NullDataSource(labels=["A", "B", "C"], index_type="integer")),
            (30, NullDataSource(labels=["A", "B", "C"], index_type="integer")),
            (40, NullDataSource(labels=["A", "B", "C"], index_type="integer")),
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
