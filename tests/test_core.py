import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np
from pandas.core.indexes.numeric import Int64Index
import pytest
from azure.core.exceptions import ResourceNotFoundError

from fourinsight.engineroom.utils.core import NullHandler
from fourinsight.engineroom.utils import (
    AzureBlobHandler,
    LocalFileHandler,
    PersistentJSON,
    ResultCollector,
)
from fourinsight.engineroom.utils.core import BaseHandler

REMOTE_FILE_PATH = Path(__file__).parent / "testdata/a_test_file.json"


@pytest.fixture
def local_file_handler_empty(tmp_path):
    return LocalFileHandler(tmp_path / "test.json")


@pytest.fixture
def local_file_handler_w_content():
    return LocalFileHandler(REMOTE_FILE_PATH)


@pytest.fixture
def persistent_json(local_file_handler_empty):
    return PersistentJSON(local_file_handler_empty)


@pytest.fixture
@patch("fourinsight.engineroom.utils.core.BlobClient.from_connection_string")
def azure_blob_handler_mocked(mock_from_connection_string):
    connection_string = "some_connection_string"
    container_name = "some_container_name"
    blob_name = "some_blob_name"
    handler = AzureBlobHandler(connection_string, container_name, blob_name)

    remote_content = open(REMOTE_FILE_PATH, mode="r").read()
    handler._blob_client.download_blob.return_value.readall.return_value = (
        remote_content
    )

    mock_from_connection_string.assert_called_once_with(
        "some_connection_string", "some_container_name", "some_blob_name"
    )

    return handler


class Test_LocalFileHandler:
    def test__init__(self):
        handler = LocalFileHandler("./some/path")
        assert handler._path == Path("./some/path")
        assert isinstance(handler, BaseHandler)

    def test_pull(self, local_file_handler_w_content):
        handler = local_file_handler_w_content
        text_out = handler.pull()
        text_expect = (
            '{\n    "this": 1,\n    "is": "hei",\n    "a": null,\n    "test": 1.2\n}'
        )
        assert text_out == text_expect

    def test_pull_non_existing(self):
        handler = LocalFileHandler("non-existing-file")
        assert handler.pull(raise_on_missing=False) is None

    def test_pull_non_existing_raises(self):
        handler = LocalFileHandler("non-existing-file")
        with pytest.raises(FileNotFoundError):
            handler.pull(raise_on_missing=True)

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")

        content = "Some random content\n"
        handler.push(content)

        assert open(tmp_path / "test.json", mode="r").read() == content


class Test_AzureBlobHandler:
    @patch("fourinsight.engineroom.utils.core.BlobClient.from_connection_string")
    def test__init__(self, mock_from_connection_string):
        blob_handler = AzureBlobHandler(
            "some_connection_string", "some_container_name", "some_blob_name"
        )

        assert blob_handler._conn_str == "some_connection_string"
        assert blob_handler._container_name == "some_container_name"
        assert blob_handler._blob_name == "some_blob_name"
        mock_from_connection_string.assert_called_once_with(
            "some_connection_string", "some_container_name", "some_blob_name"
        )

    def test__init__fixture(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked
        assert handler._conn_str == "some_connection_string"
        assert handler._container_name == "some_container_name"
        assert handler._blob_name == "some_blob_name"
        assert isinstance(handler._blob_client, Mock)

    def test_pull(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked
        assert (
            handler.pull()
            == '{\n    "this": 1,\n    "is": "hei",\n    "a": null,\n    "test": 1.2\n}'
        )
        handler._blob_client.download_blob.assert_called_once_with(encoding="utf-8")

    def test_pull_non_existing(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        def raise_resource_not_found(*args, **kwargs):
            raise ResourceNotFoundError

        handler._blob_client.download_blob.side_effect = raise_resource_not_found

        assert handler.pull(raise_on_missing=False) is None

    def test_pull_non_existing_raises(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        def raise_resource_not_found(*args, **kwargs):
            raise ResourceNotFoundError

        handler._blob_client.download_blob.side_effect = raise_resource_not_found

        with pytest.raises(ResourceNotFoundError):
            handler.pull(raise_on_missing=True)

    def test_push(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        content = "some random content"
        handler.push(content)

        handler._blob_client.upload_blob.assert_called_once_with(
            content, overwrite=True
        )


class Test_PersistentJSON:
    def test__init__(self, local_file_handler_empty):
        handler = local_file_handler_empty
        persistent_json = PersistentJSON(handler)
        assert persistent_json._PersistentJSON__dict == {}
        assert persistent_json._handler == handler

    def test__init__raises(self):
        with pytest.raises(TypeError):
            PersistentJSON(None)

    def test__repr__(self, persistent_json):
        assert str(persistent_json) == "PersistentJSON {}"
        persistent_json.update({"a": 1.0, "b": "test"})
        assert str(persistent_json) == "PersistentJSON {'a': 1.0, 'b': 'test'}"

    def test__delitem__(self, persistent_json):
        persistent_json.update({"a": 1.0, "b": "test"})
        del persistent_json["a"]
        persistent_json._PersistentJSON__dict == {"b": "test"}

    def test__getitem__(self, persistent_json):
        persistent_json.update({"a": 1.0, "b": "test"})
        assert persistent_json["a"] == 1.0
        assert persistent_json["b"] == "test"

        with pytest.raises(KeyError):
            persistent_json["non-existing-key"]

    def test__setitem__(self, persistent_json):
        persistent_json["a"] = "some value"
        persistent_json["b"] = "some other value"

        assert persistent_json["a"] == "some value"
        assert persistent_json["b"] == "some other value"

    def test__setitem_jsonencode(self, persistent_json):
        with patch.object(persistent_json, "_jsonencoder") as mock_jsonencoder:

            persistent_json["a"] = 1
            mock_jsonencoder.assert_called_once_with(1)

    def test__setitem___datetime_raises(self, persistent_json):
        with pytest.raises(TypeError):
            persistent_json["timestamp"] = pd.to_datetime("2020-01-01 00:00")

    def test__len__(self, persistent_json):
        assert len(persistent_json) == 0
        persistent_json.update({"a": 1, "b": None})
        assert len(persistent_json) == 2

    def test_update(self, persistent_json):
        assert persistent_json._PersistentJSON__dict == {}
        persistent_json.update({"a": 1.0, "b": "test"})
        assert persistent_json._PersistentJSON__dict == {"a": 1.0, "b": "test"}

    def test_pull(self, local_file_handler_w_content):
        handler = local_file_handler_w_content
        persistent_json = PersistentJSON(handler)
        persistent_json.pull()

        content_out = persistent_json._PersistentJSON__dict
        content_expected = {"this": 1, "is": "hei", "a": None, "test": 1.2}

        assert content_out == content_expected

    def test_pull_non_existing(self):
        handler = LocalFileHandler("./non_existing.json")
        persistent_json = PersistentJSON(handler)
        persistent_json.pull(raise_on_missing=False)

        content_out = persistent_json._PersistentJSON__dict
        content_expected = {}

        assert content_out == content_expected

    def test_pull_non_existing_raises(self):
        handler = LocalFileHandler("./non_existing.json")
        persistent_json = PersistentJSON(handler)

        with pytest.raises(FileNotFoundError):
            persistent_json.pull(raise_on_missing=True)

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        persistent_json = PersistentJSON(handler)

        content = {"this": 1, "is": "hei", "a": None, "test": 1.2}

        persistent_json.update(content)
        persistent_json.push()

        with open(tmp_path / "test.json", mode="r") as f:
            content_out = json.load(f)

        assert content_out == content

    def test_push_empty(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        persistent_json = PersistentJSON(handler)

        persistent_json.push()

        with open(tmp_path / "test.json", mode="r") as f:
            content_out = json.load(f)

        assert content_out == {}


class Test_ResultCollector:
    def test__init__default(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers)

        assert results._headers == headers
        assert results._indexing_mode == "auto"
        assert isinstance(results._handler, NullHandler)
        df_expect = pd.DataFrame(columns=headers.keys()).astype(headers)
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

    @pytest.mark.parametrize("mode", ["auto", "timestamp"])
    def test__init__auto(self, mode):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode=mode)
        assert results._indexing_mode == mode

    def test__init__mode_raises(self):
        with pytest.raises(ValueError):
            headers = {"a": float, "b": str}
            ResultCollector(headers, indexing_mode="invalid-mode")

    def test__init___dtype_raises(self):
        with pytest.raises(ValueError):
            ResultCollector({"a": tuple})

    def test__init__handler(self, local_file_handler_empty):
        handler = local_file_handler_empty
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, handler=handler)
        assert results._handler == handler

    def test_new_row_auto(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        df_expect = pd.DataFrame(
            columns=("a", "b"), index=pd.Int64Index([0])
        ).astype(headers)
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

        results.new_row()
        df_expect = pd.DataFrame(
            columns=("a", "b"), index=pd.Int64Index([0, 1])
        ).astype(headers)
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

    def test_new_row_timestamp(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="timestamp")

        results.new_row("2020-01-01 00:00")
        df_expect = pd.DataFrame(
            columns=("a", "b"),
            index=pd.DatetimeIndex(["2020-01-01 00:00"], tz="utc")
        ).astype(headers)
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

        results.new_row("2020-01-01 01:00")
        df_expect = pd.DataFrame(
            columns=("a", "b"),
            index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00"], tz="utc")
        ).astype(headers)
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

    def test_new_row_auto_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="auto")

        with pytest.raises(TypeError):
            results.new_row("2020-01-01 00:00")

    def test_new_row_timestamp_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="timestamp")

        with pytest.raises(TypeError):
            results.new_row()

    def test_new_row_duplicate_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")

        with pytest.raises(ValueError):
            results.new_row("2020-01-01 00:00")

    def test_collect(self):
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        results.collect(a=1.1, b="test")
        results.new_row()
        results.collect(a=2.2, b="overridden-value")
        results.collect(a=3.3, b="updated-value")
        results.new_row()

        a_expect = np.array([1.1, 3.3, np.nan]).astype(float)
        b_expect = np.array(["test", "updated-value", np.nan]).astype(str)
        c_expect = np.array([np.nan, np.nan, np.nan]).astype(float)
        d_expect = np.array([np.nan, np.nan, np.nan]).astype(str)

        a_out = results._dataframe["a"].values
        b_out = results._dataframe["b"].values
        c_out = results._dataframe["c"].values
        d_out = results._dataframe["d"].values

        np.testing.assert_array_almost_equal(a_out, a_expect)
        np.testing.assert_array_equal(b_out, b_expect)
        np.testing.assert_array_almost_equal(c_out, c_expect)
        np.testing.assert_array_equal(d_out, d_expect)

    def test_collect_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="auto")

        # raise on string
        with pytest.raises(ValueError):
            results.collect(a="hei")

        # raise on float
        with pytest.raises(ValueError):
            results.collect(b=1.0)

    def test_pull_auto(self):
        path = Path(__file__).parent / "testdata/results_auto.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="auto")

        results.pull()

        index_expect = pd.Int64Index([0, 1, 2])
        a_expect = np.array([1.1, 3.3, np.nan]).astype(float)
        b_expect = np.array(["foo", "bar", np.nan]).astype(str)
        c_expect = np.array([np.nan, np.nan, np.nan]).astype(float)
        d_expect = np.array([np.nan, np.nan, np.nan]).astype(str)

        index_out = results._dataframe.index
        a_out = results._dataframe["a"].values
        b_out = results._dataframe["b"].values
        c_out = results._dataframe["c"].values
        d_out = results._dataframe["d"].values

        pd.testing.assert_index_equal(index_out, index_expect)
        np.testing.assert_array_almost_equal(a_out, a_expect)
        np.testing.assert_array_equal(b_out, b_expect)
        np.testing.assert_array_almost_equal(c_out, c_expect)
        np.testing.assert_array_equal(d_out, d_expect)

    def test_pull_timestamp(self):
        path = Path(__file__).parent / "testdata/results_timestamp.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="timestamp")

        results.pull()

        index_expect = pd.DatetimeIndex([
            "2020-01-01 00:00:00+00:00",
            "2020-01-01 01:00:00+00:00",
            "2020-01-01 02:00:00+00:00"
        ], tz="utc")
        a_expect = np.array([1.1, 3.3, np.nan]).astype(float)
        b_expect = np.array(["foo", "bar", np.nan]).astype(str)
        c_expect = np.array([np.nan, np.nan, np.nan]).astype(float)
        d_expect = np.array([np.nan, np.nan, np.nan]).astype(str)

        index_out = results._dataframe.index
        a_out = results._dataframe["a"].values
        b_out = results._dataframe["b"].values
        c_out = results._dataframe["c"].values
        d_out = results._dataframe["d"].values

        pd.testing.assert_index_equal(index_out, index_expect)
        np.testing.assert_array_almost_equal(a_out, a_expect)
        np.testing.assert_array_equal(b_out, b_expect)
        np.testing.assert_array_almost_equal(c_out, c_expect)
        np.testing.assert_array_equal(d_out, d_expect)

    def test_pull_missing_header_raises(self):
        path = Path(__file__).parent / "testdata/results_auto.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str, "missing-header": float}
        results = ResultCollector(headers, handler=handler, indexing_mode="auto")

        with pytest.raises(ValueError):
            results.pull()

    def test_pull_wrong_mode_raises(self):
        path = Path(__file__).parent / "testdata/results_auto.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="timestamp")

        with pytest.raises(ValueError):
            results.pull()

    def test_pull_wrong_mode_raises2(self):
        path = Path(__file__).parent / "testdata/results_timestamp.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="auto")

        with pytest.raises(ValueError):
            results.pull()

    def test_pull_non_existing(self):
        handler = LocalFileHandler("./non_exist.csv")
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler)
        results.pull(raise_on_missing=False)
        df_expect = pd.DataFrame(columns=headers.keys()).astype(headers)
        df_out = results._dataframe
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_pull_non_existing_raises(self):
        handler = LocalFileHandler("./non_exist.csv")
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler)
        with pytest.raises(FileNotFoundError):
            results.pull(raise_on_missing=True)

    def test_push_auto(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "results.csv")
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.1, b="test")
        results.new_row()
        results.collect(a=2.2, b="value")
        results.push()

        with open(tmp_path / "results.csv", mode="r") as f:
            csv_out = f.read()

        csv_expect = ",a,b,c,d\n0,1.1,test,,nan\n1,2.2,value,,nan\n"
        assert csv_out == csv_expect

    def test_push_timestamp(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "results.csv")
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.1, b="test")
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.2, b="value")
        results.push()

        with open(tmp_path / "results.csv", mode="r") as f:
            csv_out = f.read()

        csv_expect = ",a,b,c,d\n2020-01-01 00:00:00+00:00,1.1,test,,nan\n2020-01-01 01:00:00+00:00,2.2,value,,nan\n"
        assert csv_out == csv_expect

    def test_dataframe_auto(self):
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.1, b="test")
        results.new_row()
        results.collect(a=2.2, b="value")

        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.1, 2.2],
                "b": ["test", "value"],
                "c": [np.nan, np.nan],
                "d": [np.nan, np.nan]
            },
            index=pd.Int64Index([0, 1])
        ).astype(headers)

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_dataframe_timestamp(self):
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.1, b="test")
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.2, b="value")

        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.1, 2.2],
                "b": ["test", "value"],
                "c": [np.nan, np.nan],
                "d": [np.nan, np.nan]
            },
            index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00"], tz="utc")
        ).astype(headers)

        pd.testing.assert_frame_equal(df_out, df_expect)
