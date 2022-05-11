import json
from io import BytesIO, TextIOWrapper
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from azure.core.exceptions import ResourceNotFoundError

from fourinsight.engineroom.utils import (
    AzureBlobHandler,
    BaseHandler,
    LocalFileHandler,
    NullHandler,
    PersistentDict,
    ResultCollector,
)

REMOTE_FILE_PATH = Path(__file__).parent / "testdata/a_test_file.json"


@pytest.fixture
def local_file_handler_empty(tmp_path):
    return LocalFileHandler(tmp_path / "test.json")


@pytest.fixture
def local_file_handler_w_content():
    return LocalFileHandler(REMOTE_FILE_PATH)


@pytest.fixture
def persistent_dict(local_file_handler_empty):
    return PersistentDict(local_file_handler_empty)


@pytest.fixture
@patch("fourinsight.engineroom.utils._core.BlobClient.from_connection_string")
def azure_blob_handler_mocked(mock_from_connection_string):
    connection_string = "some_connection_string"
    container_name = "some_container_name"
    blob_name = "some_blob_name"
    handler = AzureBlobHandler(connection_string, container_name, blob_name)

    remote_content = open(REMOTE_FILE_PATH, mode="r").read()
    handler._blob_client.download_blob.return_value.readinto.side_effect = (
        lambda buffer: handler.write(remote_content)
    )

    mock_from_connection_string.assert_called_once_with(
        "some_connection_string", "some_container_name", "some_blob_name"
    )

    return handler


class Test_BaseHandler:
    def test__init__(self):
        handler = BaseHandler()
        assert isinstance(handler, TextIOWrapper)
        assert isinstance(handler.buffer, BytesIO)

    def test_getvalue(self):
        handler = BaseHandler()
        handler.write("This is some test content.")
        out = handler.getvalue()
        expect = "This is some test content."
        assert out == expect

    def test__pull(self):
        handler = BaseHandler()
        with pytest.raises(NotImplementedError):
            handler._pull()

    def test__push(self):
        handler = BaseHandler()
        with pytest.raises(NotImplementedError):
            handler._push()

    def test_pull_resource_not_found(self):
        handler = BaseHandler()
        handler.write("This is some test content.")
        handler.pull(raise_on_missing=False)
        out = handler.getvalue()
        expect = ""
        assert out == expect

    def test_pull_source_not_found_raises(self):
        handler = BaseHandler()
        handler.write("This is some test content.")
        with pytest.raises(NotImplementedError):
            handler.pull(raise_on_missing=True)
        out = handler.getvalue()
        expect = "This is some test content."
        assert out == expect

    def test_pull(self):
        handler = BaseHandler()
        handler.write("This is some test content.")
        with patch.object(handler, "_pull") as mocked__pull:
            mocked__pull.side_effect = lambda: handler.write("Source content.")
            handler.pull()
            out = handler.getvalue()
            expect = "Source content."
            assert out == expect

    def test_push(self):
        handler = BaseHandler()
        with patch.object(handler, "_push") as mocked__push:
            handler.push()
            mocked__push.assert_called_once()


class Test_NullHandler:
    def test__init__(self):
        handler = NullHandler()
        assert isinstance(handler, BaseHandler)
        assert handler._SOURCE_NOT_FOUND_ERROR is NotImplementedError

    def test__repr__(self, azure_blob_handler_mocked):
        assert str(NullHandler()) == "NullHandler"

    def test_pull_raises(self):
        handler = NullHandler()
        with pytest.raises(NotImplementedError):
            handler.pull(raise_on_missing=True)

    def test_pull(self):
        handler = NullHandler()
        handler.write("Some random content.")
        handler.pull(raise_on_missing=False)
        assert handler.getvalue() == ""

    def test_push(self):
        handler = NullHandler()
        with pytest.raises(NotImplementedError):
            handler.push()


class Test_LocalFileHandler:
    def test__init__(self):
        handler = LocalFileHandler("./some/path")
        assert handler._path == Path("./some/path")
        assert isinstance(handler, BaseHandler)
        assert handler._SOURCE_NOT_FOUND_ERROR is FileNotFoundError

    def test_pull(self, local_file_handler_w_content):
        handler = local_file_handler_w_content
        assert handler.getvalue() == ""
        handler.pull()
        assert handler.getvalue() == (
            '{\n    "this": 1,\n    "is": "hei",\n    "a": null,\n    "test": 1.2\n}'
        )

    def test_pull_non_existing(self):
        handler = LocalFileHandler("non-exisiting-file")
        handler.write("Some initial content.")
        handler.pull(raise_on_missing=False)
        assert handler.getvalue() == ""

    def test_pull_non_existing_raises(self):
        handler = LocalFileHandler("non-exisiting-file")
        with pytest.raises(FileNotFoundError):
            handler.pull(raise_on_missing=True)

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        handler.write("Some random content")
        handler.push()
        assert open(tmp_path / "test.json", mode="r").read() == "Some random content"


class Test_AzureBlobHandler:
    @patch("fourinsight.engineroom.utils._core.BlobClient.from_connection_string")
    def test__init__(self, mock_from_connection_string):
        handler = AzureBlobHandler(
            "some_connection_string", "some_container_name", "some_blob_name"
        )

        assert handler._conn_str == "some_connection_string"
        assert handler._container_name == "some_container_name"
        assert handler._blob_name == "some_blob_name"
        assert handler._SOURCE_NOT_FOUND_ERROR is ResourceNotFoundError
        mock_from_connection_string.assert_called_once_with(
            "some_connection_string", "some_container_name", "some_blob_name"
        )

    def test__repr__(self, azure_blob_handler_mocked):
        assert (
            str(azure_blob_handler_mocked)
            == "AzureBlobHandler some_container_name/some_blob_name"
        )

    def test__init__fixture(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked
        assert handler._conn_str == "some_connection_string"
        assert handler._container_name == "some_container_name"
        assert handler._blob_name == "some_blob_name"
        assert handler._SOURCE_NOT_FOUND_ERROR is ResourceNotFoundError
        assert isinstance(handler._blob_client, Mock)

    def test_pull(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked
        handler.pull()
        assert (
            handler.getvalue()
            == '{\n    "this": 1,\n    "is": "hei",\n    "a": null,\n    "test": 1.2\n}'
        )
        handler._blob_client.download_blob.return_value.readinto.assert_called_once_with(
            handler.buffer
        )

    def test_pull_non_existing(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        def raise_resource_not_found(*args, **kwargs):
            raise ResourceNotFoundError

        handler._blob_client.download_blob.side_effect = raise_resource_not_found
        handler.pull(raise_on_missing=False)
        assert handler.getvalue() == ""

    def test_pull_non_existing_raises(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        def raise_resource_not_found(*args, **kwargs):
            raise ResourceNotFoundError

        handler._blob_client.download_blob.side_effect = raise_resource_not_found

        with pytest.raises(ResourceNotFoundError):
            handler.pull(raise_on_missing=True)

    def test_push(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        content = "Some random content."
        handler.write(content)
        handler.push()

        handler._blob_client.upload_blob.assert_called_once_with(
            content, overwrite=True
        )


class Test_PersistentDict:
    def test__init__(self, local_file_handler_empty):
        handler = local_file_handler_empty
        persistent_dict = PersistentDict(handler)
        assert persistent_dict._PersistentDict__dict == {}
        assert persistent_dict._handler == handler

    def test__init__raises(self):
        with pytest.raises(TypeError):
            PersistentDict("A")

    def test__init__default(self):
        persistent_dict = PersistentDict()
        assert persistent_dict._PersistentDict__dict == {}
        assert isinstance(persistent_dict._handler, NullHandler)

    def test__repr__(self, persistent_dict):
        assert str(persistent_dict) == "{}"
        persistent_dict.update({"a": 1.0, "b": "test"})
        assert str(persistent_dict) == "{'a': 1.0, 'b': 'test'}"

    def test__delitem__(self, persistent_dict):
        persistent_dict.update({"a": 1.0, "b": "test"})
        del persistent_dict["a"]
        persistent_dict._PersistentDict__dict == {"b": "test"}

    def test__getitem__(self, persistent_dict):
        persistent_dict.update({"a": 1.0, "b": "test"})
        assert persistent_dict["a"] == 1.0
        assert persistent_dict["b"] == "test"

        with pytest.raises(KeyError):
            persistent_dict["non-existing-key"]

    def test__setitem__(self, persistent_dict):
        persistent_dict["a"] = "some value"
        persistent_dict["b"] = "some other value"

        assert persistent_dict["a"] == "some value"
        assert persistent_dict["b"] == "some other value"

    def test__setitem_jsonencode(self, persistent_dict):
        with patch.object(persistent_dict, "_jsonencoder") as mock_jsonencoder:

            persistent_dict["a"] = 1
            mock_jsonencoder.assert_called_once_with(1)

    def test__setitem___datetime_raises(self, persistent_dict):
        with pytest.raises(TypeError):
            persistent_dict["timestamp"] = pd.to_datetime("2020-01-01 00:00")

    def test__len__(self, persistent_dict):
        assert len(persistent_dict) == 0
        persistent_dict.update({"a": 1, "b": None})
        assert len(persistent_dict) == 2

    def test_update(self, persistent_dict):
        assert persistent_dict._PersistentDict__dict == {}
        persistent_dict.update({"a": 1.0, "b": "test"})
        assert persistent_dict._PersistentDict__dict == {"a": 1.0, "b": "test"}

    def test_pull(self, local_file_handler_w_content):
        handler = local_file_handler_w_content
        persistent_dict = PersistentDict(handler)
        persistent_dict.pull()

        content_out = persistent_dict._PersistentDict__dict
        content_expected = {"this": 1, "is": "hei", "a": None, "test": 1.2}

        assert content_out == content_expected

    def test_pull_non_existing(self):
        handler = LocalFileHandler("./non_existing.json")
        persistent_dict = PersistentDict(handler)
        persistent_dict.pull(raise_on_missing=False)

        content_out = persistent_dict._PersistentDict__dict
        content_expected = {}

        assert content_out == content_expected

    def test_pull_non_existing_raises(self):
        handler = LocalFileHandler("./non_existing.json")
        persistent_dict = PersistentDict(handler)

        with pytest.raises(FileNotFoundError):
            persistent_dict.pull(raise_on_missing=True)

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        persistent_dict = PersistentDict(handler)

        content = {"this": 1, "is": "hei", "a": None, "test": 1.2}

        persistent_dict.update(content)
        persistent_dict.push()

        with open(tmp_path / "test.json", mode="r") as f:
            content_out = json.load(f)

        assert content_out == content

    def test_push_empty(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        persistent_dict = PersistentDict(handler)

        persistent_dict.push()

        with open(tmp_path / "test.json", mode="r") as f:
            content_out = json.load(f)

        assert content_out == {}


class Test_ResultCollector:
    def test__init__default(self):
        headers = {"a": int, "b": float, "c": str}
        results = ResultCollector(headers)

        assert results._headers == {"a": "Int64", "b": "float64", "c": "string"}
        assert results._indexing_mode == "auto"
        assert isinstance(results._handler, NullHandler)
        df_expect = pd.DataFrame(columns=headers.keys()).astype(
            {"a": "Int64", "b": "float64", "c": "string"}
        )
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

    def tests__repr__(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers)
        assert str(results) == str(results.dataframe)

    def test_new_row_auto(self):
        headers = {"a": float, "b": str, "c": int}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        df_expect = pd.DataFrame(
            columns=("a", "b", "c"), index=pd.Index([0], dtype="int64")
        ).astype({"a": "float64", "b": "string", "c": "Int64"})
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

        results.new_row()
        df_expect = pd.DataFrame(
            columns=("a", "b", "c"), index=pd.Index([0, 1], dtype="int64")
        ).astype({"a": "float64", "b": "string", "c": "Int64"})
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

    def test_new_row_timestamp(self):
        headers = {"a": float, "b": str, "c": int}
        results = ResultCollector(headers, indexing_mode="timestamp")

        results.new_row("2020-01-01 00:00")
        df_expect = pd.DataFrame(
            columns=("a", "b", "c"),
            index=pd.DatetimeIndex(["2020-01-01 00:00"], tz="utc"),
        ).astype({"a": "float64", "b": "string", "c": "Int64"})
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

        results.new_row("2020-01-01 01:00")
        df_expect = pd.DataFrame(
            columns=("a", "b", "c"),
            index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00"], tz="utc"),
        ).astype({"a": "float64", "b": "string", "c": "Int64"})
        pd.testing.assert_frame_equal(results._dataframe, df_expect)

    def test_new_row_auto_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="auto")

        with pytest.raises(ValueError):
            results.new_row("2020-01-01 00:00")

    def test_new_row_timestamp_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="timestamp")

        with pytest.raises(ValueError):
            results.new_row()

    def test_new_row_duplicate_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")

        with pytest.raises(ValueError):
            results.new_row("2020-01-01 00:00")

    def test_collect(self):
        headers = {"a": float, "b": str, "c": float, "d": int}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        results.collect(a=1.1, b="test")
        results.new_row()
        results.collect(a=2.2, c=1.0)
        results.collect(a=3.3, d=1)
        results.new_row()

        df_out = results._dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.1, 3.3, None],
                "b": ["test", None, None],
                "c": [None, 1.0, None],
                "d": [None, 1, None],
            },
            index=pd.Index([0, 1, 2], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "float64", "d": "Int64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_collect_raises(self):
        headers = {"a": float, "b": str}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()

        # raise on string
        with pytest.raises(ValueError):
            results.collect(a="hei")

    def test_collect_single_float(self):
        headers = {"a": float}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        results.collect(a=1.1)

        df_out = results._dataframe
        df_expect = pd.DataFrame(
            data={"a": [1.1]},
            index=pd.Index([0], dtype="int64"),
        ).astype({"a": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_collect_single_int(self):
        headers = {"a": int}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        results.collect(a=1)

        df_out = results._dataframe
        df_expect = pd.DataFrame(
            data={"a": [1]},
            index=pd.Index([0], dtype="int64"),
        ).astype({"a": "Int64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_collect_single_str(self):
        headers = {"a": str}
        results = ResultCollector(headers, indexing_mode="auto")

        results.new_row()
        results.collect(a="one")

        df_out = results._dataframe
        df_expect = pd.DataFrame(
            data={"a": ["one"]},
            index=pd.Index([0], dtype="int64"),
        ).astype({"a": "string"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_append(self):
        headers = {"a": float, "b": str, "c": float, "d": int}
        results = ResultCollector(headers, indexing_mode="auto")

        data = {
            "a": [1.2, 2.3, 3.4],
            "b": ["string", None, "hi"],
            "c": [1.0, None, 3.0],
            "d": [None, 1, None],
        }
        df_in = pd.DataFrame(data=data)
        results.append(df_in)
        df_expect = df_in.astype(
            {"a": "float64", "b": "string", "c": "float64", "d": "Int64"}
        )

        df_out = results._dataframe
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_append_timestamp(self):
        headers = {"a": float, "b": str, "c": float, "d": int}
        results = ResultCollector(headers, indexing_mode="timestamp")

        data = {
            "a": [1.2, 2.3, 3.4],
            "b": ["string", None, "hi"],
            "c": [1.0, None, 3.0],
            "d": [None, 1, None],
        }
        index = pd.date_range(start="2021-11-01", end="2021-11-02", periods=3, tz="UTC")
        df_in = pd.DataFrame(data=data, index=index)
        results.append(df_in)
        df_expect = df_in.astype(
            {"a": "float64", "b": "string", "c": "float64", "d": "Int64"}
        )

        df_out = results._dataframe
        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_pull_auto(self):
        path = Path(__file__).parent / "testdata/results_auto.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, handler=handler, indexing_mode="auto")

        results.pull()

        df_out = results._dataframe

        df_expect = pd.DataFrame(
            data={
                "a": [1.1, 3.3, None],
                "b": ["foo", "bar", None],
                "c": [None, None, None],
                "d": [None, None, None],
            },
            index=pd.Index([0, 1, 2], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "float64", "d": "string"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_pull_timestamp(self):
        path = Path(__file__).parent / "testdata/results_timestamp.csv"
        handler = LocalFileHandler(path)
        headers = {"a": float, "b": str, "c": float, "d": int}
        results = ResultCollector(headers, handler=handler, indexing_mode="timestamp")

        results.pull()

        df_out = results._dataframe

        index_expect = pd.DatetimeIndex(
            [
                "2020-01-01 00:00:00+00:00",
                "2020-01-01 01:00:00+00:00",
                "2020-01-01 02:00:00+00:00",
            ],
            tz="utc",
        )

        df_expect = pd.DataFrame(
            data={
                "a": [1.1, 3.3, None],
                "b": ["foo", "bar", None],
                "c": [None, None, None],
                "d": [None, None, None],
            },
            index=index_expect,
        ).astype({"a": "float64", "b": "string", "c": "float64", "d": "Int64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

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
        df_expect = pd.DataFrame(columns=headers.keys()).astype(
            {"a": "float64", "b": "string", "c": "float64", "d": "string"}
        )
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

        csv_expect = ",a,b,c,d\n0,1.1,test,,\n1,2.2,value,,\n"
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

        csv_expect = (
            ",a,b,c,d\n"
            "2020-01-01 00:00:00+00:00,1.1,test,,\n"
            "2020-01-01 01:00:00+00:00,2.2,value,,\n"
        )
        assert csv_out == csv_expect

    def test_dataframe_auto(self):
        headers = {"a": int, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1, b="test")
        results.new_row()
        results.collect(a=2, b="value")

        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1, 2],
                "b": ["test", "value"],
                "c": [np.nan, np.nan],
                "d": [np.nan, np.nan],
            },
            index=pd.Index([0, 1], dtype="int64"),
        ).astype({"a": "Int64", "b": "string", "c": "float64", "d": "string"})

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
                "d": [np.nan, np.nan],
            },
            index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00"], tz="utc"),
        ).astype({"a": "float64", "b": "string", "c": "float64", "d": "string"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_cast_dtypes(self):
        headers = {"a": int, "b": str, "c": float, "d": str}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="test")
        results.new_row()
        results.collect(c=2, d=23.4)

        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1, None],
                "b": ["test", None],
                "c": [None, 2.0],
                "d": [None, "23.4"],
            },
            index=pd.Index([0, 1], dtype="int64"),
        ).astype({"a": "Int64", "b": "string", "c": "float64", "d": "string"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_single_int(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.delete_rows(2)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 4.0, 5.0],
                "b": ["1", "2", "4", "5"],
                "c": [1, 2, 4, 5],
                "d": [None, None, None, None],
            },
            index=pd.Index([0, 1, 2, 3], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_list_int(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.delete_rows([1, 2, 3])
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 5.0],
                "b": ["1", "5"],
                "c": [1, 5],
                "d": [None, None],
            },
            index=pd.Index([0, 1], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_single_datetime(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.delete_rows("2020-01-01 02:00")
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 4.0, 5.0],
                "b": ["1", "2", "4", "5"],
                "c": [1, 2, 4, 5],
                "d": [None, None, None, None],
            },
            index=pd.DatetimeIndex(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 03:00",
                    "2020-01-01 04:00",
                ],
                tz="utc",
            ),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_list_datetime(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.delete_rows(
            ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]
        )
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 5.0],
                "b": ["1", "5"],
                "c": [1, 5],
                "d": [None, None],
            },
            index=pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 04:00"], tz="utc"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_timestamp(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before="2020-01-01 01:00", after="2020-01-01 03:00")
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [2.0, 3.0, 4.0],
                "b": ["2", "3", "4"],
                "c": [2, 3, 4],
                "d": [None, None, None],
            },
            index=pd.DatetimeIndex(
                [
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 03:00",
                ],
                tz="utc",
            ),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_timestamp_before_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=None, after="2020-01-01 03:00")
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": ["1", "2", "3", "4"],
                "c": [1, 2, 3, 4],
                "d": [None, None, None, None],
            },
            index=pd.DatetimeIndex(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 03:00",
                ],
                tz="utc",
            ),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_timestamp_after_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before="2020-01-01 01:00", after=None)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [2.0, 3.0, 4.0, 5.0],
                "b": ["2", "3", "4", "5"],
                "c": [2, 3, 4, 5],
                "d": [None, None, None, None],
            },
            index=pd.DatetimeIndex(
                [
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 03:00",
                    "2020-01-01 04:00",
                ],
                tz="utc",
            ),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_timestamp_both_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="timestamp")
        results.new_row("2020-01-01 00:00")
        results.collect(a=1.0, b="1", c=1)
        results.new_row("2020-01-01 01:00")
        results.collect(a=2.0, b="2", c=2)
        results.new_row("2020-01-01 02:00")
        results.collect(a=3.0, b="3", c=3)
        results.new_row("2020-01-01 03:00")
        results.collect(a=4.0, b="4", c=4)
        results.new_row("2020-01-01 04:00")
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=None, after=None)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": ["1", "2", "3", "4", "5"],
                "c": [1, 2, 3, 4, 5],
                "d": [None, None, None, None, None],
            },
            index=pd.DatetimeIndex(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 01:00",
                    "2020-01-01 02:00",
                    "2020-01-01 03:00",
                    "2020-01-01 04:00",
                ],
                tz="utc",
            ),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_int(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=1, after=3)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [2.0, 3.0, 4.0],
                "b": ["2", "3", "4"],
                "c": [2, 3, 4],
                "d": [None, None, None],
            },
            index=pd.Index([0, 1, 2], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_int_before_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=None, after=3)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": ["1", "2", "3", "4"],
                "c": [1, 2, 3, 4],
                "d": [None, None, None, None],
            },
            index=pd.Index([0, 1, 2, 3], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_int_after_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=1, after=None)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [2.0, 3.0, 4.0, 5.0],
                "b": ["2", "3", "4", "5"],
                "c": [2, 3, 4, 5],
                "d": [None, None, None, None],
            },
            index=pd.Index([0, 1, 2, 3], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)

    def test_delete_rows_truncate_int_both_none(self):
        headers = {"a": float, "b": str, "c": int, "d": float}
        results = ResultCollector(headers, indexing_mode="auto")
        results.new_row()
        results.collect(a=1.0, b="1", c=1)
        results.new_row()
        results.collect(a=2.0, b="2", c=2)
        results.new_row()
        results.collect(a=3.0, b="3", c=3)
        results.new_row()
        results.collect(a=4.0, b="4", c=4)
        results.new_row()
        results.collect(a=5.0, b="5", c=5)

        results.truncate(before=None, after=None)
        df_out = results.dataframe
        df_expect = pd.DataFrame(
            data={
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": ["1", "2", "3", "4", "5"],
                "c": [1, 2, 3, 4, 5],
                "d": [None, None, None, None, None],
            },
            index=pd.Index([0, 1, 2, 3, 4], dtype="int64"),
        ).astype({"a": "float64", "b": "string", "c": "Int64", "d": "float64"})

        pd.testing.assert_frame_equal(df_out, df_expect)
