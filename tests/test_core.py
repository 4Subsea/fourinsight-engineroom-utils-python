import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from azure.core.exceptions import ResourceNotFoundError

from fourinsight.engineroom.utils import (
    AzureBlobHandler,
    LocalFileHandler,
    PersistentJSON,
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
        assert handler.pull() is None

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

    def test_pull_no_exist(self, azure_blob_handler_mocked):
        handler = azure_blob_handler_mocked

        def raise_resource_not_found(*args, **kwargs):
            raise ResourceNotFoundError

        handler._blob_client.download_blob.side_effect = raise_resource_not_found

        assert handler.pull() is None

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
        with patch.object(PersistentJSON, "pull") as mock_pull:
            persistent_json = PersistentJSON(handler)
            assert persistent_json._PersistentJSON__dict == {}
            assert persistent_json._handler == handler
            mock_pull.assert_called_once()

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

    def test_pull_empty(self, local_file_handler_empty):
        handler = local_file_handler_empty
        persistent_json = PersistentJSON(handler)
        persistent_json.pull()

        content_out = persistent_json._PersistentJSON__dict
        content_expected = {}

        assert content_out == content_expected

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