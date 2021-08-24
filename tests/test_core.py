from pathlib import Path
import pandas as pd
import json

import pytest
from unittest.mock import Mock, patch

from fourinsight.engineroom.utils import LocalFileHandler, PersistentJSON
from fourinsight.engineroom.utils.core import BaseHandler


@pytest.fixture
def handler_empty(tmp_path):
    return LocalFileHandler(tmp_path / "test.json")

@pytest.fixture
def handler_w_content():
    path = Path(__file__).parent / "testdata/a_test_file.json"
    return LocalFileHandler(path)


@pytest.fixture
def persistent_json(handler_empty):
    return PersistentJSON(handler_empty)


class Test_LocalFileHandler:

    def test__init__(self):
        handler = LocalFileHandler("./some/path")
        assert handler._path == "./some/path"
        assert isinstance(handler, BaseHandler)

    def test_pull(self, handler_w_content):
        text_out = handler_w_content.pull()
        text_expect = "{\n    \"this\": 1,\n    \"is\": \"hei\",\n    \"a\": null,\n    \"test\": 1.2\n}"

        assert text_out == text_expect

    def test_pull_non_existing(self):
        handler = LocalFileHandler("non-existing-file")
        assert handler.pull() is None

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")

        content = "this is a test file,\nwith two lines."
        handler.push(content)

        assert open(tmp_path / "test.json", mode="r").read() == content


class Test_PersistentJSON:

    def test__init__(self, handler_empty):
        with patch.object(PersistentJSON, "pull") as mock_pull:
            persistent_json = PersistentJSON(handler_empty)
            assert persistent_json._PersistentJSON__dict == {}
            mock_pull.assert_called_once()

    def test__init__raises(self):
        with pytest.raises(TypeError):
            PersistentJSON(Mock())

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

    def test__setitem__(self, persistent_json):
        persistent_json["a"] = "some value"
        persistent_json["b"] = "some other value"

        assert persistent_json["a"] == "some value"
        assert persistent_json["b"] == "some other value"

        with pytest.raises(KeyError):
            persistent_json["non-existing-key"]

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

    def test_pull(self, handler_w_content):
        persistent_json = PersistentJSON(handler_w_content)
        persistent_json.pull()

        content_out = persistent_json._PersistentJSON__dict
        content_expected = {
            "this": 1,
            "is": "hei",
            "a": None,
            "test": 1.2
        }

        assert content_out == content_expected

    def test_pull_empty(self, handler_empty):
        persistent_json = PersistentJSON(handler_empty)
        persistent_json.pull()

        content_out = persistent_json._PersistentJSON__dict
        content_expected = {}

        assert content_out == content_expected

    def test_push(self, tmp_path):
        handler = LocalFileHandler(tmp_path / "test.json")
        persistent_json = PersistentJSON(handler)

        content = {
            "this": 1,
            "is": "hei",
            "a": None,
            "test": 1.2
        }

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
