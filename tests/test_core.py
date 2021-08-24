from pathlib import Path

import pytest
from unittest.mock import Mock

from fourinsight.engineroom.utils import LocalFileHandler, PersistentJSON
from fourinsight.engineroom.utils.core import BaseHandler


@pytest.fixture
def local_file_handler(tmp_path):
    yield LocalFileHandler(tmp_path / "test.json")


@pytest.fixture
def persistent_json(local_file_handler):
    yield PersistentJSON(local_file_handler)


class Test_LocalFileHandler:

    def test__init__(self):
        handler = LocalFileHandler("./some/path")
        assert handler._path == "./some/path"
        assert isinstance(handler, BaseHandler)

    def test_pull(self):
        path = Path(__file__).parent / "testdata/a_test_file.json"
        handler = LocalFileHandler(path)

        text_out = handler.pull()
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

    def test__len__(self, persistent_json):
        assert len(persistent_json) == 0
        persistent_json.update({"a": 1, "b": None})
        assert len(persistent_json) == 2

    def test_update(self, persistent_json):
        assert persistent_json._PersistentJSON__dict == {}
        persistent_json.update({"a": 1.0, "b": "test"})
        assert persistent_json._PersistentJSON__dict == {"a": 1.0, "b": "test"}
