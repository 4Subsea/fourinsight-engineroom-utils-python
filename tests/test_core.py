from pathlib import Path
from fourinsight.engineroom.utils import LocalFileHandler
from fourinsight.engineroom.utils.core import BaseHandler


class TestBaseHandler:
    pass


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
