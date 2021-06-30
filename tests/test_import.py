def test_import_version():
    import fourinsight.engineroom.utils as feru
    assert isinstance(feru.__version__, str)
