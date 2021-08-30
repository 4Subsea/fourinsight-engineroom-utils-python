import json
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path
from io import StringIO

import pandas as pd

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient


class BaseHandler(ABC):
    """
    Abstract class for push/pull file content from a remote/persistent source.
    """

    @abstractmethod
    def pull(self):
        raise NotImplementedError()

    @abstractmethod
    def push(self, local_content):
        raise NotImplementedError()


class NullHandler(BaseHandler):
    """
    NullHandler
    """

    def pull(self):
        raise ValueError("No handler is provided.")

    def push(self, local_content):
        raise ValueError("No handler is provided.")


class LocalFileHandler(BaseHandler):
    """
    Handler for push/pull file content to/from local file.

    Parameters
    ----------
    path : str
        File path.
    """

    def __init__(self, path):
        self._path = Path(path)

    def pull(self):
        """
        Pull content from file. Returns None if file is not found.
        """
        try:
            remote_content = open(self._path, mode="r").read()
        except FileNotFoundError:
            remote_content = None
        return remote_content

    def push(self, local_content):
        """
        Push content to file.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, mode="w") as f:
            f.write(local_content)


class AzureBlobHandler(BaseHandler):
    """
    Handler for push/pull file content to/from Azure Blob Storage.

    Parameters
    ----------
    conn_str : str
        A connection string to an Azure Storage account.
    container_name : str
        The container name for the blob.
    blob_name : str
        The name of the blob with which to interact.
    """

    def __init__(self, conn_str, container_name, blob_name):
        self._conn_str = conn_str
        self._container_name = container_name
        self._blob_name = blob_name
        self._blob_client = BlobClient.from_connection_string(
            conn_str, container_name, blob_name
        )

    def pull(self):
        """
        Pull content from blob as text. Returns None if resource is not found.
        """
        try:
            remote_content = self._blob_client.download_blob(encoding="utf-8").readall()
        except ResourceNotFoundError:
            remote_content = None
        return remote_content

    def push(self, local_content):
        """
        Push content to blob.
        """
        self._blob_client.upload_blob(local_content, overwrite=True)


class PersistentJSON(MutableMapping):
    """
    Persistent JSON.

    Push/pull a JSON object stored persistently in a "remote" location.
    This class is usefull when loading configurations or keeping persistent
    state.

    The class behaves exactly like a `dict` but only accepts values that are
    JSON encodable.

    Parameters
    ----------
    handler : cls
        Handler extended from `BaseHandler`.
    """

    def __init__(self, handler):
        self.__dict = {}
        self._jsonencoder = json.JSONEncoder().encode
        self._handler = handler

        if not isinstance(self._handler, BaseHandler):
            raise TypeError("Handler does not inherit from BaseHandler")

    def __repr__(self):
        return "PersistentJSON " + repr(self.__dict)

    def __delitem__(self, key):
        del self.__dict[key]

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __setitem__(self, key, value):
        try:
            self._jsonencoder(value)
        except TypeError as err:
            raise err
        else:
            self.__dict[key] = value

    def pull(self):
        """
        Pull content from source. Remote source overwrites existing values.
        """
        remote_content = self._handler.pull()
        if remote_content is None:
            remote_content = "{}"
        self.__dict.update(json.loads(remote_content))

    def push(self):
        """
        Push content to source.
        """
        local_content = json.dumps(self.__dict, indent=4)
        self._handler.push(local_content)


class ResultCollector:
    _INDEX_DTYPE_MAP = {"auto": int, "timestamp": pd.Timestamp}
    _VALID_DATA_DTYPES = {float, str}

    def __init__(self, headers, handler=None, indexing_mode="auto"):
        self._headers = headers
        self._indexing_mode = indexing_mode.lower()
        if handler:
            self._handler = handler
        else:
            self._handler = NullHandler()

        if not self._VALID_DATA_DTYPES.issuperset(self._headers.values()):
            raise ValueError("Only 'float' and 'str' dtypes are supported.")

        if self._indexing_mode not in ("auto", "timestamp"):
            raise ValueError("Indexing mode must be 'auto' or 'timestamp'.")

        self._dataframe = pd.DataFrame(columns=headers.keys()).astype(self._headers)
        self._index_counter = 0

    def new_row(self, index=None):
        next_index = self._next_index(index)
        row_new = pd.DataFrame(index=[next_index])
        self._dataframe = self._dataframe.append(row_new, verify_integrity=True, sort=False)
        self._index_counter += 1

    def _next_index(self, index):
        if index:
            next_index = pd.to_datetime(index, utc=True)
        else:
            next_index = self._index_counter
        self._verify_index(next_index)
        return next_index

    def _verify_index(self, index):
        expected_dtype = self._INDEX_DTYPE_MAP[self._indexing_mode]
        if not isinstance(index, expected_dtype):
            raise TypeError(
                f"Index dtype '{type(index)}' is not valid when using indexing "
                + f"mode '{self._indexing_mode}'."
            )

        if index in (self._dataframe.index):
            raise ValueError("Index already exists.")

    def collect(self, **results):
        if not set(self._headers.keys()).issuperset(results):
            raise KeyError("Keyword must be in headers.")

        self._check_types(results)
        current_index = self._dataframe.index[-1]
        row_update = pd.DataFrame(results, index=[current_index])
        self._dataframe.update(row_update, errors="ignore")

    def _check_types(self, results):
        for header, value in results.items():
            if not isinstance(value, self._headers[header]):
                raise ValueError(f"Invalid dtype in '{header}'")

    def pull(self):
        """
        Pull results from source. Remote source overwrites existing values.
        """
        if self._indexing_mode == "auto":
            parse_dates = False
        else:
            parse_dates = True
        dataframe_csv = StringIO(self._handler.pull())
        df = pd.read_csv(dataframe_csv, index_col=0, parse_dates=parse_dates)

        if not (set(df.columns) == set(self._headers.keys())):
            raise ValueError("Header is not valid.")

        if (self._indexing_mode == "auto") and not (isinstance(df.index, pd.Int64Index)):
            raise ValueError(f"Index must be 'Int64Index'.")
        elif (self._indexing_mode == "timestamp") and not (isinstance(df.index, pd.DatetimeIndex)):
            raise ValueError("Index must be 'DatetimeIndex'.")

        self._dataframe = df.astype(self._headers)

    def push(self):
        """
        Push results to source.
        """
        local_content = self._dataframe.to_csv(sep=",", index=True, line_terminator="\n")
        self._handler.push(local_content)

    def dataframe(self):
        """Return a copy of internal dataframe"""
        return self._dataframe.copy(deep=True)
