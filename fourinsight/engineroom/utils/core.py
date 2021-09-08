import json
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from io import StringIO
from pathlib import Path

import pandas as pd
from azure.storage.blob import BlobClient


class BaseHandler(ABC):
    """
    Abstract class for push/pull file content from a remote/persistent source.
    """

    @abstractmethod
    def pull(self, raise_on_missing=True):
        raise NotImplementedError()

    @abstractmethod
    def push(self, local_content):
        raise NotImplementedError()


class NullHandler(BaseHandler):
    """
    NullHandler.

    Will raise an exception if push() or pull() is called.
    """
    def __repr__(self):
        return "NullHandler"

    def pull(self, *args, **kwargs):
        raise ValueError(
            "The 'NullHandler' does not provide any push or pull functionality."
        )

    def push(self, *args, **kwargs):
        raise ValueError(
            "The 'NullHandler' does not provide any push or pull functionality."
        )


class LocalFileHandler(BaseHandler):
    """
    Handler for push/pull text content to/from local file.

    Parameters
    ----------
    path : str
        File path.
    """

    def __init__(self, path):
        self._path = Path(path)

    def __repr__(self):
        return f"LocalFileHandler {self._path.resolve()}"

    def pull(self, raise_on_missing=True):
        """
        Pull text content from file. Returns None if file is not found.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if content can not be pulled from file.
        """
        try:
            remote_content = open(self._path, mode="r").read()
        except Exception as e:
            remote_content = None
            if raise_on_missing:
                raise e
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
    Handler for push/pull text content to/from Azure Blob Storage.

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

    def __repr__(self):
        return f"AzureBlobHandler {self._container_name}/{self._blob_name}"

    def pull(self, raise_on_missing=True):
        """
        Pull text content from blob. Returns None if resource is not found.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if content can not be pulled from blob.
        """
        try:
            remote_content = self._blob_client.download_blob(encoding="utf-8").readall()
        except Exception as e:
            remote_content = None
            if raise_on_missing:
                raise e
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
        return repr(self.__dict)

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

    def pull(self, raise_on_missing=True):
        """
        Pull content from source. Remote source overwrites existing values.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if content can not be pulled from source.
        """
        remote_content = self._handler.pull(raise_on_missing=raise_on_missing)
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
    """
    Collect and store indexed results.

    This class provides a simple interface to collect, store, and index
    intermediate results. The results are stored in a pandas.DataFrame internally.
    Using a handler, the results can be 'pushed' or 'pulled' from a remote source.

    Parameters
    ----------
    headers : dict
        Header names and data types as key/value pairs. The collector will only accept
        intermediate results defined here.
    handler:
        Handler extended from `BaseHandler`. Default handler is `NullHandler`, which
        does not provide any push or pull functionality.
    indexing_mode : str
        Indexing mode. Should be 'auto' or 'timestamp'.
    """

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

    def __repr__(self):
        return repr(self._dataframe)

    def new_row(self, index=None):
        """
        Make a new row.

        Parameters
        ----------
        index :
            The new index value. If indexing_mode is set to 'auto', index should be None.
            If indexing_mode is set to 'timestamp', index should be a unique datetime
            that is passed on to pandas.to_datetime.
        """
        if index:
            index = pd.to_datetime(index, utc=True)
        else:
            index = self._index_counter
        self._verify_index(index)

        row_new = pd.DataFrame(index=[index])
        self._dataframe = self._dataframe.append(
            row_new, verify_integrity=True, sort=False
        )
        self._index_counter += 1

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
        """
        Collect and store results under the current index.

        Parameters
        ----------
        results : keyword arguments
            The results are passed as keyword arguments, where the keyword must
            be one of the `headers`. Provided values must be of correct type (provided
            during initialization).
        """
        if not set(self._headers.keys()).issuperset(results):
            raise KeyError("Keyword must be in headers.")

        self._check_types(results)
        current_index = self._dataframe.index[-1]
        row_update = pd.DataFrame(data=results, index=[current_index])
        self._dataframe.update(row_update, errors="ignore")

    def _check_types(self, results):
        for header, value in results.items():
            if not isinstance(value, self._headers[header]):
                raise ValueError(f"Invalid dtype in '{header}'")

    def pull(self, raise_on_missing=True):
        """
        Pull results from source. Remote source overwrites existing values.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if results can not be pulled from source.
        """

        dataframe_csv = self._handler.pull(raise_on_missing=raise_on_missing)
        if not dataframe_csv:
            return

        df = pd.read_csv(StringIO(dataframe_csv), index_col=0, parse_dates=True)

        if not (set(df.columns) == set(self._headers.keys())):
            raise ValueError("Header is not valid.")

        if (self._indexing_mode == "auto") and not (
            isinstance(df.index, pd.Int64Index)
        ):
            raise ValueError("Index must be 'Int64Index'.")
        elif (self._indexing_mode == "timestamp") and not (
            isinstance(df.index, pd.DatetimeIndex)
        ):
            raise ValueError("Index must be 'DatetimeIndex'.")

        self._dataframe = df

    def push(self):
        """
        Push results to source.
        """
        local_content = self._dataframe.to_csv(
            sep=",", index=True, line_terminator="\n"
        )
        self._handler.push(local_content)

    @property
    def dataframe(self):
        """Return a copy of internal dataframe"""
        return self._dataframe.copy(deep=True)
