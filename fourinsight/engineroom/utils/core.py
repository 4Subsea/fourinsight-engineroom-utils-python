import json
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from io import StringIO
from pathlib import Path

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
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
    Goes nowhere, does nothing. This handler is intended for objects that
    required a handler, but the push/pull functionality is not needed.

    Will raise an exception if :meth:`.push()` or :meth:`pull()` is called.
    """

    _ERROR_MSG = "The 'NullHandler' does not provide push/pull functionality."

    def __repr__(self):
        return "NullHandler"

    def pull(self, *args, **kwargs):
        raise NotImplementedError(self._ERROR_MSG)

    def push(self, *args, **kwargs):
        raise NotImplementedError(self._ERROR_MSG)


class LocalFileHandler(BaseHandler):
    """
    Handler for push/pull text content to/from a local file.

    Parameters
    ----------
    path : str or path object
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
        except FileNotFoundError as e:
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
        except ResourceNotFoundError as e:
            remote_content = None
            if raise_on_missing:
                raise e
        return remote_content

    def push(self, local_content):
        """
        Push content to blob.

        Parameters
        ----------
        local_content : str-like
            ``str`` or ``str``-like stream (e.g. :class:`io.StringIO`)
        """
        self._blob_client.upload_blob(local_content, overwrite=True)


class PersistentJSON(MutableMapping):
    """
    Persistent JSON.

    Push/pull a JSON object stored persistently in a "remote" location.
    This class is usefull when loading configurations or keeping persistent
    state.

    The class behaves exactly like a ``dict`` but only accepts values that are
    JSON encodable.

    Parameters
    ----------
    handler : object
        Handler extended from :class:`~fourinsight.engineroom.utils.core.BaseHandler`.
        Default handler is :class:`NullHandler`, which does not provide any
        push or pull functionality.
    """

    def __init__(self, handler=None):
        self.__dict = {}
        self._jsonencoder = json.JSONEncoder().encode
        self._handler = handler or NullHandler()

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
    Using a handler, the results can be *pushed or *pulled* from a remote source.

    Parameters
    ----------
    headers : dict
        Header names and data types as key/value pairs; ``int``, ``float``, and
        ``str`` are allowed as data types. The collector will only accept
        intermediate results defined here.
    handler: object
        Handler extended from :class:`~fourinsight.engineroom.utils.core.BaseHandler`.
        Default handler is :class:`NullHandler`, which does not provide any
        push or pull functionality.
    indexing_mode : str
        Indexing mode. Should be 'auto' or 'timestamp'.

    Notes
    -----
    The data types are casted to Pandas equivalent data types, so that missing
    values are handled correctly.
    """

    _DTYPES_MAP = {int: "Int64", float: "float64", str: "string"}

    def __init__(self, headers, handler=None, indexing_mode="auto"):
        if not set(self._DTYPES_MAP).issuperset(headers.values()):
            raise ValueError("Only 'int', 'float', and 'str' dtypes are supported.")

        self._headers = {
            header: self._DTYPES_MAP[dtype_] for header, dtype_ in headers.items()
        }
        self._indexing_mode = indexing_mode.lower()

        self._handler = handler or NullHandler()

        if self._indexing_mode == "auto":
            self._ignore_index = True
        elif self._indexing_mode == "timestamp":
            self._ignore_index = False
        else:
            raise ValueError("Indexing mode must be 'auto' or 'timestamp'.")

        self._dataframe = pd.DataFrame(columns=headers.keys()).astype(self._headers)

    def __repr__(self):
        return repr(self._dataframe)

    def new_row(self, index=None):
        """
        Make a new row.

        Parameters
        ----------
        index : None or datetime-like
            The new index value. If indexing_mode is set to 'auto', index
            should be ``None``. If indexing_mode is set to 'timestamp', index
            should be a unique datetime that is passed on to
            :func:`pandas.to_datetime`.
        """
        if self._indexing_mode == "auto" and index is not None:
            raise ValueError(
                "'indexing_mode' is set to 'auto'. " "Only 'index=None' is allowed."
            )
        elif self._indexing_mode == "timestamp" and index is None:
            raise ValueError(
                "'indexing_mode' is set to 'timestamp'. " "'index=None' is not allowed."
            )
        else:
            index = pd.to_datetime(index, utc=True)

            if index in (self._dataframe.index):
                raise ValueError("Index already exists.")

        row_new = pd.DataFrame(
            {header: None for header in self._headers}, index=[index]
        ).astype(self._headers)
        self._dataframe = self._dataframe.append(
            row_new, verify_integrity=True, ignore_index=self._ignore_index, sort=False
        )

    def collect(self, **results):
        """
        Collect and store results under the current index.

        Parameters
        ----------
        results : keyword arguments
            The results are passed as keyword arguments, where the keyword must
            be one of the 'headers'. Provided values must be of correct data
            type (defined during instantiation).
        """
        if not set(self._headers.keys()).issuperset(results):
            raise KeyError("Keyword must be in headers.")

        current_index = self._dataframe.index[-1]

        try:
            row_update = pd.DataFrame(data=results, index=[current_index],).astype(
                {
                    header: dtype_
                    for header, dtype_ in self._headers.items()
                    if header in results
                }
            )
        except ValueError:
            raise ValueError("Unable to cast 'results' to correct dtype")
        self._dataframe.loc[current_index, list(results.keys())] = row_update.iloc[0]

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

        df = pd.read_csv(
            StringIO(dataframe_csv), index_col=0, parse_dates=True, dtype=self._headers
        )

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
        """Return a (deep) copy of the internal dataframe"""
        return self._dataframe.copy(deep=True)
