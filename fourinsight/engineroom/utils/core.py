import json
from abc import abstractmethod
from collections.abc import MutableMapping
from io import BytesIO, TextIOWrapper
from pathlib import Path

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobClient


class BaseHandler(TextIOWrapper):
    """
    Abstract class for push/pull text content from a remote/persistent source.

    The class inherits from ``io.TextIOWrapper``, and will behave like a stream.

    Parameters
    ----------
    *args : tuple
        Passed on to the TextIOWrapper's constructor.
    **kwargs : dict, optional
        Passed on to the TextIOWrapper's constructor.
    """

    _SOURCE_NOT_FOUND_ERROR = Exception

    def __init__(self, *args, **kwargs):
        super().__init__(BytesIO(), *args, **kwargs)

    def pull(self, raise_on_missing=True):
        """
        Pull text content from source, and overwrite the original content of the
        stream.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if content can not be pulled from source.
        """
        current_pos = self.tell()
        self.seek(0)
        try:
            characters_written = self._pull()
        except self._SOURCE_NOT_FOUND_ERROR as e:
            if raise_on_missing:
                self.seek(current_pos)
                raise e
            else:
                self.truncate(0)
        else:
            self.truncate(characters_written)

    def push(self):
        """
        Push text content to source.
        """
        self._push()

    @abstractmethod
    def _pull(self):
        """
        Pull text content from source, and write the string to stream.

        Returns
        -------
        int
            Number of characters written to stream (which is always equal to the
            length of the string).
        """
        raise NotImplementedError()

    @abstractmethod
    def _push(self):
        """
        Push the stream content to source.
        """
        raise NotImplementedError()

    def getvalue(self):
        """
        Get all stream content (without changing the stream position).

        Returns
        -------
        str
            Retrieve the entire content of the object.
        """
        self.flush()
        return self.buffer.getvalue().decode(self.encoding)


class NullHandler(BaseHandler):
    """
    Goes nowhere, does nothing. This handler is intended for objects that
    requires a handler, but the push/pull functionality is not needed.

    Will raise an exception if :meth:`push()` or :meth:`pull()` is called.
    """

    _ERROR_MSG = "The 'NullHandler' does not provide push/pull functionality."
    _SOURCE_NOT_FOUND_ERROR = NotImplementedError

    def __repr__(self):
        return "NullHandler"

    def _pull(self):
        raise NotImplementedError(self._ERROR_MSG)

    def _push(self):
        raise NotImplementedError(self._ERROR_MSG)


class LocalFileHandler(BaseHandler):
    """
    Handler for push/pull text content to/from a local file.

    Inherits from ``io.TextIOWrapper``, and behaves like a stream.

    Parameters
    ----------
    path : str or path object
        File path.
    encoding : str
        The name of the encoding that the stream will be decoded or encoded with.
        Defaults to 'utf-8'.
    newline : str
        Controls how line endings are handled. Will be passed on to TextIOWrapper's
        constructor.
    """

    _SOURCE_NOT_FOUND_ERROR = FileNotFoundError

    def __init__(self, path, encoding="utf-8", newline="\n"):
        self._path = Path(path)
        super().__init__(encoding=encoding, newline=newline)

    def __repr__(self):
        return f"LocalFileHandler {self._path.resolve()}"

    def _pull(self):
        return self.write(open(self._path, mode="r").read())

    def _push(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, mode="w") as f:
            f.write(self.getvalue())


class AzureBlobHandler(BaseHandler):
    """
    Handler for push/pull text content to/from Azure Blob Storage.

    Inherits from ``io.TextIOWrapper``, and behaves like a stream.

    Parameters
    ----------
    conn_str : str
        A connection string to an Azure Storage account.
    container_name : str
        The container name for the blob.
    blob_name : str
        The name of the blob with which to interact.
    encoding : str
        The name of the encoding that the stream will be decoded or encoded with.
        Defaults to 'utf-8'.
    newline : str
        Controls how line endings are handled. Will be passed on to TextIOWrapper's
        constructor.
    """

    _SOURCE_NOT_FOUND_ERROR = ResourceNotFoundError

    def __init__(
        self, conn_str, container_name, blob_name, encoding="utf-8", newline="\n"
    ):
        self._conn_str = conn_str
        self._container_name = container_name
        self._blob_name = blob_name
        self._blob_client = BlobClient.from_connection_string(
            self._conn_str, self._container_name, self._blob_name
        )
        super().__init__(encoding=encoding, newline=newline)

    def __repr__(self):
        return f"AzureBlobHandler {self._container_name}/{self._blob_name}"

    def _pull(self):
        return self._blob_client.download_blob().readinto(self.buffer)

    def _push(self):
        self._blob_client.upload_blob(self.getvalue(), overwrite=True)


class PersistentDict(MutableMapping):
    """
    Persistent :func:``dict``.

    Push/pull a :func:``dict`` stored persistently in a "remote" location as
    JSON. This class is usefull when loading configurations or keeping persistent
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
        self._handler.pull(raise_on_missing=raise_on_missing)
        remote_content = self._handler.getvalue()
        if not remote_content:
            remote_content = "{}"
        self.__dict.update(json.loads(remote_content))

    def push(self):
        """
        Push content to source.
        """
        self._handler.seek(0)
        self._handler.truncate()
        json.dump(self.__dict, self._handler, indent=4)
        self._handler.push()


# TODO: Remove after 2021-12-31
class PersistentJSON(PersistentDict):
    """
    DEPRECATED, use :class:`PersistentDict` instead. Will stop working
    after 2021-12-31.
    """

    def __init__(self, *args, **kwargs):
        import datetime
        import warnings

        warnings.warn(
            "DEPRECATED, use :class:`PersistentDict` instead. Will stop working"
            "after 2021-12-31.",
            FutureWarning,
        )
        if datetime.date.today() > datetime.date(2021, 12, 31):
            raise FutureWarning(
                "DEPRECATED, use :class:`PersistentDict` instead. Will stop working"
                "after 2021-12-31."
            )
        super().__init__(*args, **kwargs)


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
        Indexing mode. Should be 'auto' (default) or 'timestamp'.

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

        self._dataframe = pd.concat(
            [self._dataframe, row_new],
            verify_integrity=True,
            ignore_index=self._ignore_index,
            sort=False,
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

        # hotfix - pandas bug when setting string
        # https://github.com/pandas-dev/pandas/issues/44103
        if len(row_update.columns) == 1:
            value_update = row_update.iloc[0].values[0]
        else:
            value_update = row_update.iloc[0]
        self._dataframe.loc[current_index, list(results.keys())] = value_update

    def append(self, dataframe):
        """
        Append rows of `dataframe` to the results.

        Columns of `dataframe` must be in the headers.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The results to append.
        """

        for row_i, result_i in dataframe.to_dict(orient="index").items():
            if self._indexing_mode == "auto":
                row_i = None
            self.new_row(row_i)
            self.collect(**result_i)

    def pull(self, raise_on_missing=True):
        """
        Pull results from source. Remote source overwrites existing values.

        Parameters
        ----------
        raise_on_missing : bool
            Raise exception if results can not be pulled from source.
        """

        self._handler.pull(raise_on_missing=raise_on_missing)
        if not self._handler.getvalue():
            return

        self._handler.seek(0)
        df = pd.read_csv(
            self._handler, index_col=0, parse_dates=True, dtype=self._headers
        )

        if not (set(df.columns) == set(self._headers.keys())):
            raise ValueError("Header is not valid.")

        if (
            not df.index.empty
            and (self._indexing_mode == "auto")
            and not (df.index.dtype == "int64")
        ):
            raise ValueError("Index dtype must be 'int64'.")
        elif (
            not df.index.empty
            and (self._indexing_mode == "timestamp")
            and not (isinstance(df.index, pd.DatetimeIndex))
        ):
            raise ValueError("Index must be 'DatetimeIndex'.")

        self._dataframe = df

    def push(self):
        """
        Push results to source.
        """
        self._handler.seek(0)
        self._handler.truncate()
        self._dataframe.to_csv(self._handler, sep=",", index=True, line_terminator="\n")
        self._handler.push()

    @property
    def dataframe(self):
        """Return a (deep) copy of the internal dataframe"""
        return self._dataframe.copy(deep=True)

    def delete_rows(self, index):
        """
        Delete rows.

        The index will be reset if 'indexing_mode' is set to 'auto'.

        Parameters
        ----------
        index : single label or list-like
            Index labels to drop.
        """
        self._dataframe = self._dataframe.drop(index=index)

        if self._indexing_mode == "auto":
            self._dataframe = self._dataframe.reset_index(drop=True)

    def truncate(self, before=None, after=None):
        """
        Truncate results by deleting rows before and/or after given index values.

        The index will be reset if 'indexing_mode' is set to 'auto'.

        Parameters
        ----------
        before : int or datetime-like, optional
            Delete results with index smaller than this value.
        after : int or datetime-like, optional
            Delete results with index greater than this value.
        """
        index_drop = []
        if before:
            index_drop.extend(self._dataframe.index[(self._dataframe.index < before)])
        if after:
            index_drop.extend(self._dataframe.index[(self._dataframe.index > after)])
        if index_drop:
            self.delete_rows(index_drop)
