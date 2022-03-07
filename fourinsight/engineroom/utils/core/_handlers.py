__all__ = ["BaseHandler", "NullHandler", "LocalFileHandler", "AzureBlobHandler"]

from abc import abstractmethod
from io import BytesIO, TextIOWrapper
from pathlib import Path

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