import json
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from pathlib import Path

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

        self.pull()

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
