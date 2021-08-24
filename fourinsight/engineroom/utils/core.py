from abc import ABC, abstractmethod
import json
from collections.abc import MutableMapping


class BaseHandler(ABC):
    """
    Abstract class for push/pull file content from a remote/persistent source.
    """
    @abstractmethod
    def pull(self):
        raise NotImplementedError()

    @abstractmethod
    def push(self, local_dict):
        raise NotImplementedError()


class LocalFileHandler(BaseHandler):
    def __init__(self, path):
        self._path = path

    def pull(self):
        try:
            remote_content = open(self._path, mode="r").read()
        except FileNotFoundError:
            remote_content = None
        return remote_content

    def push(self, local_content):
        with open(self._path, mode="w") as f:
            f.write(local_content)


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
        local_content = json.dumps(self.__dict)
        self._handler.push(local_content)
