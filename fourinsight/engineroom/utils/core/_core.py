__all__ = ["PersistentJSON", "PersistentDict", "ResultCollector"]


import json
from collections.abc import MutableMapping

import pandas as pd

from ._handlers import BaseHandler, NullHandler


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
