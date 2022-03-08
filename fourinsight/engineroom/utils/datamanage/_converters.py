__all__ = [
    "BaseIndexConverter",
    "IntegerIndexConverter",
    "FloatIndexConverter",
    "DatetimeIndexConverter",
]

from abc import ABC, abstractmethod, abstractproperty

import numpy as np
import pandas as pd


class BaseIndexConverter(ABC):
    """
    Abstract class for 'index converters'.
    """

    @abstractmethod
    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        raise NotImplementedError()

    @abstractmethod
    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        raise NotImplementedError()

    def to_native_index(self, index):
        """Convert index to 'native' type"""
        return self.to_universal_index(index)

    @abstractproperty
    def reference(self):
        """Index reference"""
        raise NotImplementedError()

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError()

    @classmethod
    def __eq__(cls, other):
        return isinstance(other, cls)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__repr__())


class DatetimeIndexConverter(BaseIndexConverter):
    """
    Datetime index converter.

    Index values will be passed on to meth:`pandas.to_datetime` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return pd.to_datetime(index, utc=True)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        return pd.to_timedelta(delta)

    @property
    def reference(self):
        """Index reference"""
        return pd.to_datetime(0, utc=True)

    def __repr__(self):
        return "DatetimeIndexConverter"


class IntegerIndexConverter(BaseIndexConverter):
    """
    Integer index converter.

    Index values will be passed on to meth:`numpy.int64` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return np.int64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        return np.int64(delta)

    @property
    def reference(self):
        """Index reference"""
        return 0

    def __repr__(self):
        return "IntegerIndexConverter"


class FloatIndexConverter(BaseIndexConverter):
    """
    Float index converter.

    Index values will be passed on to meth:`numpy.float64` function to convert
    to universal type.
    """

    def to_universal_index(self, index):
        """Convert index to 'universal' type"""
        return np.float64(index)

    def to_universal_delta(self, delta):
        """Convert index partition to 'universal' type"""
        return np.float64(delta)

    @property
    def reference(self):
        """Index reference"""
        return 0.0

    def __repr__(self):
        return "FloatIndexConverter"
