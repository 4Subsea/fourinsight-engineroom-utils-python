Data source
===========

Timeseries data (or other types of sequential data) are often considered in "groups", since
many timeseries can share a common index and are likely to be used together.

.. note: A "group" can only have a single member, and still benefit from
        the utilities described below.

'Data source' objects provide an interface to retrieve groups of sequential data that share a common index
from a source. The primary purpose is to have a common interface for handling timeseries (or sequential) data
regardless of the source.

:mod:`fourinsight.engineroom.utils` comes with the following built-in data sources:

:class:`~fourinsight.engineroom.utils.DrioDataSource`
    Handles data from DataReservoir.io_.

:class:`~fourinsight.engineroom.utils.CompositeDataSource`
    Handles data from multiple data sources but appears as a single data source. Useful when it is
    necessary to switch between different sources for different index intervals. For instance,
    seamlessly switch between data sources based on time periods.

.. _DataReservoir.io: https://www.datareservoir.io/

We aim to add other popular data sources as part of :mod:`fourinsight.engineroom.utils`.

However, it is possible (and encouraged) to define custom data sources. A data source class must
inherit from :class:`~fourinsight.engineroom.utils.BaseDataSource`, and override the
abstract method, :meth:`~fourinsight.engineroom.utils.BaseDataSource._get()`,
and the abstract properties, :attr:`~fourinsight.engineroom.utils.BaseDataSource.labels`
and :attr:`~fourinsight.engineroom.utils.BaseDataSource._fingerprint`.

The following code examples are given with :class:`~fourinsight.engineroom.utils.DrioDataSource`, but
all the functionality shown will be common for any data source class that inherits from :class:`~fourinsight.engineroom.utils.BaseDataSource`.
The only difference will be how the classes are initialized. For instance,
:class:`~fourinsight.engineroom.utils.DrioDataSource` is initialized with a
:class:`datareservoirio.Client` instance and a dictionary containing labels and timeseries
IDs as key/value pairs.

.. code-block:: python

    from fourinsight.engineroom.utils import DrioDataSource


    labels = {
        "Ax": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "Ay": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "Az": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "Gx": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "Gy": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "Gz": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    }

    source = DrioDataSource(drio_client, labels)

The data index can be synced during download by setting the ``index_sync`` flag
to ``True`` and providing a suitable ``tolerance`` limit. Neighboring datapoints are
then merged together at a 'common' index. The common index will be the first
index of the neighboring datapoints. The tolerance describe the expected spacing
between neighboring datapoints to merge.

.. code-block:: python

    source = DrioDataSource(
        drio_client,
        labels,
        index_sync=True,
        tolerance=pd.to_timedelta("1ms")
    )

.. warning::
    Be careful when setting the tolerance limit for synchronization. A too small
    or too large tolerance could lead to loss of data. The tolerance should at least
    be smaller than the sampling frequency of the data, and it shoud be greater than
    the expected jitter between datapoints to merge.

    The synchronization algorithm will make a common index by concatenating all
    the different label indexes, do a sorting, and then remove all index steps that are
    smaller than the tolerance. Datapoints are then merged into the common index
    if they are closer than the tolerance limit.

Caching is enabled by providing a `cache` folder and an appropriate `cache_size`.
Caching will significantly speed-up the data downloading if the same data is requested
multiple times. First time some data is retrieved from the source, it is split up in
'chunks' and stored in a local folder. Then, the data is more readily available
next time it is requested.

.. code-block:: python

    source = DrioDataSource(
        drio_client,
        labels,
        cache='.cache'
        cache_size=pd.to_timedelta("3h")
    )

.. tip::
    Due to how the caching routine is designed, it may be useful to set the `refresh_cache`
    flag to ``True`` first time data is downloaded. This will speed-up the first
    data download. After the first download, set the flag to ``False``, so that data
    is retrieved from the cache that was built during the first download.


Download data
-------------

The :meth:`~fourinsight.engineroom.utils.BaseDataSource.get()` method is used to download data from the source between two index values.

.. code-block:: python

    # download data as a 'pandas.DataFrame'
    df = source.get("2020-01-01 00:00", "2020-01-02 00:00")

Iterators
---------
The :meth:`~fourinsight.engineroom.utils.BaseDataSource.iter()` method is used to iterate over 'chunks' of data. Lists of start and
end indecies are required as input.

.. code-block:: python

    start = ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    end = ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]

    for index_i, data_i in source.iter(start, end):
        pass


Convenience functions for generating list of start and end indecies are available in the
:mod:`~fourinsight.engineroom.utils.iter_index` sub-module. For example, for timeseries data where
the index is datetime-like, fixed-frequency start and end index pairs can be generated with
:meth:`~fourinsight.engineroom.utils.iter_index.date_range()`.

.. code-block:: python

    from fourinsight.engineroom.utils import iter_index


    start, end = iter_index.date_range(
        start="2020-01-01 00:00", end="2020-02-01 00:00", freq="1h"
    )

    for index_i, data_i in source.iter(start, end):
        pass