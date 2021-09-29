Data Source
===========

Timeseries data (or other types of sequential data) are often most valueable when
they are considered in groups. Insight is usually found by investigating the relationship
between different state variables of a system. An example could be measurements from
a motion sensor; to be able to calcuate the tilt angle of the sensor, you would need
access to acceleration and gyro measurements for all three axis of the sensor. Another
example could be parameterized wave spectrum data, where the spectrum is only fully
described when you have all parameters available.

'Data source' objects provide an interface to retrieve groups of sequential data
from a source. A data source class must inherit from
:class:`~fourinsight.engineroom.utils.datamanage.BaseDataSource`, and override the abstract
method, ``_get()``, and the abstract property, ``labels``.

The :class:`DrioDataSource` class handles data from the DataReservoir.io. It is
initialized with a :class:`datareservoirio.Client` object and a dictionary containing
labels and timeseries IDs as key/value pairs.

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
then merged together at a 'common' index. The common index will be the smallest
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


Download data
-------------

The ``get()`` method is used to download data from the source between two index values.

.. code-block:: python

    # download data as a 'pandas.DataFrame'
    df = source.get("2020-01-01 00:00", "2020-01-02 00:00")

The ``iter()`` method is used to iterate over 'chunks' of data. Lists of start and
end indexes are required as input.

.. code-block:: python

    start = ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]
    end = ["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]

    for index_i, data_i in source.iter(start, end):
        pass


Iterator 'start' and 'end' indexes
..................................

Convenience functions for generating iterator start and end indexes are available in the
:mod:`iter_index` module. For example, for timeseries data where the index is datetime-like,
fixed-frequency start and end index pairs can be generated with ``iter_index.date_range()``.

.. code-block:: python

    from fourinsight.engineroom.utils import iter_index


    start, end = iter_index.date_range(
        start="2020-01-01 00:00", end="2020-02-01 00:00", freq="1H"
    )

    for index_i, data_i in source.iter(start, end):
        pass