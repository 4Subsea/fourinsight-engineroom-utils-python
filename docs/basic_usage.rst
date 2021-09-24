.. py:currentmodule:: fourinsight.engineroom.utils

Basic Usage
===========

Handlers
--------

.. _text_content_handlers:

'Push' and 'pull' text content from a source
............................................

Some of the core functionality provided by :mod:`fourinsight.engineroom.utils` relies
on handlers that facilitate downloading and uploading of text content from a source.
The source can be a local file, an Azure Storage Blob, or any other suitable storage
place. Two handlers, the :class:`LocalFileHandler` and the :class:`AzureBlobHandler`
are available out-of-the-box. Custom handlers are easily set up by inheriting from
:class:`~fourinsight.engineroom.utils.core.BaseHandler`.

.. note::
    In the Cookbook section there is an :ref:`example<example_custom_handler_ftp>`
    on how you can set up a custom handler based on FTP.

The :class:`LocalFileHandler` is used to store text content in a local file.

.. code-block:: python

    from fourinsight.engineroom.utils import LocalFileHandler


    handler = LocalFileHandler(<file-path>)

The :class:`AzureBlobHandler` is used to store text content in *Azure Blob Storage*.

.. code-block:: python

    from fourinsight.engineroom.utils import AzureBlobHandler


    handler = AzureBlobHandler(<connection-string>, <container-name>, <blob-name>)

The handlers behave like *streams*, and provide all the normal stream capabilities.

.. code-block:: python

    # Write text content to stream
    handler.write("Hello, World!")

    # Read stream content
    handler.read()

    # Write 'pandas.DataFrame' to stream
    df.to_csv(handler)

    # Load 'pandas.DataFrame' from stream
    df = pd.read_csv(handler, index_col=0)

    # etc...

In addition, downloading and uploading of the text content is provided by a push/pull
strategy; content is retrieved from the source by a ``pull()`` request, and uploaded
to the source by a ``push()``.


State
-----

Sometimes it is useful to remember the current 'state' of your Python application.
Using the :class:`PersistentJSON` class and an appropriate :ref:`handler<text_content_handlers>`,
key state parameters can be stored persistently at a remote location, and be available
next time the application runs.

:class:`PersistentJSON` behaves similar to dictionaries, and can keep track of state
parameters in key/value pairs.

.. code-block:: python

    from fourinsight.engineroom.utils import PersistentJSON


    state = PersistentJSON(handler)

Values can be updated,

.. code-block:: python

    new_state = {"state parameter #1": 0, "State parameter #2": "some value"}
    state.update(new_state)

And retrieved,

.. code-block:: python

    state.get("state parameter #1")

As well as deleted, printed, etc...

.. code-block:: python

    # Delete
    del state["state parameter #1]

    # Print
    print(state)

    # etc...

To store the state for later, you simply just update the source with a ``push()``.

.. code-block:: python

    # Update remote source
    state.push()

Then, the state is available next time you run your script by doing a ``pull()``.

.. code-block:: python

    # Update state from remote source
    state.pull()


Collect and store results
-------------------------
The :class:`ResultCollector` is a useful tool when you want to collect and store results.
The basic usage is illustrated with the examples below.

.. code-block:: python

    from fourinsight.engineroom.utils import ResultCollector


    headers = {"a": float, "b": str} # collect parameter 'a' as 'float' and 'b' as 'string'
    results = ResultCollector(headers)

    # make a new row
    results.new_row()

    # collect some results for that row
    results.collect(a=1.0, b="some text")

    # make another row
    results.new_row()

    # collect some results for the new row
    results.collect(a=1.5, b="some more text")

    # return the results as a 'pandas.DataFrame'
    df = results.dataframe


If you are dealing with time-dependent results, and want to 'stamp' the results
with a datetime value, this is facilitated by setting 'indexing_mode' to 'timestamp'
during initialization. Then, a datetime value must be passed to :meth:`.new_row()` when
collecting results.

.. code-block:: python

    headers = {"a": float, "b": str}
    results = ResultCollector(headers, indexing_mode="timestamp")

    # stamp the results with a datetime value
    results.new_row("2020-01-01 00:00")

    # and collect your results
    results.collect(a=1.0)


By initializing with a suitable :ref:`handler<text_content_handlers>`, results can
be pushed and pulled from a source.

.. code-block:: python

    headers = {"a": float, "b": str}
    results = ResultCollector(headers, handler=handler)

    # pull the results from an existing source
    results.pull()

    # make a new row
    results.new_row()

    # collect some more results
    results.collect(a=1.0)

    # update the source with the latest results
    results.push()


Data Source
-----------

Timeseries data (or other types of sequential data) are often most valueable when
they are considered in groups. Insight is usually found by investigating the relationship
between different state variables of a system. An example could be measurements from
a motion sensor; to be able to calcuate the tilt angle of the sensor, you would need
access to acceleration and gyro measurements for all three axis of the sensor. Another
example could be parameterized wave spectrum data, where the spectrum is only fully
described when you have all parameters available.

'Data source' objects provide an interface to retrieve groups of sequential data
from a source. Data source classes must inherit from :class:`BaseDataSource`, and
override the abstract method, ``_get()``, and the abstract property, ``labels``.

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

The ``get()`` method is used to download data from the source.

.. code-block:: python

    # download data as a 'pandas.DataFrame'
    df = source.get("2020-01-01 00:00", "2020-01-02 00:00")

The data index can be synced during download by setting the ``index_sync`` flag
to ``True`` and providing a suitable ``tolerance`` limit. Neighboring datapoints are
then merged together at a 'common' index. The common index will be the smallest
index of the neighboring datapoints. The tolerance describe the expected spacing
between neighboring datapoints to merge.

.. code-block:: python

    df = source.get(
        "2020-01-01 00:00",
        "2020-01-02 00:00",
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
