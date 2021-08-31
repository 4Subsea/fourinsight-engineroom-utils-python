Basic Usage
===========

Handlers
--------

``LocalFileHandler`` and ``AzureBlobHandler`` provide an interface to store and
load text content to/from a source.

.. note::
    Custom handlers need to inherit from ``BaseHandler``. In the Cookbook section
    there is an :ref:`example<example_custom_handler_ftp>` on how you can set up a
    handler based on FTP.

The ``LocalFileHandler`` is used to store text in a local file.

.. code-block:: python

    from fourinsight.engineroom.utils import LocalFileHandler


    handler = LocalFileHandler(<file-path>)

The ``AzureBlobHandler`` can be used to store text in *Azure Blob Storage*.

.. code-block:: python

    from fourinsight.engineroom.utils import AzureBlobHandler


    handler = AzureBlobHandler(<connection-string>, <container-name>, <blob-name>)


State
-----

Sometimes it is useful to be able to remember the current 'state' of your Python
application. Using the ``PersistentJSON`` class and an appropriate handler, key
state parameters can be stored persistently at a remote location, and be available
next time the application runs.

``PersistentJSON`` behaves similar to dictionaries, and can keep track of state
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

.. warning::
    The ``push`` method will overwrite the content of the remote source.


Collect and store results
-------------------------
The ``ResultCollector`` is a useful tool when you want to collect and store results.
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

    # access the results as a 'pandas.DataFrame'
    df = results.dataframe


If you are dealing with time-dependent results, and want to 'stamp' the results
with a datetime value, this can be done by setting 'indexing_mode' to 'timestamp'
during initialization. Then, a datetime value must be passed to ``new_row()`` when
collecting results.

.. code-block:: python

    headers = {"a": float, "b": str}
    results = ResultCollector(headers, indexing_mode="timestamp")

    # stamp the results with a datetime value
    results.new_row("2020-01-01 00:00")

    # and collect your results
    results.collect(a=1.0)


By initializing with a suitable handler, results can be pushed and pulled from a
source.

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
