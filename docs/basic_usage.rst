Basic Usage
===========

State
-----

Sometimes it is useful to be able to remember the current state of your Python
application. Using the ``PersistentJSON`` class and an appropriate file handler,
key state parameters can be stored persistently at a remote location, and be available
next time the application runs.

.. note::
    File handlers provide an interface to store and load data from text files.
    Custom handlers need to inherit from ``BaseHandler``. In the Cookbook section
    there is an example on how you can set up a file handler based on FTP.

The ``LocalFileHandler`` is used to store the state in a local json-file.

.. code-block:: python

    from fourinsight.engineroom.utils import LocalFileHandler

    handler = LocalFileHandler(<file-path>)

The ``AzureBlobHandler`` can be used to store the state in Azure Blob Storage.

.. code-block:: python

    from fourinsight.engineroom.utils import AzureBlobHandler


    handler = AzureBlobHandler(<connection-string>, <container-name>, <blob-name>)

The ``PersistentJSON`` class keep track of state parameters in key:value pairs.
PersistentJSON behaves similar as dictionaries, but also includes a ``push``
and a ``pull`` method for interaction with the remote source.

.. code-block:: python

    from fourinsight.engineroom.utils import PersistentJSON

    # During initiation, the handler 'pulls' the latest state from the remote source
    state = PersistentJSON(handler)

Values can be updated,

.. code-block:: python

    new_state = {"state parameter #1": 0, "State parameter #2": "some value"}
    state.update(new_state)

And retrieved,

.. code-block:: python

    state.get("state parameter #1")

To store the state for later, you simply just update the remote source with a ``push``.

.. code-block:: python

    # Update remote source
    state.push()

Then, the state is available next time you run your script by doing a ``pull``.

.. code-block:: python

    # Update state from remote source
    state.pull()
