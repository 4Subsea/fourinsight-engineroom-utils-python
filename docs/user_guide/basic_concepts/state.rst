State
=====

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

    new_state = {"state parameter #1": 0, "state parameter #2": "some text value"}
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