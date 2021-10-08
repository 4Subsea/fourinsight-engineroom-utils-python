Collect and store results
=========================
The :class:`~fourinsight.engineroom.utils.ResultCollector` is a useful tool when you want to collect and store results.
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
during initialization. Then, a datetime value must be passed to :meth:`~fourinsight.engineroom.utils.ResultCollector.new_row()` when
collecting results.

.. code-block:: python

    headers = {"a": float, "b": str}
    results = ResultCollector(headers, indexing_mode="timestamp")

    # stamp the results with a datetime value
    results.new_row("2020-01-01 00:00")

    # and collect your results
    results.collect(a=1.0)


By initializing with a suitable :ref:`handler<text_content_handlers>`, results can
be 'pushed' and 'pulled' from a source.

.. code-block:: python

    # pull the results from an existing source
    results.pull()

    # update the source with the latest results
    results.push()
