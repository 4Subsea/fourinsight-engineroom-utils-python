Collect and store results
=========================
Aapplication results can be stored inside a dedicated folder, so that the application output is easily available after app execution

Application results can be any set of files (or files hierarchy), that is produced by your application, for example output CSV files
containing processed data. Results are moved to a permanent store after every execution, regardless of exit code.

Output files are appended to the result set from previous execution. If your application creates a file with the same name, it will
override the previous result file. If you need to append to a file from a previous execution, the file must be loaded into the 
application first. This can be achieved using :meth:`~fourinsight.engineroom.utils.load_previous_engineroom_results`, which will
load the results from the permanent store into the application.


.. code-block:: python

    from fourinsight.api import UserSession
    from fourinsight.engineroom.utils import load_previous_engineroom_results

    session = UserSession()

    #download all available results
    load_previous_engineroom_results(ENGINE_ROOM_APP_ID, session, download_all=True)

    #download specific results file
    load_previous_engineroom_results(ENGINE_ROOM_APP_ID, session, path="config.json")


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

    # collect a dataframe of results
    df = pd.DataFrame(data={"a": [2, 5], "b": ["txt", "even more text"]})
    results.append(df)

    # return the results as a 'pandas.DataFrame'
    df = results.dataframe

It is also possible to delete previously collected results with
:meth:`~fourinsight.engineroom.utils.ResultCollector.delete_rows()`
and :meth:`~fourinsight.engineroom.utils.ResultCollector.truncate()`.

If you are dealing with time-dependent results, and want to 'stamp' the results
with a datetime value, this is facilitated by setting 'indexing_mode' to 'timestamp'
during initialization. Then, a datetime value must be passed to :meth:`~fourinsight.engineroom.utils.ResultCollector.new_row()` when
collecting results. For :meth:`~fourinsight.engineroom.utils.ResultCollector.append()`, the indices of the dataframe must be a datetime.

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
