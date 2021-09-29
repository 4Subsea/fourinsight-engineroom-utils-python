Putting it all together
=======================

An advanced EngineRoom application
----------------------------------

Here is an example on how you could utilize some of the :ref:`basic concepts<basic-concepts>`
and utilities provided by :mod:`fourinsight.engineroom.utils` in your Python application.
The example script will download two timeseries, 'A' and 'B', from the DataReservoir.io.
Then it collects the standard deviation of the two signals and a calculated variable, 'C'.
In the example, :class:`PersistentJSON` is used to keep track of 'state' parameters,
:mod:`ResultCollector` provides collecting and storing of results, and the
:mod:`LocalFileHandler` facilitates 'pushing' and 'pulling' of text content to local files.

This example script could go into the `run.py` file of an EngineRoom application.
See the :ref:`simple application example<simple-application>` for details on how to set up your first EngineRoom application.

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd

    import datareservoirio as drio
    from datareservoirio.authenticate import ClientAuthenticator
    from fourinsight.engineroom.utils import (
        DrioDataSource,
        PersistentJSON,
        ResultCollector,
        LocalFileHandler,
        iter_index,
    )


    auth = ClientAuthenticator(os.environ["APP_CLIENT_ID"], os.environ["APP_CLIENT_SECRET"])
    drio_client = drio.Client(auth)

    data_labels = {
        "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
        "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2"
    }
    source = DrioDataSource(drio_client, data_labels)

    # get the application state
    state_handler = LocalFileHandler("state.json")
    state = PersistentJSON(state_handler)
    state.pull(raise_on_missing=False)

    # initialize a ResultCollector
    result_handler = LocalFileHandler("results.csv")
    result_headers = {"a_std": float, "b_std": float, "c_std": float}
    results = ResultCollector(result_headers, handler=result_handler, indexing_mode="timestamp")

    start = state.get("TimeOfLastSample", default="2021-09-28 00:00")
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime("now", utc=True)

    # iterate over the data as 1-hour chunks
    for index_i, data_i in source.iter(*iter_index.date_range(start, end, freq="1H")):
        results.new_row(index_i)

        series_a = data_i["A"]
        series_b = data_i["B"]

        # collect the standard deviation of timeseries A and B
        results.collect(a_std=np.std(series_a), b_std=np.std(series_b))

        # do some calculations with your timeseries data
        series_a = series_a + np.random.random(size=len(series_a))
        series_b = series_b - 1.0
        series_c = (series_a + series_b) / 2.0

        # collect the standard deviation of the calculated variable c
        results.collect(c_std=np.std(series_c))

    # store the results
    results.push()

    # update the application state 
    state["TimeOfLastSample"] = results.dataframe.index[-1].isoformat()
    state.push()

Store secret parameters as environment variables
................................................

Store secret parameters as environment variables in EngineRoom.



Going forward
-------------

This is an advanced application, where the provided tools are utilized.

::

    advanced_example_app/
    ├── README.md
    └── src/
        ├── .config/
        │   ├── data.json
        │   └── general.json
        ├── app/
        │   ├── __init__.py
        │   ├── module_a.py
        │   └── module_b.py
        ├── run.py
        └── requirements.txt

Put configuration parameters in separate files
..............................................

It is good practice to separate the application code and configuration parameters.
This ensures overview and easy altering of the configuration parameters.


Divide application into smaller sub-modules
...........................................

For more complex application, it can be useful to separate the application into
sub modules.
