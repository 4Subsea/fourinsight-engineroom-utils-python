.. _runpy-example:

Example
=======

Here is an example of `run.py` that use some of the utilities in :mod:`fourinsight.engineroom.utils`.

In this example, two timeseries 'A' and 'B' are downloaded from the *DataReservoir.io*.
Another variable, 'C', is then calculated from 'A' and 'B'. Then, the 1-hour standard deviation
of each variable is calculated, collected, and stored in *Azure Blob Storage*.

A :class:`~fourinsight.engineroom.utils.DrioDataSource` object is used to download the data group, 'A' and 'B',
from the *DataReservoir.io*. The data is downloaded in 1-hour chunks using the :meth:`~fourinsight.engineroom.utils.DrioDataSource.iter()`
method and the :meth:`~fourinsight.engineroom.utils.iter_index.date_range()` convenience function.
Caching is enabled by providing a `cache` folder (and an appropriate `cache_size` if you don't want the default 24-hours caching).

:class:`~fourinsight.engineroom.utils.PersistentDict` is used to keep track of the state parameter, 'TimeOfLastIndex'.
This state parameter tells the script which results it has already collected, so that the
script can continue where it left off last time it ran. The state is stored persistently
in a local file using the :class:`~fourinsight.engineroom.utils.LocalFileHandler`.

The :class:`~fourinsight.engineroom.utils.ResultCollector` provides collecting and storing of results. Results
are stored in *Azure Blob Storage* using the :class:`~fourinsight.engineroom.utils.AzureBlobHandler`.

.. code-block:: python

    import os

    import numpy as np
    import pandas as pd

    import datareservoirio as drio
    from datareservoirio.authenticate import ClientAuthenticator
    from fourinsight.engineroom.utils import (
        DrioDataSource,
        PersistentDict,
        ResultCollector,
        LocalFileHandler,
        AzureBlobHandler,
        iter_index,
    )


    auth = ClientAuthenticator(os.environ["APP_CLIENT_ID"], os.environ["APP_CLIENT_SECRET"])
    drio_client = drio.Client(auth)

    # Initialize a data source object with labels 'A' and 'B'
    data_labels = {
        "A": "8b1683bb-32a9-4e64-b122-6a0534eff592",
        "B": "4bf4606b-b18e-408d-9d4d-3f1465ed23f2"
    }
    source = DrioDataSource(drio_client, data_labels, cache='.cache')

    # Get the application state
    state_handler = LocalFileHandler("state.json")
    state = PersistentDict(state_handler)
    state.pull(raise_on_missing=False)

    # Initialize a ResultCollector
    results_handler = AzureBlobHandler(
        os.environ["APP_CONNECTION_STRING"],
        "example_container",
        "example_blob_folder/results.csv"
    )
    result_headers = {"a_std": float, "b_std": float, "c_std": float}
    results = ResultCollector(result_headers, handler=result_handler, indexing_mode="timestamp")
    results.pull()   # 'pull' already collected results from source

     # Start from '2021-09-28 00:00' and end 'now'
     # If the app has already run previously, start from last collected index
    start = state.get("TimeOfLastIndex", default="2021-09-28 00:00")
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime("now", utc=True)

    # Iterate over the data in 1-hour chunks
    for index_i, data_i in source.iter(*iter_index.date_range(start, end, freq="1h")):
        results.new_row(index_i)

        series_a = data_i["A"]
        series_b = data_i["B"]

        # Collect the standard deviation of timeseries A and B
        results.collect(a_std=np.std(series_a), b_std=np.std(series_b))

        # Do some calculations with your timeseries data
        series_a = series_a + np.random.random(size=len(series_a))
        series_b = series_b - 1.0
        series_c = (series_a + series_b) / 2.0

        # Collect the standard deviation of the calculated variable C
        results.collect(c_std=np.std(series_c))

    # Store the results
    results.push()

    # Update the application state wih the latest collected index
    state["TimeOfLastIndex"] = results.dataframe.index[-1].isoformat()
    state.push()

.. note: `APP_CLIENT_ID`, `APP_CLIENT_SECRET` and `APP_CONNECTION_STRING`
         are retrieved as environmental variables. See :ref:`separate-config` for more
         details.