.. _advanced-application:

Utilize the tools in an 'advanced' application
==============================================

An advanced EngineRoom application
----------------------------------

Here is an example on how you could utilize some of the :ref:`basic concepts<basic-concepts>`
and utilities provided by :mod:`fourinsight.engineroom.utils` in your Python application.
This example script will download two timeseries, 'A' and 'B', from the DataReservoir.io.
Another variable, 'C', is then calculated from signal 'A' and 'B'. The script will collect
the 1-hour standard deviation of each variable, and store it in Azure Blob Storage.

A :class:`~fourinsight.engineroom.utils.DrioDataSource` object is used to download the data group, 'A' and 'B',
from the DataReservoir.io. The data is downloaded in 1-hour chunks using the :meth:`~fourinsight.engineroom.utils.DrioDataSource.iter()`
method and the :meth:`~fourinsight.engineroom.utils.iter_index.date_range()` convenience function.

:class:`~fourinsight.engineroom.utils.PersistentJSON` is used to keep track of the state parameter, 'TimeOfLastIndex'.
This state parameter tells the script which results it has already collected, so that the
script can continue where it left off last time it ran. The state is stored persistently
in a local file using the :class:`~fourinsight.engineroom.utils.LocalFileHandler`.

The :class:`~fourinsight.engineroom.utils.ResultCollector` provides collecting and storing of results. Results
are stored in an Azure Storage Blob using the :class:`~fourinsight.engineroom.utils.AzureBlobHandler`.

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
    source = DrioDataSource(drio_client, data_labels)

    # Get the application state
    state_handler = LocalFileHandler("state.json")
    state = PersistentJSON(state_handler)
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
    for index_i, data_i in source.iter(*iter_index.date_range(start, end, freq="1H")):
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

This code could go into the `run.py` file of an EngineRoom application.
See the :ref:`simple application example<simple-application>` for details on how
to set up your first EngineRoom application.

Store secret parameters as environment variables
................................................

Secret parameters, that you do not want to expose to others, can be stored as environmental
variables in EngineRoom. In the example code above, three parameters, i.e., the
'APP_CLIENT_ID', the 'APP_CLIENT_SECRET' and the 'APP_CONNECTION_STRING', are
retrieved from the user's environmental variables.

.. tip::
    Environmental variables can be used to store other configuration parameters as well,
    even though they are not really secret. This way you can separate the configuration
    of your application from the code.

Going forward
-------------

The only files that EngineRoom really needs to run a Python application, is the
`run.py` file and the `requirements.txt` file. Going forward with more complex applications,
you may want to include some extra files in your application. Here is an example of
a more extensive folder structure of a more advanced application:

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
        ├── packages/
        │   └── private_package.whl
        ├── run.py
        └── requirements.txt

Divide application into smaller sub-modules
...........................................
In larger applications, it may be useful to divide the application into several
sub-modules. It is good practice to keep such sub-modules in an importable 'app'
module. Import these modules in your `run.py` file, and execute each sub-module
from there.

Separate the configuration from the code
........................................
It is good practice to separate the application code and the configuration parameters.
This ensures overview and easy altering of the configuration. One way to store
configuration parameters, is to keep them in json files and read these files in the
application code. Another way of storing configuration parameters, is to define
them as environmental variables in EngineRoom. Note that these are just two
examples of how to store configuration, there may be other ways that better fit
your purpose.

Include private packages as WHL files
.....................................
Sometimes your application requires Python packages that are not available through
PyPi. Such packages can be included in the application by pip-installable WHL files.
Remember to add these packages to the `requirements.txt` file:

::

    -f ./packages
    private_package

Finally, be creative and use the utilities you find useful to create your own Python application that creates insight!
......................................................................................................................
Don't let these guidelines be a showstopper when you start setting up your own EngineRoom application.
The utilities provided by :mod:`fourinsight.engineroom.utils` are just meant to
aid and speed-up the Python app development. If you don't find any of the utilities
and basic concepts useful, don't bother using them. EngineRoom is able to execute
any type of Python code - as long as you provide a `run.py` file and a `requirements.txt`
file.
