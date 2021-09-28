Make a simple EngineRoom application
====================================

This tutorial walks you through how to set up a simple EngineRoom application. It
will show you how to add the necessary files and structure, how to make a zip-file
of the application content, and how to upload it to EngineRoom.

A simple application
--------------------

We will set up a simple application named `example_app`. The application will import
two packages, :mod:`numpy` and :mod:`pandas`, and print the text, "Hello, World!".

Create the following folder structure locally:

::

    example_app/
    ├── README.md
    └── src/
        ├── run.py
        └── requirements.txt

`run.py` is the Python file that will be run by EngineRoom. Open this file and enter
the following content:

.. code-block:: python

    import numpy
    import pandas


    print("Hello, World!")

The `requirements.txt` file includes all the application dependencies. Open this file
and enter the following content:

::

    numpy
    pandas

EngineRoom applications does not require a `README.md` file. But we recommend to
include it anyway, and provide a description of the application and how it should
be used. Open the file and enter the following content:

::

    # Example Application

    This is a simple example application.


Make a zip-file of the application content
------------------------------------------

Make a zip-file of the content inside the `src` folder. All that EngineRoom needs to
run an application, is a `run.py` file with the Python script to run and a `requirements.txt`
file listing all required packages.

::

    example_app_zip/
    ├── run.py
    └── requirements.txt


Upload the application to EngineRoom
------------------------------------

Upload the zip file to EngineRoom.
