.. _simple-application:

Make a 'simple' EngineRoom application
======================================

This tutorial walks you through how to set up a simple EngineRoom Python application.
It will show you how to add the necessary files and structure, how to make a zip-file
of the application content, and how to upload it to EngineRoom.

A simple application
--------------------

We will set up a simple Python application named `example_app`. The application will
import two packages, :mod:`numpy` and :mod:`pandas`, and print the text, "Hello, World!".

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

The `requirements.txt` specifies which Python packages are required to run the application.
Open this file and enter the following content:

::

    numpy
    pandas

The `README.md` file should contain a description of the application and information
on how to use it. This file is not required by EngineRoom, but it is considered
good practice to include it in the project folder. Open the file and enter the following
content:

::

    # Example Application

    This is a simple example application.


Create a zip-file of the application content
--------------------------------------------

EngineRoom only requires two files to run a Python application, i.e., the `run.py`
file and the `requirements.txt` file. For this simple application, these two files
are sufficient. For more complex applications, it may be necessary to include other
supporting files.

Make a zip-file of the content inside the `src` folder.

::

    example_app.zip/
    ├── run.py
    └── requirements.txt


Upload the application to EngineRoom
------------------------------------

Go to `<https://4insight.io/#/engineroom>`_, and upload the zip-file to your EngineRoom
application.
