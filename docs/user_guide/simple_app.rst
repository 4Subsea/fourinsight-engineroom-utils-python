.. _simple-application:

Hello World!
============
This tutorial is a walkthrough on creating a simple Python application for `4Insight EngineRoom`_.
It will show how to structure, package, and upload your application.


Project Structure
-----------------

We will set up a simple Python application named `example_app`. The application will
import two packages, :mod:`numpy` and :mod:`pandas`, and print ``Hello, World!``.

In general, *EngineRoom* only requires 2 files: `run.py` and `requirements.txt`. However, it
may be helpful to structure your project locally as:

::

    example_app/
    ├── README.md
    └── src/
        ├── run.py
        └── requirements.txt

The above folder structure is not a requirement, but may be considered as a best practice.
The application content in placed under `src` while other supporting files are placed in root.

`run.py` is the Python file that is executed by *EngineRoom*. Open this file and enter
the following content:

.. code-block:: python

    import numpy
    import pandas


    print("Hello, World!")

Note that we have not used any of the utilities available in :mod:`fourinsight.engineroom.utils`. This is
intentional as the purpose of this tutorial is to show how to set up a bare-minimum application that will
run in *EngineRoom*. A more extensive :ref:`example <runpy-example>` with all the bells and whistles is introduced
together with the :ref:`concepts and utilities <concepts-utilities>`.

The `requirements.txt` specifies which Python packages are required to run the application,
i.e., so-called dependencies. Open this file and enter the following content:

::

    numpy
    pandas

The `README.md` file may contain a description of the application and information
on how to use it. This file is not required by *EngineRoom*, but it is considered
good practice to include it in the project folder. Open the file and enter the following
content:

::

    # Example Application

    This is a simple example application.


Packaging
---------

For this simple application, `run.py` and `requirements.txt` are all we need! Make a zip-file of the
content inside the `src` folder.

::

    example_app.zip/
    ├── run.py
    └── requirements.txt


Upload
------

Go to `<https://4insight.io/#/engineroom>`_, and upload the zip-file to your *EngineRoom*
application. Done!

.. _4Insight EngineRoom: https://4insight.io/#/engineroom
