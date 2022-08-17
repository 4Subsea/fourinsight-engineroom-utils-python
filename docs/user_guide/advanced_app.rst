.. _advanced-application:

Advanced Applications
=====================

Here are more recommendatations and suggestion that may be useful as your application grows in capability
and complexity.

Project Structure
-----------------

A more comprehensive folder structure for advanced applications:

::

    advanced_example_app/
    ├── README.md
    ├── src/
    │   ├── .config/
    │   │   ├── config_a.json
    │   │   └── config_b.json
    │   ├── app/
    │   │   ├── __init__.py
    │   │   ├── module_a.py
    │   │   └── module_b.py
    │   ├── packages/
    │   │   └── private_package.whl
    │   ├── run.py
    │   └── requirements.txt
    └── tests/

Divide Application into Smaller Sub-modules
-------------------------------------------
In larger applications, it may be useful to divide the application into several
sub-modules. It is good practice to keep such sub-modules in an importable 'app'
module. Import these modules in your `run.py` file, and execute each sub-module
from there.

.. _separate-config:

Separate the Configuration from the Code
----------------------------------------
It is good practice to separate the application code and the configuration parameters.
This ensures overview and easy altering of the configuration.

One way to store configuration parameters, is to keep them in `JSON` files and
read these files in the application code. These files may be stored in a folder
under `src`, e.g. ``src/.config/config_a.json`` as suggested above.

Another way of storing configuration parameters, is to define
them as environmental variables, i.e. configure as `Variables` in *EngineRoom*.
Especially, secret parameters that you do not want to expose to others. Secrets
such as passwords and "connection strings" shall never be stored together with the
application code!

Note that these are just two examples of how to store configuration, but there may
be other ways that fit your purpose better.

Include Private Packages as `wheel` Files
-----------------------------------------
Sometimes your application requires Python packages that are not available through
`PyPI`_ or any other repository accesible with ``pip``. Such packages can be included
in the application as pip-installable `wheel` files. Remember to add these packages to the
`requirements.txt` file, e.g.:

::

    -f ./packages
    private_package

Test Your Application
---------------------
To validate that the application works as expected, it is important to write unit
tests and integration tests for your application. Test files should be stored in
the ``tests/`` folder. It is good practice to separate the test files from the application
source code. 


.. _PyPI: https://pypi.org/
.. _4Insight EngineRoom: https://4insight.io/#/engineroom
