.. _text_content_handlers:

Handlers
========

Some of the core functionality provided by :mod:`fourinsight.engineroom.utils` relies
on handlers that facilitate downloading and uploading of text content from a source.
The source can be a local file, an Azure Storage Blob, or any other suitable storage
place. Two handlers, the :class:`~fourinsight.engineroom.utils.LocalFileHandler` and the :class:`~fourinsight.engineroom.utils.AzureBlobHandler`
are available out-of-the-box. :ref:`Custom handlers<custom_handlers>` are easily set up by inheriting from
:class:`~fourinsight.engineroom.utils.core.BaseHandler`.

The :class:`~fourinsight.engineroom.utils.LocalFileHandler` is used to store text content in a local file.

.. code-block:: python

    from fourinsight.engineroom.utils import LocalFileHandler


    handler = LocalFileHandler(<file-path>)

The :class:`~fourinsight.engineroom.utils.AzureBlobHandler` is used to store text content in `Azure Blob Storage`_.

.. code-block:: python

    from fourinsight.engineroom.utils import AzureBlobHandler


    handler = AzureBlobHandler(<connection-string>, <container-name>, <blob-name>)

The handlers behave like 'streams', and provide all the normal stream capabilities. Downloading and uploading is done  by a push/pull
strategy; content is retrieved from the source by a :meth:`~fourinsight.engineroom.utils.core.BaseHandler.pull()` request, and uploaded
to the source by a :meth:`~fourinsight.engineroom.utils.core.BaseHandler.push()`. Correspondingly reading and writing to the handler is
done using :meth:`~io.TextIOWrapper.read()` and :meth:`~io.TextIOWrapper.write()`.

For reading from handlers:

.. code-block:: python

    # Pull stream
    handler.pull()

    # Read stream content
    handler.seek(0)
    handler.read()

and writing to handlers:

.. code-block:: python

    # Write text content to stream
    handler.write("Hello, World!")

    # Push stream content
    handler.push()

More interestingly, handler can also be used with :func:`pandas.read_csv()`:

.. code-block:: python

    # Pull stream w/ CSV content
    handler.pull()

    # Load stream content as 'pandas.DataFrame'
    handler.seek(0)
    df = pd.read_csv(handler, index_col=0)

and :meth:`pandas.DataFrame.to_csv()`:

.. code-block:: python

    df = pd.DataFrame({"Hello": [1, 2], "World!": [3, 4]})

    # Write 'pandas.DataFrame' to stream
    df.to_csv(handler)

    # Push stream content
    handler.push()

.. important::
    Remember to perform ``seek(0)`` to go to the beginning of the stream before reading.


.. _custom_handlers:

Custom handlers
---------------

The custom handler must inherit from :class:`~fourinsight.engineroom.utils.core.BaseHandler`, and override
the two abstract methods, :meth:`~fourinsight.engineroom.utils.core.BaseHandler._push()` and :meth:`~fourinsight.engineroom.utils.core.BaseHandler._pull()`. It is recommended to also
set the class variable, :attr:`~fourinsight.engineroom.utils.core.BaseHandler._SOURCE_NOT_FOUND_ERROR`, to the type of exception that
is expected to be raised if the source file can not be read. The example below shows how you can set up a custom handler based on FTP.

.. code-block:: python

    from io import BytesIO
    from ftplib import FTP, error_perm
    from fourinsight.engineroom.utils.core import BaseHandler


    class FTPHandler(BaseHandler):
        """
        Handler for push/pull text content to/from an FTP server file.

        Parameters
        ----------
        host : str
            FTP host.
        user: str
            FTP user.
        passwd : str
            FTP password.
        folder : str
            Folder where the file should be stored.
        filename : str
            Filename.
        """
        _SOURCE_NOT_FOUND_ERROR = error_perm

        def __init__(self, host, user, passwd, folder, filename):
            self._folder = folder
            self._filename = filename
            self._ftp = FTP(host=host, user=user, passwd=passwd)
            self._cwd(self._folder)
            super().__init__()

        def _cwd(self, folder):
            """
            Change current working directory, and make it if it does not exist.
            """
            try:
                self._ftp.cwd(folder)
            except error_perm:
                self._ftp.mkd(folder)
                self._ftp.cwd(folder)
                
        def _pull(self):
            """
            Pull text content from FTP server, and write the string to stream.

            Returns
            -------
            int
                Number of characters written to stream (which is always equal to the
                length of the string).
            """
            with BytesIO() as binary_content:
                self._ftp.retrbinary("RETR " + self._filename, binary_content.write)
                characters_written = self.write(binary_content.getvalue().decode(self.encoding))
                    
            return characters_written
            
        def _push(self):
            """
            Push the stream content to source.
            """
            self.seek(0)
            self._ftp.storbinary("STOR " + self._filename, self.buffer)

.. _Azure Blob Storage: https://azure.microsoft.com/nb-no/services/storage/blobs/
