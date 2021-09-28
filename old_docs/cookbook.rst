Cookbook
========


.. _example_custom_handler_ftp:

Set up a custom handler based on FTP
------------------------------------

This example shows how you can set up a custom handler based on FTP. The handler
must inherit from :class:`~fourinsight.engineroom.utils.core.BaseHandler`, and override
the two abstract methods, ``_push()`` and ``_pull()``. It is recommended to also
set the class variable, ``_SOURCE_NOT_FOUND_ERROR``, to the type of exception that
is expected to be raised if the source file can not be read.

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
