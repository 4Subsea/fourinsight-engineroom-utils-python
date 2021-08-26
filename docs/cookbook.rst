Cookbook
========


.. _example_custom_handler_ftp:

Custom file handler
-------------------

This example shows how you can set up your own custom file handler using FTP.

.. code-block:: python

    from io import StringIO
    from ftplib import FTP, error_perm
    from fourinsight.engineroom.utils.handlers import BaseHandler

    class FTPHandler(BaseHandler):
        """
        Handler for push/pull file content to/from an FTP server.

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

        def __init__(self, host, user, passwd, folder, filename):
            self._folder = folder
            self._filename = filename
            self._ftp = FTP(host=host, user=user, passwd=passwd)
            self._cwd(self._folder)
                
        def _cwd(self, folder):
            """
            Change current working directory, and make it if it does not exist.
            """
            try:
                self._ftp.cwd(folder)
            except error_perm:
                self._ftp.mkd(folder)
                self._ftp.cwd(folder)
        
        def pull(self):
            """
            Pull content from FTP server. Returns None if file is not found.
            """
            try:
                with StringIO() as content_stream:
                    ftplib.retrlines("RETR " + self.path, content_stream.write)
                    content_stream.seek(0)
                    remote_content = content_stream.read()
            except error_perm:
                remote_content = None
            return remote_content
        
        def push(self, local_content):
            """
            Push content to FTP server
            """
            self._ftp.storlines("STOR " + self._filename, StringIO(local_content))
