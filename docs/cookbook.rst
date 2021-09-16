Cookbook
========


.. _example_custom_handler_ftp:

Set up a custom handler based on FTP
------------------------------------

This example shows how you can set up a custom handler based on FTP. The handler
must inherit from :class:`~fourinsight.engineroom.utils.core.BaseHandler`, and override
the two abstract methods, ``push()`` and ``pull()``. If the file content you
want to download does not exist, the ``pull()`` method should return ``None``.

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

        def pull(self, raise_on_missing=True):
            """
            Pull text content from FTP server. Returns None if file is not found.

            Parameters
            ----------
            raise_on_missing : bool
                Raise exception if content can not be pulled from file.
            """
            try:
                with BytesIO() as binary_content:
                    self._ftp.retrbinary("RETR " + self._filename, binary_content.write)
                    binary_content.seek(0)
                    remote_content = binary_content.read().decode()
            except Exception as e:
                remote_content = None
                if raise_on_missing:
                    raise e
            return remote_content

        def push(self, local_content):
            """
            Push text content to FTP server file.
            """
            self._ftp.storbinary("STOR " + self._filename, BytesIO(local_content.encode()))
