#! /usr/bin/env python3
import os
import errno
def makedirs(path):
    """
    Make a directory and its parents if needed. If the directory already
    exists, do nothing.
    """
    try:
            os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise