from typing import IO
import logging
from PIL import ImageFile
from contextlib import contextmanager

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
NII_EXTS = ('.nii', '.nii.gz')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def is_io_object(obj):
    """
    Check if an object is a file-like object.
    """
    return callable(getattr(obj, "read", None))

@contextmanager
def peek(io_object: IO):
    """Context manager to preserve the position of a file-like object."""
    pos = io_object.tell()
    try:
        yield io_object
    finally:
        io_object.seek(pos)






