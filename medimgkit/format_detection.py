from .nifti_utils import check_nifti_magic_numbers, NIFTI_MIMES
import mimetypes
from pathlib import Path
from .dicom_utils import is_dicom
import logging
from typing import IO
from .io_utils import is_io_object, peek

_LOGGER = logging.getLogger(__name__)


def guess_extension(type: str) -> str | None:
    ext = mimetypes.guess_extension(type, strict=False)
    if ext is None:
        if type in NIFTI_MIMES:
            return ".nii"
    return ext


def magic_from_buffer(buffer: bytes, mime=True) -> str:
    try:
        import magic
        mime_type = magic.from_buffer(buffer, mime=mime)
        if mime_type != 'application/octet-stream':
            return mime_type
    except ImportError:
        pass

    import puremagic
    try:
        mime_type = puremagic.from_string(buffer, mime=mime)
        return mime_type
    except puremagic.PureError:
        pass

    if check_nifti_magic_numbers(buffer):
        return 'image/x.nifti'

    if is_dicom(buffer):
        return 'application/dicom'

    _LOGGER.info('Unable to determine MIME type from buffer, returning default mimetype')

    return 'application/octet-stream'


def guess_type(name: str | Path | IO, use_magic=True):
    if is_io_object(name):
        io_obj: IO = name
        name = getattr(name, 'name', None)

    else:
        io_obj = None

    # Try mimetypes first if we have a name
    if name:
        gtype = mimetypes.guess_type(name, strict=False)
        if gtype[0]:
            return gtype

    # Try magic if requested
    if use_magic:
        if io_obj is not None:
            with peek(io_obj):  # Ensure we don't change the stream position
                data_bytes = io_obj.read(2048)
        else:
            with open(name, 'rb') as f:
                data_bytes = f.read(2048)
        mime_type = magic_from_buffer(data_bytes, mime=True)
        return mime_type, guess_extension(mime_type)

    return None, None
