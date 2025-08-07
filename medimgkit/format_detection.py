from .nifti_utils import check_nifti_magic_numbers, NIFTI_MIMES, NIFTI_EXTENSIONS
import mimetypes
from pathlib import Path
from .dicom_utils import is_dicom
import logging
from typing import IO
from .io_utils import is_io_object, peek

_LOGGER = logging.getLogger(__name__)
DEFAULT_MIME_TYPE = 'application/octet-stream'

_MIME_MAP = {
    'application/x-numpy-data': '.npy',
    'application/x-nifti-gz': '.nii.gz',
}
_MIME_MAP.update({k: '.nii' for k in NIFTI_MIMES})


def guess_extension(type: str) -> str | None:
    ext = mimetypes.guess_extension(type, strict=False)
    if ext:
        return ext
    return _MIME_MAP.get(type, ext)


def magic_from_buffer(buffer: bytes, mime=True) -> str:
    try:
        import magic
        mime_type = magic.from_buffer(buffer, mime=mime)
        if mime_type != DEFAULT_MIME_TYPE:
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

    return DEFAULT_MIME_TYPE


def guess_type(name: str | Path | IO | bytes, use_magic=True):
    if isinstance(name, bytes):
        data_bytes = name
        name = ''
        io_obj = None
    elif is_io_object(name):
        io_obj = name
        name = getattr(name, 'name', '')
        data_bytes = None
    else:
        io_obj = None
        data_bytes = None

    name = Path(name).expanduser()
    suffix = name.suffix

    if suffix in ('.npy', '.npz'):
        return 'application/x-numpy-data', suffix
    if suffix == '.gz':
        return 'application/gzip', suffix
    if suffix in NIFTI_EXTENSIONS:
        return 'image/x.nifti', suffix

    # Try magic if requested
    if use_magic:
        if data_bytes is None:
            if io_obj is not None:
                with peek(io_obj):  # Ensure we don't change the stream position
                    data_bytes = io_obj.read(2048)
            else:
                with open(name, 'rb') as f:
                    data_bytes = f.read(2048)
        mime_type = magic_from_buffer(data_bytes, mime=True).strip()
        if mime_type != DEFAULT_MIME_TYPE:
            if not suffix:
                suffix = guess_extension(mime_type)
            return mime_type, suffix

    mime_type, encoding = mimetypes.guess_type(name, strict=False)

    return mime_type, suffix
