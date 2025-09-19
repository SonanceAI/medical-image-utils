from .nifti_utils import check_nifti_magic_numbers, NIFTI_MIMES, NIFTI_EXTENSIONS, DEFAULT_NIFTI_MIME
import mimetypes
from pathlib import Path
from .dicom_utils import is_dicom
import logging
from typing import IO
from .io_utils import is_io_object, peek
import gzip
import io

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
        return DEFAULT_NIFTI_MIME

    if is_dicom(buffer):
        return 'application/dicom'

    _LOGGER.info('Unable to determine MIME type from buffer, returning default mimetype')

    return DEFAULT_MIME_TYPE


def guess_type(name: str | Path | IO | bytes,
               use_magic=True):
    """
    Guess the MIME type and file extension of a file or file-like object.

    Args:
        name: The file path, file-like object, or byte data.
        use_magic: Whether to use magic library for MIME type detection.

    Returns:
        A tuple of (MIME type, file extension).
    """
    if isinstance(name, bytes):
        data_bytes = name
        name = ''
        io_obj = None
    elif is_io_object(name):
        io_obj = name
        data_bytes = None
        if isinstance(io_obj, gzip.GzipFile):
            if io_obj.name.endswith('.gz'):
                name = io_obj.name[:-3]
            else:
                name = io_obj.name
        else:
            name = getattr(name, 'name', '')
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
        return DEFAULT_NIFTI_MIME, suffix

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
            suffix = guess_extension(mime_type)
            return mime_type, suffix

    mime_type, encoding = mimetypes.guess_type(name, strict=False)
    suffix = guess_extension(mime_type) if mime_type else None

    return mime_type, suffix


def guess_typez(name: str | Path | IO | bytes,
                use_magic=True) -> tuple[list[str | None], str | None]:
    """
    Guess the MIME type and file extension of a file or file-like object,
    handling compressed files properly.

    Args:
        name: The file path, file-like object, or byte data.
        use_magic: Whether to use magic library for MIME type detection.

    Returns:
        A tuple of (MIME type, file extension).
    """
    mime_type, suffix = guess_type(name, use_magic=use_magic)
    if mime_type not in ('application/gzip', 'application/x-gzip'):
        return [mime_type], suffix
    
    if mime_type is None:
        _LOGGER.debug(f'Could not determine MIME type for file: {name}')
        return [None], None

    if suffix is None:
        _LOGGER.info(f"File has gzip MIME type ({mime_type}) but unknown extension! This should not happen!"
                     " Proceeding with '.gz' extension.")
        suffix = '.gz'

    # Handle gzip files
    if is_io_object(name):
        with peek(name) as io_obj:
            with gzip.open(io_obj, 'rb') as gz:
                mime_type2, suffix2 = guess_type(gz, use_magic=use_magic)
    elif isinstance(name, bytes):
        with gzip.open(io.BytesIO(name), 'rb') as gz:
            mime_type2, suffix2 = guess_type(gz, use_magic=use_magic)
    else:
        with gzip.open(name, 'rb') as gz:
            mime_type2, suffix2 = guess_type(gz, use_magic=use_magic)

    if suffix2 is None:
        suffix2 = ''
    return [mime_type2, mime_type], suffix2+suffix
