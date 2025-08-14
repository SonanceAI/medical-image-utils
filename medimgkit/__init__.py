"""Medical Image Utils - Utilities for working with medical images."""
import importlib.metadata

__version__ = importlib.metadata.version(__name__)

_MAP_STANDARD_MIME_TYPES_ = {
    "image/dicom": "application/dicom",
    'image/x-dicom': 'application/dicom',
    "image/nifti": "application/nifti",
    'application/x-nifti': 'application/nifti'
}


def standardize_mimetype(mimetype: str) -> str:
    """
    Standardize the MIME type string, due to several mimetypes meaning the same thing.
    """
    if mimetype not in _MAP_STANDARD_MIME_TYPES_:
        return mimetype.strip().lower()
    return _MAP_STANDARD_MIME_TYPES_[mimetype]
