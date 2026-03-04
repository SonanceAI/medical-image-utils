"""Medical Image Utils - Utilities for working with medical images."""
import importlib.metadata
from typing import Literal, TypeAlias

__version__ = importlib.metadata.version(__name__)

_MAP_STANDARD_MIME_TYPES_ = {
    # dicom
    "image/dicom": "application/dicom",
    'image/x-dicom': 'application/dicom',
    # nifti
    "image/nifti": "application/nifti",
    'application/x-nifti': 'application/nifti',
    'image/x.nifti': 'application/nifti',
    # gzip
    'application/x-gzip': 'application/gzip',
}

GZIP_MIME_TYPES = ('application/gzip', 'application/x-gzip')

ViewPlane: TypeAlias = Literal['axial', 'sagittal', 'coronal']

def standardize_mimetype(mimetype: str) -> str:
    """
    Standardize the MIME type string, due to several mimetypes meaning the same thing.
    """
    if mimetype not in _MAP_STANDARD_MIME_TYPES_:
        return mimetype.strip().lower()
    return _MAP_STANDARD_MIME_TYPES_[mimetype]
