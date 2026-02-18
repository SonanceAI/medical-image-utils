"""
This module reads medical imaging data from various medical image file formats,
normalizing it into a consistent NumPy array format.

File formats supported:
- NIfTI files (.nii, .nii.gz) 
- DICOM files
- Standard image formats (PNG, JPEG, ...)
- Video files (MP4, AVI, etc.)
- NumPy arrays (.npy)

The main function `read_array_normalized` provides a unified interface for reading
different file formats and returns arrays in a consistent format with shape
(#frames, C, H, W) for multi-frame data or (C, H, W) for single frames.

Functions:
    read_array_normalized: Unified reader for all supported formats
    read_video: Read video files and extract frames
    read_image: Read standard image formats (PNG, JPEG, etc.)

The module handles format detection automatically and provides optional metadata
extraction for supported formats.
"""


import pydicom
import os
import cv2
import numpy as np
from PIL import Image
import logging
import tempfile
import shutil
from .nifti_utils import read_nifti, NIFTI_MIMES
from .dicom_utils import read_dicom_standardized as read_dicom
from .format_detection import guess_type, GZIP_MIME_TYPES
from typing import Any, BinaryIO, overload, Literal

_LOGGER = logging.getLogger(__name__)


def read_video(file_path: str | BinaryIO, index: int | None = None) -> np.ndarray:
    is_path = isinstance(file_path, (str, os.PathLike))
    temp_path = None
    video_source = file_path

    if not is_path:
        # Create a temporary file to read the video from stream
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        with os.fdopen(fd, 'wb') as f:
            file_path.seek(0)
            shutil.copyfileobj(file_path, f)
        video_source = temp_path

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise ValueError(f"Failed to open video file: {file_path}")
    try:
        if index is None:
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB and transpose to (C, H, W) format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.transpose(2, 0, 1)
                frames.append(frame)
            imgs = np.array(frames)  # shape: (#frames, C, H, W)
        else:
            while index > 0:
                cap.grab()
                index -= 1
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {index} from video file: {file_path}")
            # Convert BGR to RGB and transpose to (C, H, W) format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgs = frame.transpose(2, 0, 1)
    finally:
        cap.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    if imgs is None or len(imgs) == 0:
        raise ValueError(f"No frames found in video file: {file_path}")

    return imgs


def read_image(file_path: str | BinaryIO) -> np.ndarray:
    with Image.open(file_path) as pilimg:
        imgs = np.array(pilimg)
    if imgs.ndim == 2:  # (H, W)
        imgs = imgs[np.newaxis, np.newaxis]
    elif imgs.ndim == 3:  # (H, W, C)
        imgs = imgs.transpose(2, 0, 1)[np.newaxis]  # (H, W, C) -> (1, C, H, W)

    return imgs


@overload
def read_array_normalized(file_path: str | BinaryIO | bytes,
                          index: int | None = None,
                          return_metainfo: Literal[False] = False,
                          use_magic=True,
                          mime_type: str | None = None
                          ) -> np.ndarray: ...


@overload
def read_array_normalized(file_path: str | BinaryIO | bytes,
                          index: int | None = None,
                          *,
                          return_metainfo: Literal[True],
                          use_magic=True,
                          mime_type: str | None = None
                          ) -> tuple[np.ndarray, Any]: ...


def read_array_normalized(file_path: str | BinaryIO | bytes,
                          index: int | None = None,
                          return_metainfo: bool = False,
                          use_magic=True,
                          mime_type: str | None = None
                          ) -> np.ndarray | tuple[np.ndarray, Any]:
    """
    Read an array from a file.

    Args:
        file_path: The path to the file or a file-like object.
        index: If specified, read only the frame at this index (0-based).
            If None, read all frames.
        Supported file formats are NIfTI (.nii, .nii.gz), PNG (.png), JPEG (.jpg, .jpeg) and npy (.npy).

    Returns:
        The array read from the file in shape (#frames, C, H, W), if `index=None`,
            or (C, H, W) if `index` is specified.
    """
    is_path = isinstance(file_path, (str, os.PathLike))
    if is_path and not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metainfo = None
    if isinstance(file_path, bytes):
        from io import BytesIO
        file_path = BytesIO(file_path)

    try:
        if mime_type is None:
            mime_type, _ = guess_type(file_path, use_magic=use_magic)
            _LOGGER.debug(f"Detected MIME type: {mime_type}")
            if mime_type is None:
                raise ValueError(f"Could not determine MIME type for file: {file_path}")
        else:
            mime_type = mime_type.lower()

        if mime_type.split('/')[-1] == 'dicom':
            if not is_path:
                file_path.seek(0)
            ds = pydicom.dcmread(file_path)
            if index is not None:
                imgs, _ = read_dicom(ds, index=index)
                imgs = imgs[0]
            else:
                imgs, _ = read_dicom(ds)
            # Free up memory
            if hasattr(ds, '_pixel_array'):
                ds._pixel_array = None
            if hasattr(ds, 'PixelData'):
                ds.PixelData = None
            metainfo = ds
        elif mime_type.endswith('nifti') or mime_type in GZIP_MIME_TYPES:
            if not is_path:
                file_path.seek(0)
            imgs, nibmetainfo = read_nifti(file_path,
                                           mimetype=mime_type,
                                           slice_index=index,
                                           slice_axis=None)
            # For NIfTI files, try to load associated JSON metadata
            if return_metainfo:
                metainfo = nibmetainfo
                json_path = None
                if is_path:
                    if file_path.endswith('.nii.gz'):
                        json_path = file_path[:-7] + '.json'
                    elif file_path.endswith('.nii'):
                        json_path = file_path[:-4] + '.json'
                    elif file_path.endswith('.gz'):
                        json_path = file_path[:-3] + '.json'

                if json_path and os.path.exists(json_path):
                    try:
                        import json
                        with open(json_path, 'r') as f:
                            metainfo = json.load(f)
                        _LOGGER.debug(f"Loaded JSON metadata from {json_path}")
                    except Exception as e:
                        _LOGGER.warning(f"Failed to load JSON metadata from {json_path}: {e}")
        else:
            if mime_type.startswith('video/'):
                imgs = read_video(file_path, index)

            elif mime_type.startswith('image/'):
                imgs = read_image(file_path)
            elif mime_type == 'application/x-numpy-data':
                if not is_path:
                    file_path.seek(0)
                imgs = np.load(file_path)
                # if is an NpzFile, convert to array
                if isinstance(imgs, np.lib.npyio.NpzFile):
                    imgs = imgs[imgs.files[0]]
                if imgs.ndim != 4:
                    raise ValueError(
                        f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}. Expected 4 (N, C, H, W).")
            else:
                raise ValueError(f"Unsupported file format '{mime_type}' of '{file_path}'")

            if index is not None:
                if len(imgs) > 1:
                    _LOGGER.warning(f"It is inefficient to load all frames from '{file_path}' to access a single frame." +
                                    " Consider converting the file to a format that supports random access (DICOM), or" +
                                    " convert to png/jpeg files or" +
                                    " manually handle all frames at once instead of loading a specific frame.")
                imgs = imgs[index]

        if return_metainfo:
            return imgs, metainfo
        return imgs

    except Exception as e:
        _LOGGER.error(f"Failed to read array from '{file_path}': {e}")
        raise e
