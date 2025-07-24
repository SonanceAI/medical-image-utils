import numpy as np
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage as NibFileBasedImage
from PIL import Image
from .dicom_utils import load_image_normalized, is_dicom
import pydicom
import os
from typing import Any
import logging
from PIL import ImageFile
import cv2
import gzip

ImageFile.LOAD_TRUNCATED_IMAGES = True

_LOGGER = logging.getLogger(__name__)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')
NII_EXTS = ('.nii', '.nii.gz')
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')


def read_video(file_path: str, index: int | None = None) -> np.ndarray:
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
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

    if imgs is None or len(imgs) == 0:
        raise ValueError(f"No frames found in video file: {file_path}")

    return imgs


def read_nifti(file_path: str, mimetype: str | None = None) -> np.ndarray:
    """
    Read a NIfTI file and return the image data in standardized format.
    
    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz)
        mimetype: Optional MIME type of the file. If provided, it can help in determining how to read the file.
        
    Returns:
        np.ndarray: Image data with shape (#frames, C, H, W)
    """
    from nibabel.filebasedimages import ImageFileError
    try:
        imgs = nib.load(file_path).get_fdata()  # shape: (W, H, #frame) or (W, H)
    except ImageFileError as e:
        if mimetype is None:
            raise e
        # has_ext = os.path.splitext(file_path)[1] != ''
        if mimetype == 'application/gzip':
            with gzip.open(file_path, 'rb') as f:
                imgs = nib.Nifti1Image.from_stream(f).get_fdata()
        elif mimetype in ('image/x.nifti', 'application/x-nifti'):
            with open(file_path, 'rb') as f:
                imgs = nib.Nifti1Image.from_stream(f).get_fdata()
        else:
            raise e
    if imgs.ndim == 2:
        imgs = imgs.transpose(1, 0)
        imgs = imgs[np.newaxis, np.newaxis]
    elif imgs.ndim == 3:
        imgs = imgs.transpose(2, 1, 0)
        imgs = imgs[:, np.newaxis]
    else:
        raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")

    return imgs


def read_image(file_path: str) -> np.ndarray:
    with Image.open(file_path) as pilimg:
        imgs = np.array(pilimg)
    if imgs.ndim == 2:  # (H, W)
        imgs = imgs[np.newaxis, np.newaxis]
    elif imgs.ndim == 3:  # (H, W, C)
        imgs = imgs.transpose(2, 0, 1)[np.newaxis]  # (H, W, C) -> (1, C, H, W)

    return imgs


def read_array_normalized(file_path: str,
                          index: int | None = None,
                          return_metainfo: bool = False,
                          use_magic=False) -> np.ndarray | tuple[np.ndarray, pydicom.Dataset | NibFileBasedImage | None]:
    """
    Read an array from a file.

    Args:
        file_path: The path to the file.
        index: If specified, read only the frame at this index (0-based).
            If None, read all frames.
        Supported file formats are NIfTI (.nii, .nii.gz), PNG (.png), JPEG (.jpg, .jpeg) and npy (.npy).

    Returns:
        The array read from the file in shape (#frames, C, H, W), if `index=None`,
            or (C, H, W) if `index` is specified.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metainfo = None

    try:
        if is_dicom(file_path):
            ds = pydicom.dcmread(file_path)
            if index is not None:
                imgs = load_image_normalized(ds, index=index)[0]
            else:
                imgs = load_image_normalized(ds)
            # Free up memory
            if hasattr(ds, '_pixel_array'):
                ds._pixel_array = None
            if hasattr(ds, 'PixelData'):
                ds.PixelData = None
            metainfo = ds
        else:
            if use_magic:
                import magic  # it is important to import here because magic has an OS lib dependency.
                mime_type = magic.from_file(file_path, mime=True)
            else:
                mime_type = ""

            if mime_type.startswith('video/') or file_path.endswith(VIDEO_EXTS):
                imgs = read_video(file_path, index)
            else:
                if mime_type in ('image/x.nifti', 'application/x-nifti') or mime_type == 'application/gzip' or file_path.endswith(NII_EXTS):
                    imgs = read_nifti(file_path, mimetype=mime_type)
                    # For NIfTI files, try to load associated JSON metadata
                    if return_metainfo:
                        json_path = file_path.replace('.nii.gz', '.json').replace('.nii', '.json')
                        if os.path.exists(json_path):
                            try:
                                import json
                                with open(json_path, 'r') as f:
                                    metainfo = json.load(f)
                                _LOGGER.debug(f"Loaded JSON metadata from {json_path}")
                            except Exception as e:
                                _LOGGER.warning(f"Failed to load JSON metadata from {json_path}: {e}")
                                metainfo = None
                elif mime_type.startswith('image/') or file_path.endswith(IMAGE_EXTS):
                    imgs = read_image(file_path)
                elif file_path.endswith('.npy') or mime_type == 'application/x-numpy-data':
                    imgs = np.load(file_path)
                    if imgs.ndim != 4:
                        raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim}")
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
