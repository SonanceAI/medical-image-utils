from nibabel.spatialimages import SpatialImage
from nibabel.filebasedimages import ImageFileError
import numpy as np
import logging
from pathlib import Path
import nibabel as nib
import gzip
from medimgkit import GZIP_MIME_TYPES

_LOGGER = logging.getLogger(__name__)

DEFAULT_NIFTI_MIME = 'application/nifti'
NIFTI_MIMES = ['application/x-nifti', 'image/x.nifti', 'application/nifti']
NIFTI_EXTENSIONS = ('.nii', '.hdr')
_AXIS_MAPPING = {
    'sagittal': 0,
    'coronal': 1,
    'axial': 2
}


def _read_slice_or_full(nibdata: SpatialImage,
                        slice_index: int | None,
                        slice_axis: int | None) -> np.ndarray:
    """
    Read a slice or the full volume from a NIfTI image.
    """
    if slice_index is not None:
        if slice_axis is None:
            shape = nibdata.shape
            if len(shape) < 3:
                raise ValueError("NIfTI image must be at least 3D to extract a slice")
            if len(shape) == 3:
                slice_axis = 2
            else:
                slice_axis = 3

        return get_slice(nibdata, slice_index, slice_axis)

    if slice_axis is not None:
        raise ValueError("slice_index must be provided if slice_axis is provided")

    return nibdata.get_fdata()


def read_nifti(file_path: str,
               mimetype: str | None = None,
               slice_index: int | None = None,
               slice_axis: int | None = None) -> tuple[np.ndarray, SpatialImage]:
    """
    Read a NIfTI file and return the image data in standardized format.

    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz)
        mimetype: Optional MIME type of the file. If provided, it can help in determining how to read the file.

    Returns:
        np.ndarray: Image data with shape (#frames, C, H, W)
    """
    if slice_axis is not None and slice_index is None:
        raise ValueError("slice_index must be provided if slice_axis is provided")

    try:
        nibdata = nib.load(file_path)
        imgs = _read_slice_or_full(nibdata, slice_index, slice_axis)
    except ImageFileError:
        if mimetype is None:
            raise
        # has_ext = os.path.splitext(file_path)[1] != ''
        if mimetype in GZIP_MIME_TYPES:
            with gzip.open(file_path, 'rb') as f:
                nibdata = nib.Nifti1Image.from_stream(f)
                imgs = _read_slice_or_full(nibdata, slice_index, slice_axis)
        elif mimetype in NIFTI_MIMES:
            with open(file_path, 'rb') as f:
                nibdata = nib.Nifti1Image.from_stream(f)
                imgs = _read_slice_or_full(nibdata, slice_index, slice_axis)
        else:
            raise
    if imgs.ndim == 2:
        imgs = imgs.transpose(1, 0)
        if slice_index is None:
            imgs = imgs[np.newaxis, np.newaxis]
        else:
            imgs = imgs[np.newaxis]
    elif imgs.ndim == 3 and slice_index is None:
        imgs = imgs.transpose(2, 1, 0)
        imgs = imgs[:, np.newaxis]
    elif imgs.ndim == 4:
        # (H, W, #frames, C)
        imgs = imgs.transpose(2, 3, 1, 0)
    else:
        raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim} with {imgs.shape=}")

    # remove any cached data to free up memory
    if hasattr(nibdata, 'uncache'):
        nibdata.uncache()
    return imgs, nibdata


def slice_location_to_slice_index(data: SpatialImage,
                                  slice_location: float,
                                  slice_axis: int,
                                  ) -> int:
    """
    Convert a slice location in world coordinates to a slice index in the NIfTI image.
    """
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1 or 2")

    origin = data.affine[:3, 3]  # Location at voxel [0, 0, 0] in world coordinates. (translation vector)
    rotation_matrix = data.affine[:3, :3]

    # Get the directional vectors from the rotation matrix
    axis_vector = rotation_matrix[:, slice_axis]  # This is the direction of the slice axis in world coordinates

    # check that axis_vector is zero along other axes
    if not np.isclose(axis_vector[(slice_axis + 1) % 3], 0) or not np.isclose(axis_vector[(slice_axis + 2) % 3], 0):
        raise ValueError("Slice axis vector is not aligned with the specified slice axis.")

    slice_index = (slice_location-origin[slice_axis]) / axis_vector[slice_axis]
    slice_index = int(round(slice_index))
    return slice_index


def coplanar_vector_to_slice_axis(data: SpatialImage,
                                  coplanar_vector: np.ndarray,
                                  ) -> int:
    """
    IMPORTANT: ASSUMES coplanar_vector is not oblique to the image plane
        (i.e., the line is parallel to one of the image axes).
    """
    if not isinstance(coplanar_vector, np.ndarray) or coplanar_vector.ndim != 1 or coplanar_vector.size != 3:
        raise ValueError("coplanar_vector must be a 3-element numpy array")

    rotation_matrix = data.affine[:3, :3]
    coplanar_vector = coplanar_vector / np.linalg.norm(coplanar_vector)  # Normalize the vector

    # Find the slice axis that is most aligned with the coplanar vector
    dot_products = np.abs(rotation_matrix.T @ coplanar_vector)
    slice_axis = np.argmin(dot_products)

    return slice_axis


def get_slice_location_from_slice_axis(data: SpatialImage,
                                       world_point: np.ndarray,
                                       slice_axis: int) -> float:
    """    Get the slice location in world coordinates from a point and the slice axis.
    """
    if not isinstance(world_point, np.ndarray) or world_point.ndim != 1 or world_point.size != 3:
        raise ValueError("world_point must be a 3-element numpy array")

    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1 or 2")

    rotation_matrix = data.affine[:3, :3]
    axis_vector = rotation_matrix[:, slice_axis]  # This is the direction of the slice axis in world coordinates
    world_slice_axis = np.argmax(np.abs(axis_vector))
    return world_point[world_slice_axis]


def line_to_slice_index(data: SpatialImage,
                        world_point1: np.ndarray | None = None,
                        world_point2: np.ndarray | None = None,
                        coplanar_vector: np.ndarray | None = None) -> tuple[int, int]:
    """
    Convert a line defined by two points OR coplanar_vector in world coordinates to a slice index.
    IMPORTANT: Assumes the line is coplanar with the image plane (i.e., not oblique and aligned with the image axes).
    """
    # either world_point1 and world_point2 must be provided, or coplanar_vector must be provided
    if (world_point1 is None or world_point2 is None) and coplanar_vector is None:
        raise ValueError("Either world_point1 and world_point2 or coplanar_vector must be provided")

    if world_point1 is not None:
        coplanar_vector = world_point2 - world_point1

    slice_axis = coplanar_vector_to_slice_axis(data, coplanar_vector)
    slice_location = get_slice_location_from_slice_axis(data, world_point1, slice_axis)
    slice_index = slice_location_to_slice_index(data,
                                                slice_location=slice_location,
                                                slice_axis=slice_axis
                                                )

    return slice_index, slice_axis


def get_slice_from_line(data: SpatialImage,
                        world_point1: np.ndarray,
                        world_point2: np.ndarray) -> np.ndarray:
    """
    Get the slice 2D image from a line defined by two points in world coordinates.
    """
    slice_index, slice_axis = line_to_slice_index(data, world_point1, world_point2)
    return get_slice(data, slice_index, slice_axis)


def get_slice(data: SpatialImage,
              slice_index: int,
              slice_axis: int) -> np.ndarray:
    """
    Get a 2D slice from a 3D NIfTI volume based on the slice index and axis.

    Args:
        data (SpatialImage): The NIfTI image data whose slice is to be extracted.
        slice_index (int): The index of the slice to extract.
        slice_axis (int): The axis along which to extract the slice (0 for x, 1 for y, 2 for z).

    Returns:
        np.ndarray: The extracted 2D slice image with shape (W, H).
    """
    # Check the on-disk data order ('C' or 'F')
    # 'C' means C-contiguous (row-major), fastest changing is the first index.
    # Slicing the first axis (e.g., r.dataobj[0, :, :]) is fastest for 'C' order.
    # 'F' means Fortran-contiguous (column-major), fastest changing is the last index.
    # Slicing the last axis (e.g., r.dataobj[:, :, 0]) is fastest for 'F' order.
    dataorder = data.dataobj.order

    if slice_axis == 0:
        if dataorder == 'C':
            slice_image = data.dataobj[slice_index, :, :]
        else:
            slice_image = data.get_fdata()[slice_index, :, :]
    elif slice_axis == 1:
        slice_image = data.get_fdata()[:, slice_index, :]
    elif slice_axis == 2:
        if dataorder == 'F':
            slice_image = data.dataobj[:, :, slice_index]
        else:
            slice_image = data.get_fdata()[:, :, slice_index]
    else:
        raise ValueError(f"Invalid slice axis: {slice_axis}. Must be 0, 1, or 2.")

    return slice_image


def is_nifti_file(file_path: Path | str) -> bool:
    """
    Check if the file is a NIfTI file based on its extension, mimetype, or magic number.
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    # Check file extension
    if file_path.name.lower().endswith(NIFTI_EXTENSIONS):
        return True

    # Check if file exists before trying to read magic number
    if not file_path.exists():
        return False

    # Check magic number
    try:
        import magic
        import gzip
        file_type = magic.from_file(str(file_path), mime=True)
        if file_type in NIFTI_MIMES:
            return True
        if file_type in GZIP_MIME_TYPES:
            with gzip.open(file_path, 'rb') as f:
                subfiletype = magic.from_buffer(f.read(1024), mime=True)
            if subfiletype in NIFTI_MIMES:
                return True
    except ImportError:
        # If the magic module is not available, we cannot check magic numbers
        _LOGGER.warning("The 'magic' module is not available. Cannot check magic numbers for NIfTI files.")
    except (IOError, OSError):
        return False

    return False


def check_nifti_magic_numbers(data: bytes) -> bool:
    """
    Check if the provided byte data contains NIfTI magic numbers.
    """
    # NIfTI-1 magic numbers
    NIFTI1_MAGIC = b'\x6e\x2b\x31\x00'  # "n+1\0"
    NIFTI1_MAGIC_ALT = b'\x6e\x69\x31\x00'  # "ni1\0"

    # NIfTI-2 magic numbers
    NIFTI2_MAGIC = b'\x6e\x2b\x32\x00'  # "n+2\0"
    NIFTI2_MAGIC_ALT = b'\x6e\x69\x32\x00'  # "ni2\0"

    if len(data) < 4:
        return False

    # Check for NIfTI-1 magic numbers at offset 344
    if len(data) >= 348:
        magic_at_344 = data[344:348]
        if magic_at_344 in (NIFTI1_MAGIC, NIFTI1_MAGIC_ALT):
            return True

    # Check for NIfTI-2 magic numbers at offset 4
    magic_at_4 = data[4:8]
    if magic_at_4 in (NIFTI2_MAGIC, NIFTI2_MAGIC_ALT):
        return True

    # Check for magic numbers at the beginning (some implementations)
    magic_at_0 = data[0:4]
    if magic_at_0 in (NIFTI1_MAGIC, NIFTI1_MAGIC_ALT, NIFTI2_MAGIC, NIFTI2_MAGIC_ALT):
        return True

    return False


def axis_name_to_axis_index(data: SpatialImage,
                            axis_name: str) -> int:
    """
    Convert an axis name to its corresponding index in the NIfTI image.
    ASSUMES data indices are aligned with the axis.

    Args:
        data (SpatialImage): The NIfTI image data.
        axis_name (str): The name of the axis ('sagittal', 'coronal', 'axial').

    Returns:
        int: The index of the axis (0, 1, or 2).
    """
    rotation_matrix = data.affine[:3, :3]
    axis_name = axis_name.lower()
    idx = _AXIS_MAPPING.get(axis_name)
    if idx is None:
        raise ValueError(f"Unknown axis name: {axis_name}. Expected one of {set(_AXIS_MAPPING.keys())}.")
    axis_index = np.argmax(np.abs(rotation_matrix[:, idx]))
    return axis_index


def get_nifti_shape(file_path: str) -> tuple:
    """
    Get the shape of a NIfTI file.

    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz)

    Returns:
        tuple: Shape of the NIfTI image (X, Y, Z)
    """
    try:
        return nib.load(file_path).shape
    except ImageFileError as e:
        from .format_detection import guess_type
        mimetype, _ = guess_type(file_path)
        if mimetype is None:
            raise
        if mimetype in GZIP_MIME_TYPES:
            with gzip.open(file_path, 'rb') as f:
                nibdata = nib.Nifti1Image.from_stream(f)
                return nibdata.shape
        elif mimetype in NIFTI_MIMES:
            with open(file_path, 'rb') as f:
                nibdata = nib.Nifti1Image.from_stream(f)
                return nibdata.shape
        else:
            raise


def world_to_voxel(data: SpatialImage,
                   world_coords: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates to voxel indices in the NIfTI image.

    Args:
        data (SpatialImage): The NIfTI image data.
        world_coords (np.ndarray): World coordinates of shape (N, 3), for multiple points, or (3,), for a single point.

    Returns:
        np.ndarray: Voxel indices of shape (N, 3) or (3,).

    """

    if world_coords.ndim == 1:
        single_point = True
        if world_coords.size != 3:
            raise ValueError("world_coords must be of shape (3,) for a single point.")
        world_coords = world_coords[np.newaxis, :]  # Convert to shape (1, 3)
    elif world_coords.ndim == 2:
        single_point = False
        if world_coords.shape[1] != 3:
            raise ValueError("world_coords must be of shape (N, 3) for multiple points.")
    else:
        raise ValueError("world_coords must be either 1D or 2D numpy array.")

    # 1. Convert world coordinates to voxel coordinates
    inv_affine = np.linalg.inv(data.affine)
    # Add homogeneous coordinate (w=1)
    points_hom = np.hstack((world_coords, np.ones((world_coords.shape[0], 1))))
    # Apply inverse affine transformation
    voxel_points = points_hom @ inv_affine.T
    # Remove homogeneous coordinate
    voxel_points = voxel_points[:, :3]

    # 2. Round to nearest integer voxel indices
    voxel_indices = np.rint(voxel_points).astype(int)
    if single_point:
        return voxel_indices[0]
    return voxel_indices
