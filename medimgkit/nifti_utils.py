from contextlib import contextmanager
import gzip
from io import IOBase
import logging
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.loadsave import load as nib_load
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
import nibabel as nib

from medimgkit import GZIP_MIME_TYPES, ViewPlane

_LOGGER = logging.getLogger(__name__)

DEFAULT_NIFTI_MIME = 'application/nifti'
NIFTI_MIMES = ['application/x-nifti', 'image/x.nifti', 'application/nifti']
NIFTI_EXTENSIONS = ('.nii', '.hdr')
DEFAULT_SLICE_PLANE: ViewPlane = 'axial'
VALID_PLANES = frozenset({'axial', 'sagittal', 'coronal'})
_PLANE_FORWARD_CODE = {
    'sagittal': 'R',
    'coronal': 'A',
    'axial': 'S',
}
_PLANES_TO_CODES = {
    "sagittal": {'L', 'R'},
    "coronal": {'A', 'P'},
    "axial": {'I', 'S'}
}


def _get_requested_plane(slice_axis: int | None,
                         plane: ViewPlane | None,
                         *,
                         default_plane: ViewPlane | None = None) -> ViewPlane | None:
    if plane is not None:
        return plane
    if slice_axis is None:
        return default_plane
    return None


def _get_plane_axis(data: SpatialImage, plane: ViewPlane) -> int:
    if plane not in VALID_PLANES:
        raise ValueError(f"Invalid plane '{plane}'. Must be one of {VALID_PLANES}.")

    axcodes = nib.orientations.aff2axcodes(data.affine)
    target_codes = _PLANES_TO_CODES[plane]

    for axis_index, code in enumerate(axcodes):
        if code in target_codes:
            return axis_index

    raise RuntimeError(f"Could not map the {plane} plane from affine codes: {axcodes}")


def _get_affine(data: SpatialImage) -> np.ndarray:
    affine = data.affine
    if affine is None:
        raise ValueError("NIfTI image does not define an affine matrix")
    return affine


def _load_spatial_image(file_path: str | Path) -> SpatialImage:
    return cast(SpatialImage, nib_load(file_path))


def _resolve_slice_axis(data: SpatialImage,
                        slice_axis: int | None = None,
                        plane: ViewPlane | None = None) -> int | None:
    if slice_axis is not None and plane is not None:
        raise ValueError("slice_axis and plane are mutually exclusive")

    if plane is not None:
        return _get_plane_axis(data, plane)

    if slice_axis is None:
        return None

    if len(data.shape) < 3:
        raise ValueError("NIfTI image must be at least 3D to extract a slice")

    if slice_axis not in (0, 1, 2):
        raise ValueError(f"Invalid slice axis: {slice_axis}. Must be 0, 1, or 2.")

    return slice_axis


def _resolve_slice_context(data: SpatialImage,
                           slice_axis: int | None = None,
                           plane: ViewPlane | None = None,
                           *,
                           default_plane: ViewPlane | None = None) -> tuple[ViewPlane | None, int | None, bool]:
    requested_plane = _get_requested_plane(slice_axis, plane, default_plane=default_plane)
    resolved_slice_axis = _resolve_slice_axis(data, slice_axis=slice_axis, plane=requested_plane)
    reverse_slice_order = (
        resolved_slice_axis is not None
        and _should_reverse_slice_order(data, resolved_slice_axis, requested_plane)
    )
    return requested_plane, resolved_slice_axis, reverse_slice_order


def _should_reverse_slice_order(data: SpatialImage,
                                slice_axis: int,
                                plane: ViewPlane | None) -> bool:
    if plane is None:
        return False

    axcodes = nib.orientations.aff2axcodes(data.affine)
    return axcodes[slice_axis] != _PLANE_FORWARD_CODE[plane]


def _normalize_slice_index(slice_index: int | np.ndarray,
                           axis_size: int,
                           reverse_order: bool) -> np.ndarray:
    slice_index_arr = np.asarray(slice_index, dtype=np.float64)
    rounded_slice_index = np.rint(slice_index_arr)

    if not np.allclose(slice_index_arr, rounded_slice_index):
        raise ValueError("slice_index must contain integer indices.")

    raw_slice_index = rounded_slice_index.astype(int)
    if np.any((raw_slice_index < 0) | (raw_slice_index >= axis_size)):
        raise ValueError(
            f"slice_index contains values outside the valid range [0, {axis_size - 1}].")

    if reverse_order:
        return axis_size - 1 - raw_slice_index
    return raw_slice_index


def _standardize_slice_array(imgs: np.ndarray) -> np.ndarray:
    if imgs.ndim == 2:
        return imgs.transpose(1, 0)[np.newaxis]

    if imgs.ndim == 3:
        return imgs.transpose(2, 1, 0)

    raise ValueError(f"Unsupported slice shape: {imgs.shape}")


def _standardize_volume_array(imgs: np.ndarray,
                              file_path: str | Path | BinaryIO,
                              slice_axis: int | None = None) -> np.ndarray:
    if imgs.ndim == 2:
        return imgs.transpose(1, 0)[np.newaxis, np.newaxis]

    if imgs.ndim == 3:
        if slice_axis is None:
            raise ValueError("slice_axis must be provided for 3D NIfTI volumes")
        axis_order = [slice_axis] + [axis for axis in range(2, -1, -1) if axis != slice_axis]
        return imgs.transpose(*axis_order)[:, np.newaxis]

    if imgs.ndim == 4:
        if slice_axis is None:
            raise ValueError("slice_axis must be provided for 4D NIfTI volumes")
        spatial_axes = [axis for axis in range(2, -1, -1) if axis != slice_axis]
        axis_order = [slice_axis, 3, *spatial_axes]
        return imgs.transpose(*axis_order)

    raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim} with {imgs.shape=}")


def _read_slice_or_full(nibdata: SpatialImage,
                        slice_index: int | None,
                        slice_axis: int | None,
                        plane: ViewPlane | None = None) -> np.ndarray:
    """
    Read a slice or the full volume from a NIfTI image.
    """
    if slice_index is not None:
        return get_slice(nibdata, slice_index, slice_axis=slice_axis, plane=plane)

    return nibdata.get_fdata()


def rawplaneaxis2stdplaneaxis_idx(axis_idx: int) -> int:
    """
    Convert a raw plane axis index (0, 1, 2) to the corresponding standardized plane axis index.
    Commonly used with py:func:`medimgkit.readers.read_array_normalized`.
    """
    if axis_idx == 0:
        return 3
    elif axis_idx == 1:
        return 2
    elif axis_idx == 2:
        return 0
    else:
        raise ValueError(f"Invalid raw plane axis index: {axis_idx}. Must be 0, 1, or 2.")


@contextmanager
def _open_stream(f_io: BinaryIO,
                 mimetype: str | None = None):
    try:
        if mimetype is None or mimetype in GZIP_MIME_TYPES:
            with gzip.open(f_io, 'rb') as f:
                yield Nifti1Image.from_stream(f)
                return
    except gzip.BadGzipFile:
        pass  # Not a gzip file, try other methods

    yield Nifti1Image.from_stream(cast(IOBase, f_io))


def read_nifti(file_path: str | Path | BinaryIO,
               mimetype: str | None = None,
               slice_index: int | None = None,
               slice_axis: int | None = None,
               plane: ViewPlane | None = None,
               orientation_normalized: bool = False) -> tuple[np.ndarray, SpatialImage]:
    """
    Read a NIfTI file and return the image data in standardized format.

    Args:
        file_path: Path to the NIfTI file (.nii or .nii.gz)
        mimetype: Optional MIME type of the file. If provided, it can help in determining how to read the file.
        slice_index: Optional slice index. When provided, the returned array is a
            standardized slice with shape (C, H, W).
        slice_axis: Optional spatial axis index (0, 1, or 2) used when
            standardizing the output volume or when ``slice_index`` is provided.
        plane: Optional anatomical plane used instead of ``slice_axis``. When a
            plane is used, slices are ordered using the affine, but the image is
            not otherwise rotated or canonicalized.
        orientation_normalized: If True, the returned full volume is reoriented to a standard orientation (RAS) and slice order is determined by the affine.
            This has no effect when ``slice_index`` is provided.

    Returns:
        np.ndarray: Image data with shape (#frames, C, H, W)
    """
    if isinstance(file_path, (str, Path)):
        try:
            nibdata = _load_spatial_image(file_path)
            # if 3d, shape is (sagittal, coronal, axial).
            # If 4d, shape is (sagittal, coronal, axial, time).
            # TODO: consider using `nib.funcs.as_closest_canonical(nibdata)`
            # nibdata = nib.funcs.as_closest_canonical(nibdata)
            imgs = _read_slice_or_full(nibdata, slice_index, slice_axis, plane=plane)
        except ImageFileError:
            # it is possible that the file is a NIfTI file but with an unrecognized extension.
            with open(file_path, 'rb') as f:
                with _open_stream(f, mimetype) as nibdata:
                    imgs = _read_slice_or_full(nibdata, slice_index, slice_axis, plane=plane)
    else:
        with _open_stream(file_path, mimetype) as nibdata:
            imgs = _read_slice_or_full(nibdata, slice_index, slice_axis, plane=plane)

    if slice_index is None:
        if orientation_normalized:
            _, resolved_slice_axis, reverse_slice_order = _resolve_slice_context(
                nibdata,
                slice_axis,
                plane,
                default_plane=DEFAULT_SLICE_PLANE if len(nibdata.shape) >= 3 else None,
            )
            if reverse_slice_order:
                imgs = np.flip(imgs, axis=resolved_slice_axis)
            imgs = _standardize_volume_array(imgs, file_path, slice_axis=resolved_slice_axis)
        else:
            if imgs.ndim == 3:
                # (row, col, slice) -> (slice, 1, row, col)
                imgs = imgs.transpose(2, 0, 1)[:, np.newaxis]
            elif imgs.ndim == 4:
                # (row, col, slice, time) -> (slice, time, row, col)
                imgs = imgs.transpose(2, 3, 0, 1)
            elif imgs.ndim == 2:
                # (row, col) -> (1, 1, row, col)
                imgs = imgs[np.newaxis, np.newaxis]
            else:
                raise ValueError(f"Unsupported number of dimensions in '{file_path}': {imgs.ndim} with {imgs.shape=}")
    else:
        imgs = _standardize_slice_array(imgs)

    # remove any cached data to free up memory
    uncache = getattr(nibdata, 'uncache', None)
    if callable(uncache):
        uncache()
    return imgs, nibdata


def read_nifti_slice(file_path: str | Path | BinaryIO,
                     slice_index: int,
                     *,
                     mimetype: str | None = None,
                     slice_axis: int | None = None,
                     plane: ViewPlane | None = None) -> tuple[np.ndarray, SpatialImage]:
    """Read a single standardized slice from a NIfTI file.

    The returned slice follows the same normalized layout used across the
    package: (C, H, W), where grayscale slices have C=1.
    """
    return read_nifti(file_path,
                      mimetype=mimetype,
                      slice_index=slice_index,
                      slice_axis=slice_axis,
                      plane=plane)


def slice_location_to_slice_index(data: SpatialImage,
                                  slice_location: float,
                                  slice_axis: int,
                                  ) -> int:
    """
    Convert a slice location in world coordinates to a slice index in the NIfTI image.
    """
    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1 or 2")

    affine = _get_affine(data)
    origin = affine[:3, 3]  # Location at voxel [0, 0, 0] in world coordinates. (translation vector)
    rotation_matrix = affine[:3, :3]

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

    rotation_matrix = _get_affine(data)[:3, :3]
    coplanar_vector = coplanar_vector / np.linalg.norm(coplanar_vector)  # Normalize the vector

    # Find the slice axis that is most aligned with the coplanar vector
    dot_products = np.abs(rotation_matrix.T @ coplanar_vector)
    slice_axis = np.argmin(dot_products)

    return int(slice_axis)


def get_slice_location_from_slice_axis(data: SpatialImage,
                                       world_point: np.ndarray,
                                       slice_axis: int) -> float:
    """    Get the slice location in world coordinates from a point and the slice axis.
    """
    if not isinstance(world_point, np.ndarray) or world_point.ndim != 1 or world_point.size != 3:
        raise ValueError("world_point must be a 3-element numpy array")

    if slice_axis not in (0, 1, 2):
        raise ValueError("slice_axis must be 0, 1 or 2")

    rotation_matrix = _get_affine(data)[:3, :3]
    axis_vector = rotation_matrix[:, slice_axis]  # This is the direction of the slice axis in world coordinates
    world_slice_axis = np.argmax(np.abs(axis_vector))
    return world_point[world_slice_axis]


def line_to_slice_index(data: SpatialImage,
                        world_point1: np.ndarray | None = None,
                        world_point2: np.ndarray | None = None,
                        coplanar_vector: np.ndarray | None = None) -> tuple[int, int]:
    """
    Convert a line defined by two points OR coplanar_vector in world coordinates to a slice index and slice axis.
    IMPORTANT: Assumes the line is coplanar with the image plane (i.e., not oblique and aligned with the image axes).
    """
    # either world_point1 and world_point2 must be provided, or coplanar_vector must be provided
    if (world_point1 is None or world_point2 is None) and coplanar_vector is None:
        raise ValueError("Either world_point1 and world_point2 or coplanar_vector must be provided")

    if world_point1 is not None:
        if world_point2 is None:
            raise ValueError("world_point2 must be provided when world_point1 is provided")
        coplanar_vector = world_point2 - world_point1

    if world_point1 is None or coplanar_vector is None:
        raise ValueError("world_point1 and coplanar_vector are required to compute the slice index")

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
    return get_slice(data, slice_index, slice_axis=slice_axis)


def get_slice(data: SpatialImage,
              slice_index: int,
              slice_axis: int | None = None,
              plane: ViewPlane | None = None) -> np.ndarray:
    """
    Get a slice from a 3D or 4D NIfTI volume based on the slice index and axis.

    Args:
        data (SpatialImage): The NIfTI image data whose slice is to be extracted.
        slice_index (int): The index of the slice to extract.
        slice_axis (int | None): The spatial axis along which to extract the
            slice (0 for x, 1 for y, 2 for z).
        plane (str | None): Optional anatomical plane name used instead of
            ``slice_axis``.

    Returns:
        np.ndarray: The extracted slice. For 3D data the result is 2D. For 4D
            data the trailing dimension is preserved.
    """
    _, resolved_slice_axis, reverse_slice_order = _resolve_slice_context(
        data,
        slice_axis=slice_axis,
        plane=plane,
        default_plane=DEFAULT_SLICE_PLANE,
    )
    if resolved_slice_axis is None:
        raise ValueError("Could not resolve a slice axis for this NIfTI image")

    raw_slice_index = _normalize_slice_index(
        slice_index,
        int(data.shape[resolved_slice_axis]),
        reverse_slice_order,
    )

    slicer: list[slice | int] = [slice(None)] * len(data.shape)
    slicer[resolved_slice_axis] = int(raw_slice_index)
    indexer = tuple(slicer)

    try:
        return np.asarray(data.dataobj[indexer])
    except Exception:
        return np.asarray(data.get_fdata()[indexer])


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


def get_plane_axis(data: SpatialImage,
                   plane: ViewPlane) -> int:
    """
    Maps an anatomical plane ('axial', 'sagittal', 'coronal') to its 
    corresponding axis index (0, 1, or 2) in the get_fdata() numpy array.

    Args:
        data (SpatialImage): The NIfTI image data.
        plane (str): The anatomical plane to map ('axial', 'sagittal', 'coronal').

    Returns:
        int: The axis index corresponding to the specified plane (0, 1, or 2).

    """
    return _get_plane_axis(data, plane)


axis_name_to_axis_index = get_plane_axis  # alias for backward compatibility


def get_dim_size(data: SpatialImage,
                 axis_index: int | ViewPlane) -> int:
    """
    Get the size of a specific dimension (axis) from a NIfTI image.

    Reads only the header — voxel data is never loaded into memory.

    Args:
        data (SpatialImage): The NIfTI image.
        axis_index (int | str): Index of the axis, or an anatomical plane name
            ('axial', 'sagittal', 'coronal') which is resolved via ``get_plane_axis``.

    Returns:
        int: Size of the specified dimension.
    """
    if isinstance(axis_index, str):
        axis_index = get_plane_axis(data, axis_index)

    return int(data.shape[axis_index])


def get_nifti_shape(file_path: str) -> tuple[int, ...]:
    """
    Get the shape of a NIfTI file.

    Args:
        file_path (str): Path to the NIfTI file (.nii or .nii.gz)

    Returns:
        tuple[int, ...]: Shape of the NIfTI image (X, Y, Z)
    """
    try:
        return _load_spatial_image(file_path).shape
    except ImageFileError as e:
        from .format_detection import guess_type
        mimetype, _ = guess_type(file_path)
        if mimetype is None:
            raise
        if mimetype in GZIP_MIME_TYPES:
            with gzip.open(file_path, 'rb') as f:
                nibdata = Nifti1Image.from_stream(f)
                return nibdata.shape
        elif mimetype in NIFTI_MIMES:
            with open(file_path, 'rb') as f:
                nibdata = Nifti1Image.from_stream(f)
                return nibdata.shape
        else:
            raise


def _normalize_point_array(points: np.ndarray) -> tuple[np.ndarray, bool]:
    points = np.asarray(points, dtype=np.float64)

    if points.ndim == 1:
        if points.size != 3:
            raise ValueError("points must be of shape (3,) for a single point.")
        return points[np.newaxis, :], True

    if points.ndim == 2:
        if points.shape[1] != 3:
            raise ValueError("points must be of shape (N, 3) for multiple points.")
        return points, False

    raise ValueError("points must be either a 1D or 2D numpy array.")


def voxel_to_world(data: SpatialImage,
                   voxel_coords: np.ndarray) -> np.ndarray:
    """
    Convert raw voxel coordinates to world coordinates using the NIfTI affine.

    Args:
        data (SpatialImage): The NIfTI image data.
        voxel_coords (np.ndarray): Raw voxel coordinates of shape (N, 3), for
            multiple points, or (3,), for a single point.

    Returns:
        np.ndarray: World coordinates of shape (N, 3) or (3,).
    """
    voxel_coords, single_point = _normalize_point_array(voxel_coords)

    affine = _get_affine(data)
    points_hom = np.hstack((voxel_coords, np.ones((voxel_coords.shape[0], 1), dtype=np.float64)))
    world_points = points_hom @ affine.T
    world_points = world_points[:, :3]

    if single_point:
        return world_points[0]
    return world_points


def _pixel_to_raw_voxel_coords(data: SpatialImage,
                               pixel_x: float | np.ndarray,
                               pixel_y: float | np.ndarray,
                               slice_index: int | np.ndarray | None = None,
                               slice_axis: int | None = None,
                               plane: ViewPlane | None = None) -> tuple[np.ndarray, bool]:
    if len(data.shape) < 2:
        raise ValueError("NIfTI image must be at least 2D to convert pixel coordinates.")

    pixel_x_arr = np.asarray(pixel_x, dtype=np.float64)
    pixel_y_arr = np.asarray(pixel_y, dtype=np.float64)

    if len(data.shape) == 2:
        if slice_axis is not None or plane is not None:
            raise ValueError("slice_axis and plane are only supported for 3D or higher NIfTI images.")

        if slice_index is None:
            slice_index = 0

        pixel_x_arr, pixel_y_arr, slice_index_arr = np.broadcast_arrays(
            np.atleast_1d(pixel_x_arr),
            np.atleast_1d(pixel_y_arr),
            np.atleast_1d(np.asarray(slice_index, dtype=np.float64)),
        )

        if not np.allclose(slice_index_arr, 0.0):
            raise ValueError("slice_index must be 0 for 2D NIfTI images.")

        voxel_coords = np.zeros((pixel_x_arr.size, 3), dtype=np.float64)
        voxel_coords[:, 0] = pixel_x_arr.reshape(-1)
        voxel_coords[:, 1] = pixel_y_arr.reshape(-1)
        return voxel_coords, voxel_coords.shape[0] == 1

    _, resolved_slice_axis, reverse_order = _resolve_slice_context(
        data,
        slice_axis=slice_axis,
        plane=plane,
        default_plane=DEFAULT_SLICE_PLANE,
    )
    if resolved_slice_axis is None:
        raise ValueError("Could not resolve a slice axis for this NIfTI image.")

    if slice_index is None:
        if int(data.shape[resolved_slice_axis]) != 1:
            raise ValueError("slice_index is required for 3D or 4D NIfTI images.")
        slice_index = 0

    pixel_x_arr, pixel_y_arr, slice_index_arr = np.broadcast_arrays(
        np.atleast_1d(pixel_x_arr),
        np.atleast_1d(pixel_y_arr),
        np.atleast_1d(np.asarray(slice_index, dtype=np.float64)),
    )

    axis_size = int(data.shape[resolved_slice_axis])
    raw_slice_indices = _normalize_slice_index(slice_index_arr, axis_size, reverse_order).reshape(-1)

    remaining_axes = [axis for axis in range(2, -1, -1) if axis != resolved_slice_axis]
    voxel_coords = np.zeros((pixel_x_arr.size, 3), dtype=np.float64)
    voxel_coords[:, resolved_slice_axis] = raw_slice_indices
    voxel_coords[:, remaining_axes[1]] = pixel_x_arr.reshape(-1)
    voxel_coords[:, remaining_axes[0]] = pixel_y_arr.reshape(-1)
    return voxel_coords, voxel_coords.shape[0] == 1


def pixel_to_world(data: SpatialImage,
                   pixel_x: float | np.ndarray,
                   pixel_y: float | np.ndarray,
                   slice_index: int | np.ndarray | None = None,
                   slice_axis: int | None = None,
                   plane: ViewPlane | None = None) -> np.ndarray:
    """
    Convert slice-style pixel coordinates to world coordinates for a NIfTI image.

    The input coordinates follow the same conventions used by ``read_nifti`` and
    ``get_slice`` in this module:

    - When ``plane`` is used, slice ordering follows the affine orientation.
    - When ``slice_axis`` is used explicitly, raw voxel order is preserved.
    - ``pixel_x`` is the column index and ``pixel_y`` is the row index in the
      standardized slice image.

    Args:
        data (SpatialImage): The NIfTI image data.
        pixel_x (float | np.ndarray): X coordinate in slice pixel space.
        pixel_y (float | np.ndarray): Y coordinate in slice pixel space.
        slice_index (int | np.ndarray | None): Slice index in the standardized
            slice order. Required for 3D or 4D images unless the selected slice
            axis has length 1.
        slice_axis (int | None): Explicit raw spatial axis (0, 1, or 2).
        plane (ViewPlane | None): Anatomical plane used instead of
            ``slice_axis``.

    Returns:
        np.ndarray: World coordinates of shape (N, 3) or (3,).
    """
    voxel_coords, single_point = _pixel_to_raw_voxel_coords(
        data,
        pixel_x,
        pixel_y,
        slice_index=slice_index,
        slice_axis=slice_axis,
        plane=plane,
    )
    return voxel_to_world(data, voxel_coords[0] if single_point else voxel_coords)


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

    world_coords, single_point = _normalize_point_array(world_coords)

    # 1. Convert world coordinates to voxel coordinates
    inv_affine = np.linalg.inv(_get_affine(data))
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
