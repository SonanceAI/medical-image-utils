# MedImgKit

A comprehensive toolkit for medical image processing, providing utilities for DICOM, NIfTI, and other medical image formats with seamless multi-format I/O operations.

## Features

- **DICOM Support**: Read, anonymize, and manipulate DICOM files
- **NIfTI Support**: Work with neuroimaging data in NIfTI format
- **Multi-format I/O**: Unified interface for reading various image formats
- **Anonymization**: DICOM anonymization following DICOM standards
- **Coordinate Conversion**: Convert between pixel and patient coordinates
- **Multi-frame Assembly**: Combine multiple DICOM files into multi-frame volumes

## Installation

### From PyPI
```bash
pip install medimgkit
```

### From Source
```bash
pip install git+https://github.com/SonanceAI/medimgkit
```

## Quick Start

### DICOM Operations
```python
import medimgkit as mik
import pydicom

# Read and normalize DICOM image
ds = pydicom.dcmread('path/to/dicom.dcm')
image_array = mik.load_image_normalized(ds)

# Anonymize DICOM
anonymized_ds = mik.anonymize_dicom(ds)

# Convert pixel coordinates to patient coordinates
patient_coords = mik.pixel_to_patient(ds, pixel_x=100, pixel_y=150)
```

### NIfTI Operations
```python
import nibabel as nib
import medimgkit as mik

# Load NIfTI file
nifti_data = nib.load('path/to/image.nii.gz')

# Get a specific slice
slice_image = mik.get_slice(nifti_data, slice_index=50, slice_axis=2)

# Convert world coordinates to slice index
slice_idx, axis = mik.line_to_slice_index(nifti_data, point1, point2)
```

### Multi-format Reading
```python
import medimgkit as mik

# Read any supported format
image_array = mik.read_array_normalized('path/to/image.dcm')
image_array = mik.read_array_normalized('path/to/image.nii.gz')
image_array = mik.read_array_normalized('path/to/image.png')
```

## API Reference

### DICOM Utils (`medimgkit.dicom_utils`)

#### Core Functions
- `load_image_normalized(dicom, index=None)`: Load and normalize DICOM pixel data
- `anonymize_dicom(ds, retain_codes=[], copy=False, token_mapper=None)`: Anonymize DICOM following standards
- `assemble_dicoms(files_path, return_as_IO=False)`: Combine multiple DICOMs into multi-frame
- `is_dicom(f)`: Check if file is a DICOM

#### Coordinate Conversion
- `pixel_to_patient(ds, pixel_x, pixel_y, slice_index=None)`: Convert pixel to patient coordinates
- `get_image_position(ds, slice_index=None)`: Get image position in patient coordinates
- `get_pixel_spacing(ds, slice_index)`: Get pixel spacing information

#### Anatomical Analysis
- `determine_anatomical_plane_from_dicom(ds, slice_axis, alignment_threshold=0.95)`: Determine anatomical plane

### NIfTI Utils (`medimgkit.nifti_utils`)

#### Slice Operations
- `get_slice(data, slice_index, slice_axis)`: Extract 2D slice from 3D volume
- `get_slice_from_line(data, world_point1, world_point2)`: Get slice defined by line
- `slice_location_to_slice_index(data, slice_location, slice_axis)`: Convert location to index

#### Coordinate Conversion
- `line_to_slice_index(data, world_point1=None, world_point2=None, coplanar_vector=None)`: Convert line to slice
- `axis_name_to_axis_index(data, axis_name)`: Convert axis name to index

#### Utilities
- `is_nifti_file(file_path)`: Check if file is NIfTI format

### I/O Utils (`medimgkit.io_utils`)

#### Reading Functions
- `read_array_normalized(file_path, index=None, return_metainfo=False, use_magic=False)`: Universal image reader
- `read_image(file_path)`: Read standard image formats (PNG, JPEG)
- `read_nifti(file_path, mimetype=None)`: Read NIfTI files
- `read_video(file_path, index=None)`: Read video files

## Supported Formats

- **DICOM**: .dcm, .dicom (and files without extension)
- **NIfTI**: .nii, .nii.gz
- **Images**: .png, .jpg, .jpeg
- **Video**: .mp4, .avi, .mov, .mkv
- **NumPy**: .npy

## Development

### Running Tests
```bash
pytest
```
## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request