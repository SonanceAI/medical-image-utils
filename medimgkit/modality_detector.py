"""
Modality detection utilities for medical imaging files.
"""

import logging
from pathlib import Path
from typing import Optional, Union
import pydicom
import nibabel as nib

_LOGGER = logging.getLogger(__name__)


class ModalityDetector:
    """Detector for medical imaging modalities."""

    KNOWN_MODALITIES = {
        'CT': 'Computed Tomography',
        'MR': 'Magnetic Resonance',
        'PT': 'Positron Emission Tomography',
        'US': 'Ultrasound',
        'CR': 'Computed Radiography',
        'DX': 'Digital Radiography',
        'MG': 'Mammography',
        'XA': 'X-Ray Angiography',
        'NM': 'Nuclear Medicine',
        'OT': 'Other',
        'RF': 'Radio Fluoroscopy',
        'SC': 'Secondary Capture',
        'RTIMAGE': 'Radiotherapy Image',
        'RTDOSE': 'Radiotherapy Dose',
        'RTSTRUCT': 'Radiotherapy Structure Set',
        'RTPLAN': 'Radiotherapy Plan',
    }

    @staticmethod
    def detect_modality(file_path: Union[str, Path]) -> Optional[str]:
        """
        Detect the imaging modality of a medical file.

        Args:
            file_path: Path to the medical imaging file

        Returns:
            Modality code (e.g., 'CT', 'MR', 'PT') or None if detection fails

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try DICOM first
        try:
            modality = ModalityDetector._detect_dicom_modality(file_path)
            if modality:
                return modality
        except Exception:
            pass

        # Try NIfTI
        try:
            modality = ModalityDetector._detect_nifti_modality(file_path)
            if modality:
                return modality
        except Exception:
            pass

        return None

    @staticmethod
    def _detect_dicom_modality(file_path: Path) -> Optional[str]:
        """
        Detect modality from DICOM file.

        Args:
            file_path: Path to DICOM file

        Returns:
            Modality code or None
        """
        try:
            dcm = pydicom.dcmread(str(file_path),
                                  specific_tags=['Modality', 'SOPClassUID'],
                                  stop_before_pixels=True)

            # Primary method: Check Modality tag (0008,0060)
            if hasattr(dcm, 'Modality'):
                return dcm.Modality

            # Fallback: Check SOPClassUID for modality hints
            if hasattr(dcm, 'SOPClassUID'):
                sop_class = dcm.SOPClassUID
                if 'CT' in sop_class:
                    return 'CT'
                elif 'MR' in sop_class or 'Magnetic' in sop_class:
                    return 'MR'
                elif 'PET' in sop_class or 'Positron' in sop_class:
                    return 'PT'
                elif 'Ultrasound' in sop_class:
                    return 'US'
                elif 'RT' in sop_class:
                    if 'Dose' in sop_class:
                        return 'RTDOSE'
                    elif 'Structure' in sop_class:
                        return 'RTSTRUCT'
                    elif 'Plan' in sop_class:
                        return 'RTPLAN'
                    elif 'Image' in sop_class:
                        return 'RTIMAGE'

            return None
        except Exception:
            return None

    @staticmethod
    def _detect_nifti_modality(file_path: Path) -> Optional[str]:
        """
        Detect modality from NIfTI file.

        Note: NIfTI files don't have standard modality tags,
        so detection is based on filename patterns and description.

        Args:
            file_path: Path to NIfTI file

        Returns:
            Modality code or None
        """
        try:
            # Check file extension
            if not (file_path.suffix in ['.nii', '.gz'] or
                    str(file_path).endswith('.nii.gz')):
                return None

            img = nib.load(str(file_path))
            header = img.header

            # Try to get description from header
            description = ''
            if hasattr(header, 'get_data_dtype'):
                description = str(header.get('descrip', b'')).lower()

            # Check filename and description for modality hints
            filename_lower = file_path.name.lower()
            combined_text = filename_lower + ' ' + description

            if any(x in combined_text for x in ['_ct_', 'computed_tomography']):
                return 'CT'
            elif any(x in combined_text for x in ['_mr_', '_mri_', 'magnetic', '_flair_']):
                return 'MR'
            elif any(x in combined_text for x in ['_pet_', 'positron']):
                return 'PT'
            elif any(x in combined_text for x in ['ultrasound']):
                return 'US'

            return None
        except Exception:
            _LOGGER.info(f"Failed to detect NIfTI modality for file: {file_path}")
            return None

    @staticmethod
    def get_modality_description(modality_code: str) -> str:
        """
        Get the full description of a modality code.

        Args:
            modality_code: Modality code (e.g., 'CT', 'MR')

        Returns:
            Full modality description or the code itself if unknown
        """
        return ModalityDetector.KNOWN_MODALITIES.get(
            modality_code,
            modality_code
        )


def detect_modality(file_path: Union[str, Path]) -> Optional[str]:
    """
    Convenience function to detect the imaging modality of a medical file.

    Args:
        file_path: Path to the medical imaging file

    Returns:
        Modality code (e.g., 'CT', 'MR', 'PT') or None if detection fails

    Example:
        >>> modality = detect_modality('patient_scan.dcm')
        >>> print(f"Detected modality: {modality}")
        Detected modality: CT
    """
    return ModalityDetector.detect_modality(file_path)


def get_modality_info(file_path: Union[str, Path]) -> dict:
    """
    Get detailed modality information for a medical file.

    Args:
        file_path: Path to the medical imaging file

    Returns:
        Dictionary with modality information including code and description

    Example:
        >>> info = get_modality_info('patient_scan.dcm')
        >>> print(info)
        {'code': 'CT', 'description': 'Computed Tomography', 'file': 'patient_scan.dcm'}
    """
    file_path = Path(file_path)
    modality_code = ModalityDetector.detect_modality(file_path)

    return {
        'code': modality_code,
        'description': ModalityDetector.get_modality_description(modality_code) if modality_code else None,
        'file': str(file_path)
    }
