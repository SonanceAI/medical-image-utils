import pytest
import pydicom
import pydicom.uid
import numpy as np

from medimgkit.dicom_utils import anonymize_dicom, CLEARED_STR, is_dicom, TokenMapper, build_affine_matrix
import pydicom.data
from io import BytesIO
import warnings

class TestDicomUtils:
    @pytest.fixture
    def sample_dataset1(self):
        ds = pydicom.Dataset()
        ds.PatientName = "John Doe"
        ds.PatientID = "12345"
        ds.Modality = "CT"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        return ds

    def test_anonymize_dicom(self, sample_dataset1):
        # Create a sample DICOM dataset
        ds = sample_dataset1

        # Call the anonymize_dicom function
        anonymized_ds = anonymize_dicom(ds, copy=True)

        # Check if the specified DICOM tags are cleared
        assert anonymized_ds.PatientName != ds.PatientName
        assert anonymized_ds.PatientID != ds.PatientID
        assert anonymized_ds.Modality == ds.Modality
        # Check if the SOPInstanceUID and MediaStorageSOPInstanceUID are changed
        assert anonymized_ds.SOPInstanceUID != ds.SOPInstanceUID

    def test_anonymize_dicom_with_retain_codes(self, sample_dataset1):
        # Create a sample DICOM dataset
        ds = sample_dataset1

        # Specify the retain codes
        retain_codes = [(0x0010, 0x0020)]

        # Call the anonymize_dicom function
        anonymized_ds = anonymize_dicom(ds, copy=False, retain_codes=retain_codes)

        # Check if the specified DICOM tags are retained
        assert anonymized_ds.PatientName == CLEARED_STR
        assert anonymized_ds.PatientID == '12345'
        assert anonymized_ds.Modality == 'CT'

    def test_isdicom(self):
        dcmpaths = pydicom.data.get_testdata_files('**/*')

        for dcmpath in dcmpaths:
            if dcmpath.endswith('.dcm'):
                assert is_dicom(dcmpath) == True

        assert is_dicom('tests/test_dicom_utils.py') == False

        ## test empty data ##
        assert is_dicom(BytesIO()) == False

    @pytest.fixture
    def complex_dataset(self):
        """Create a dataset with various VR types and special cases"""
        ds = pydicom.Dataset()
        ds.PatientName = "Jane Smith"
        ds.PatientID = "67890"
        ds.PatientBirthDate = "19850315"
        ds.PatientSex = "F"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
        
        # Phone number (special case)
        ds.add_new((0x0008, 0x0094), 'SH', '555-123-4567')  # ReferringPhysicianTelephoneNumbers
        
        # Floating point values
        ds.add_new((0x0018, 0x0050), 'DS', '5.0')  # SliceThickness (DS)
        ds.add_new((0x0028, 0x0030), 'DS', ['1.5', '1.5'])  # PixelSpacing (DS)
        ds.add_new((0x0018, 0x1316), 'FL', 90.5)  # SAR (FL)
        ds.add_new((0x0018, 0x1318), 'FD', 123.456789)  # dB/dt (FD)
        
        # Sequence (should be deleted)
        seq_dataset = pydicom.Dataset()
        seq_dataset.PatientName = "Sequence Patient"
        ds.add_new((0x0008, 0x1140), 'SQ', [seq_dataset])  # ReferencedImageSequence
        
        # File meta
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        
        return ds

    def test_anonymize_dicom_phone_number_special_case(self, complex_dataset):
        """Test that phone numbers are set to '000-000-0000'"""
        ds = complex_dataset
        anonymized_ds = anonymize_dicom(ds, copy=True)
        
        phone_tag = (0x0008, 0x0094)
        assert anonymized_ds[phone_tag].value == "000-000-0000"

    def test_anonymize_dicom_consistent_tokenization(self):
        """Test that same values get same tokens across multiple calls"""
        ds1 = pydicom.Dataset()
        ds1.PatientID = "SAME_ID"
        ds1.StudyInstanceUID = "1.2.3.4.5"
        
        ds2 = pydicom.Dataset()
        ds2.PatientID = "SAME_ID"
        ds2.StudyInstanceUID = "1.2.3.4.5"
        
        token_mapper = TokenMapper(seed=42)
        
        anon_ds1 = anonymize_dicom(ds1, copy=True, token_mapper=token_mapper)
        anon_ds2 = anonymize_dicom(ds2, copy=True, token_mapper=token_mapper)
        
        # Same original values should get same tokens
        assert anon_ds1.PatientID == anon_ds2.PatientID
        assert anon_ds1.StudyInstanceUID == anon_ds2.StudyInstanceUID

    def test_anonymize_dicom_file_meta_update(self, complex_dataset):
        """Test that file_meta.MediaStorageSOPInstanceUID is updated"""
        ds = complex_dataset
        original_sop_uid = ds.SOPInstanceUID
        
        anonymized_ds = anonymize_dicom(ds, copy=True)
        
        # SOPInstanceUID should be changed
        assert anonymized_ds.SOPInstanceUID != original_sop_uid
        
        # file_meta should be updated to match
        assert hasattr(anonymized_ds, 'file_meta')
        assert anonymized_ds.file_meta.MediaStorageSOPInstanceUID == anonymized_ds.SOPInstanceUID

    def test_anonymize_dicom_no_file_meta(self, sample_dataset1):
        """Test anonymization when no file_meta exists"""
        ds = sample_dataset1
        # Ensure no file_meta
        if hasattr(ds, 'file_meta'):
            delattr(ds, 'file_meta')
        
        # Should not raise exception
        anonymized_ds = anonymize_dicom(ds, copy=True)
        assert anonymized_ds.PatientName == CLEARED_STR

    def test_anonymize_dicom_no_sop_instance_uid(self):
        """Test anonymization when SOPInstanceUID is missing"""
        ds = pydicom.Dataset()
        ds.PatientName = "Test Patient"
        # No SOPInstanceUID
        
        ds.file_meta = pydicom.Dataset()
        ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5"
        
        # Should not raise exception
        anonymized_ds = anonymize_dicom(ds, copy=True)
        assert anonymized_ds.PatientName == CLEARED_STR

    def test_anonymize_dicom_retain_codes_comprehensive(self, complex_dataset):
        """Test retain_codes with various tag types"""
        ds = complex_dataset
        
        retain_codes = [
            (0x0010, 0x0020),  # PatientID
            (0x0008, 0x0094),  # Phone number
            (0x0018, 0x0050),  # SliceThickness (DS)
        ]
        
        original_patient_id = ds.PatientID
        original_phone = ds[(0x0008, 0x0094)].value
        original_thickness = ds[(0x0018, 0x0050)].value
        
        anonymized_ds = anonymize_dicom(ds, copy=True, retain_codes=retain_codes)
        
        # Retained values should be unchanged
        assert anonymized_ds.PatientID == original_patient_id
        assert anonymized_ds[(0x0008, 0x0094)].value == original_phone
        assert anonymized_ds[(0x0018, 0x0050)].value == original_thickness
        
        # Non-retained values should be cleared/anonymized
        assert anonymized_ds.PatientName == CLEARED_STR

    def test_anonymize_dicom_cleared_str_values(self):
        """Test handling of values that are already CLEARED_STR"""
        ds = pydicom.Dataset()
        ds.PatientName = CLEARED_STR
        ds.PatientID = "12345"
        
        token_mapper = TokenMapper()
        anonymized_ds = anonymize_dicom(ds, copy=True, token_mapper=token_mapper)
        
        # Already cleared values should remain CLEARED_STR
        assert anonymized_ds.PatientName == CLEARED_STR
        # Other values should still be processed
        assert anonymized_ds.PatientID != "12345"

    def test_anonymize_dicom_none_values(self):
        """Test handling of None values in tags"""
        ds = pydicom.Dataset()
        ds.add_new((0x0010, 0x0010), 'PN', None)  # PatientName as None
        ds.PatientID = "12345"
        
        token_mapper = TokenMapper()
        
        # Should not raise exception
        anonymized_ds = anonymize_dicom(ds, copy=True, token_mapper=token_mapper)
        
        # None values should become CLEARED_STR for UID tags, or remain None
        patient_name_tag = (0x0010, 0x0010)
        if patient_name_tag in anonymized_ds:
            # Value should be cleared
            assert anonymized_ds[patient_name_tag].value == CLEARED_STR

    def test_token_mapper_simple_id_vs_uid(self):
        """Test TokenMapper generates different formats for simple_id vs UID"""
        mapper = TokenMapper(seed=42)
        
        tag = (0x0010, 0x0020)
        value = "TEST123"
        
        simple_token = mapper.get_token(tag, value, simple_id=True)
        uid_token = mapper.get_token(tag, value, simple_id=False)
        
        # Simple token should be different from UID token
        assert simple_token != uid_token
        # UID token should contain dots (UID format)
        assert '.' in uid_token
        # Simple token should be a hash (no dots typically)
        assert '.' not in simple_token

    def test_build_affine_matrix_single_slice(self):
        ds = pydicom.Dataset()
        ds.ImagePositionPatient = [10.0, 20.0, 30.0]
        # row dir then col dir
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0]
        # PixelSpacing: [row_spacing, col_spacing]
        ds.PixelSpacing = [2.0, 3.0]
        ds.SpacingBetweenSlices = 4.0

        aff = build_affine_matrix(ds)
        expected = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 3.0, 0.0, 20.0],
            [0.0, 0.0, 4.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        assert np.allclose(aff, expected)

    def test_build_affine_matrix_multiframe_perframe_sequences(self):
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 3

        # Shared pixel spacing is optional; we set via per-frame PixelMeasuresSequence
        perframe = []
        for i in range(int(ds.NumberOfFrames)):
            frame = pydicom.Dataset()

            pos_seq_item = pydicom.Dataset()
            pos_seq_item.ImagePositionPatient = [0.0, 0.0, float(i) * 5.0]
            frame.PlanePositionSequence = [pos_seq_item]

            orient_seq_item = pydicom.Dataset()
            orient_seq_item.ImageOrientationPatient = [1.0, 0.0, 0.0,
                                                       0.0, 1.0, 0.0]
            frame.PlaneOrientationSequence = [orient_seq_item]

            meas_seq_item = pydicom.Dataset()
            meas_seq_item.PixelSpacing = [1.0, 1.0]
            meas_seq_item.SpacingBetweenSlices = 5.0
            frame.PixelMeasuresSequence = [meas_seq_item]

            perframe.append(frame)

        ds.PerFrameFunctionalGroupsSequence = perframe

        aff = build_affine_matrix(ds)
        expected = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        assert np.allclose(aff, expected)

    def test_build_affine_matrix_raises_on_inconsistent_orientation(self):
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 2

        f0 = pydicom.Dataset()
        f0.PlanePositionSequence = [pydicom.Dataset()]
        f0.PlanePositionSequence[0].ImagePositionPatient = [0.0, 0.0, 0.0]
        f0.PlaneOrientationSequence = [pydicom.Dataset()]
        f0.PlaneOrientationSequence[0].ImageOrientationPatient = [1.0, 0.0, 0.0,
                                                                  0.0, 1.0, 0.0]
        f0.PixelMeasuresSequence = [pydicom.Dataset()]
        f0.PixelMeasuresSequence[0].PixelSpacing = [1.0, 1.0]
        f0.PixelMeasuresSequence[0].SpacingBetweenSlices = 5.0

        f1 = pydicom.Dataset()
        f1.PlanePositionSequence = [pydicom.Dataset()]
        f1.PlanePositionSequence[0].ImagePositionPatient = [0.0, 0.0, 5.0]
        f1.PlaneOrientationSequence = [pydicom.Dataset()]
        # swapped/rotated orientation
        f1.PlaneOrientationSequence[0].ImageOrientationPatient = [0.0, 1.0, 0.0,
                                                                  1.0, 0.0, 0.0]
        f1.PixelMeasuresSequence = [pydicom.Dataset()]
        f1.PixelMeasuresSequence[0].PixelSpacing = [1.0, 1.0]
        f1.PixelMeasuresSequence[0].SpacingBetweenSlices = 5.0

        ds.PerFrameFunctionalGroupsSequence = [f0, f1]

        with pytest.raises(ValueError):
            build_affine_matrix(ds)