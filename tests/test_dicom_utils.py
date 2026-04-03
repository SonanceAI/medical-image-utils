import pytest
import pydicom
import pydicom.uid
import numpy as np
import json
from pydicom.dataset import FileDataset, FileMetaDataset

from medimgkit.dicom_utils import assemble_dicoms, anonymize_dicom, CLEARED_STR, is_dicom, TokenMapper, build_affine_matrix
import pydicom.data
from io import BytesIO
import warnings

class TestDicomUtils:
    def _write_single_frame_dicom(self,
                                  tmp_path,
                                  filename: str,
                                  pixel_value: int,
                                  instance_number: int,
                                  sop_instance_uid: str,
                                  series_instance_uid: str,
                                  acquisition_time: str,
                                  private_creator: str | None = None) -> str:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID(sop_instance_uid)
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        file_path = tmp_path / filename
        ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = sop_instance_uid
        ds.StudyInstanceUID = '1.2.826.0.1.3680043.8.498.100'
        ds.SeriesInstanceUID = series_instance_uid
        ds.FrameOfReferenceUID = '1.2.826.0.1.3680043.8.498.300'
        ds.Modality = 'CT'
        ds.SeriesDate = '20260102'
        ds.SeriesTime = acquisition_time
        ds.AcquisitionTime = acquisition_time
        ds.SeriesDescription = 'Converted Test Series'
        ds.PatientName = 'Test^Patient'
        ds.PatientID = '12345'
        ds.ImageType = ['ORIGINAL', 'PRIMARY', 'AXIAL']
        ds.Rows = 2
        ds.Columns = 2
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 5.0
        ds.SpacingBetweenSlices = 5.0
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, float((instance_number - 1) * 5.0)]
        ds.InstanceNumber = instance_number
        ds.PixelData = np.full((2, 2), pixel_value, dtype=np.uint16).tobytes()

        if private_creator is not None:
            private_block = ds.private_block(0x0009, private_creator, create=True)
            private_block.add_new(0x01, 'LO', 'sentinel')

        ds.save_as(str(file_path), enforce_file_format=True)
        return str(file_path)

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
        ds.file_meta = FileMetaDataset()
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
        
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID("1.2.3.4.5")
        
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
        # Voxel coords are (pixel_x, pixel_y, slice_index) where:
        #   pixel_x = column index -> moves along row_dir by col_spacing (3.0)
        #   pixel_y = row index    -> moves along col_dir by row_spacing (2.0)
        expected = np.array([
            [3.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
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

    def test_assemble_dicoms_adds_legacy_conversion_metadata(self, tmp_path):
        original_series_uid = '1.2.826.0.1.3680043.8.498.200'
        sop_uid_1 = '1.2.826.0.1.3680043.8.498.1001'
        sop_uid_2 = '1.2.826.0.1.3680043.8.498.1002'

        path1 = self._write_single_frame_dicom(
            tmp_path,
            'slice1.dcm',
            1,
            1,
            sop_uid_1,
            original_series_uid,
            '090100.000000',
        )
        path2 = self._write_single_frame_dicom(
            tmp_path,
            'slice2.dcm',
            2,
            2,
            sop_uid_2,
            original_series_uid,
            '090200.000000',
        )

        merged = next(iter(assemble_dicoms([path1, path2], progress_bar=False)))

        assert merged.SOPClassUID == pydicom.uid.LegacyConvertedEnhancedCTImageStorage
        assert merged.file_meta.MediaStorageSOPClassUID == merged.SOPClassUID
        assert merged.file_meta.MediaStorageSOPInstanceUID == merged.SOPInstanceUID
        assert merged.SeriesInstanceUID != original_series_uid
        assert merged.SeriesTime == '090100.000000'
        assert merged.StudyInstanceUID == '1.2.826.0.1.3680043.8.498.100'

        referenced_uids = [item.ReferencedSOPInstanceUID for item in merged.SourceImageSequence]
        assert referenced_uids == [sop_uid_1, sop_uid_2]

        conversion_source_uids = [
            frame.ConversionSourceAttributesSequence[0].ReferencedSOPInstanceUID
            for frame in merged.PerFrameFunctionalGroupsSequence
        ]
        assert conversion_source_uids == [sop_uid_1, sop_uid_2]

        contributing_item = merged.ContributingEquipmentSequence[-1]
        purpose_item = contributing_item.PurposeOfReferenceCodeSequence[0]
        assert purpose_item.CodeValue == '109106'
        assert purpose_item.CodingSchemeDesignator == 'DCM'
        assert purpose_item.CodeMeaning == 'Enhanced Multi-frame Conversion Equipment'
        assert contributing_item.ContributionDescription == 'Legacy Enhanced Image created from Classic Images'

        shared_unassigned = merged.SharedFunctionalGroupsSequence[0].UnassignedSharedConvertedAttributesSequence[0]
        assert shared_unassigned.SeriesInstanceUID == original_series_uid
        assert list(shared_unassigned.ImageType) == ['ORIGINAL', 'PRIMARY', 'AXIAL']

        per_frame_acquisition_times = [
            frame.UnassignedPerFrameConvertedAttributesSequence[0].AcquisitionTime
            for frame in merged.PerFrameFunctionalGroupsSequence
        ]
        assert per_frame_acquisition_times == ['090100.000000', '090200.000000']

        medimgkit_block = merged.private_block(0x0009, 'MEDIMGKIT')
        mapping = json.loads(merged[medimgkit_block.get_tag(0x01)].value)
        assert mapping == [
            {
                'frame': 0,
                'sop_instance_uid': sop_uid_1,
                'filepath': path1,
            },
            {
                'frame': 1,
                'sop_instance_uid': sop_uid_2,
                'filepath': path2,
            },
        ]

    def test_assemble_dicoms_keeps_existing_private_creator_block(self, tmp_path):
        original_series_uid = '1.2.826.0.1.3680043.8.498.201'
        path1 = self._write_single_frame_dicom(
            tmp_path,
            'slice1_with_private.dcm',
            1,
            1,
            '1.2.826.0.1.3680043.8.498.2001',
            original_series_uid,
            '090100.000000',
            private_creator='EXISTING_CREATOR',
        )
        path2 = self._write_single_frame_dicom(
            tmp_path,
            'slice2_with_private.dcm',
            2,
            2,
            '1.2.826.0.1.3680043.8.498.2002',
            original_series_uid,
            '090200.000000',
        )

        merged = next(iter(assemble_dicoms([path1, path2], progress_bar=False)))

        medimgkit_block = merged.private_block(0x0009, 'MEDIMGKIT')
        perframe_private_ds = merged.PerFrameFunctionalGroupsSequence[0].UnassignedPerFrameConvertedAttributesSequence[0]
        existing_block = perframe_private_ds.private_block(0x0009, 'EXISTING_CREATOR')

        assert medimgkit_block.get_tag(0x01) in merged
        assert perframe_private_ds[existing_block.get_tag(0x01)].value in {'sentinel', b'sentinel'}