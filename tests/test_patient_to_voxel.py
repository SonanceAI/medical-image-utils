import pytest
import pydicom
import numpy as np
from medimgkit.dicom_utils import patient_to_voxel, pixel_to_patient

class TestPatientToVoxel:
    def test_single_slice(self):
        ds = pydicom.Dataset()
        ds.ImagePositionPatient = [0, 0, 0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0] # Identity
        ds.PixelSpacing = [1.0, 1.0]
        ds.SpacingBetweenSlices = 1.0
        ds.NumberOfFrames = 1
        
        # Point at (10, 20, 0) -> pixel (10, 20) on slice 0
        pt = np.array([10.0, 20.0, 0.0])
        vox = patient_to_voxel(ds, pt)
        
        assert np.allclose(vox, [10.0, 20.0, 0.0])
        
        # Point at (10, 20, 5) -> pixel (10, 20) on slice 0 (nearest), dist 5
        # My implementation returns k=0.0
        pt2 = np.array([10.0, 20.0, 5.0])
        vox2 = patient_to_voxel(ds, pt2)
        assert np.allclose(vox2, [10.0, 20.0, 0.0])

    def test_multi_slice_regular(self):
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 3
        
        # Create PerFrameFunctionalGroupsSequence
        ds.PerFrameFunctionalGroupsSequence = pydicom.Sequence()
        for i in range(3):
            frame = pydicom.Dataset()
            
            plane_pos = pydicom.Dataset()
            plane_pos.ImagePositionPatient = [0, 0, i * 2.0] # Spacing 2.0
            frame.PlanePositionSequence = pydicom.Sequence([plane_pos])
            
            plane_orient = pydicom.Dataset()
            plane_orient.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            frame.PlaneOrientationSequence = pydicom.Sequence([plane_orient])
            
            pixel_measures = pydicom.Dataset()
            pixel_measures.PixelSpacing = [0.5, 0.5]
            frame.PixelMeasuresSequence = pydicom.Sequence([pixel_measures])
            
            ds.PerFrameFunctionalGroupsSequence.append(frame)
            
        # Point at (5, 5, 2.0) -> Slice 1 (z=2.0). 
        # x=5 -> pixel_x = 5 / 0.5 = 10
        # y=5 -> pixel_y = 5 / 0.5 = 10
        
        pt = np.array([5.0, 5.0, 2.0])
        vox = patient_to_voxel(ds, pt)
        assert np.allclose(vox, [10.0, 10.0, 1.0])
        
        # Point at (5, 5, 3.0) -> Slice 1 (z=2.0) or Slice 2 (z=4.0)?
        # Dist to slice 1: 1.0. Dist to slice 2: 1.0.
        # Argmin might pick first one (slice 1).
        pt2 = np.array([5.0, 5.0, 3.0])
        vox2 = patient_to_voxel(ds, pt2)
        # k should be 1.0 or 2.0
        assert vox2[2] in [1.0, 2.0]

    def test_pixel_to_patient_vectorized(self):
        ds = pydicom.Dataset()
        ds.ImagePositionPatient = [0, 0, 0]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.NumberOfFrames = 1
        
        px = np.array([0, 10])
        py = np.array([0, 20])
        
        res = pixel_to_patient(ds, px, py, slice_index=0)
        expected = np.array([[0, 0, 0], [10, 20, 0]])
        assert np.allclose(res, expected)

    def test_going_back_and_forth(self):
        ds = pydicom.Dataset()
        ds.NumberOfFrames = 5
        
        ds.PerFrameFunctionalGroupsSequence = pydicom.Sequence()
        for i in range(5):
            frame = pydicom.Dataset()
            
            plane_pos = pydicom.Dataset()
            plane_pos.ImagePositionPatient = [0, 0, i * 3.0]
            frame.PlanePositionSequence = pydicom.Sequence([plane_pos])
            
            plane_orient = pydicom.Dataset()
            plane_orient.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            frame.PlaneOrientationSequence = pydicom.Sequence([plane_orient])
            
            pixel_measures = pydicom.Dataset()
            pixel_measures.PixelSpacing = [2.0, 2.0]
            frame.PixelMeasuresSequence = pydicom.Sequence([pixel_measures])
            
            ds.PerFrameFunctionalGroupsSequence.append(frame)
        
        points_patient = np.array([
            [4.0, 6.0, 3.0],
            [10.0, 10.0, 9.0],
            [0.0, 0.0, 12.0]
        ])
        
        vox_coords = patient_to_voxel(ds, points_patient)
        reconverted_pts = pixel_to_patient(
            ds,
            vox_coords[:, 0],
            vox_coords[:, 1],
            slice_index=vox_coords[:, 2].astype(int)
        )
        
        assert np.allclose(points_patient, reconverted_pts)