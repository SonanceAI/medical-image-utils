from io import BytesIO

import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset

from medimgkit.format_detection import guess_type
from medimgkit.io_utils import peek


class TestFormatDetection:
    def _make_dicom_stream(self) -> BytesIO:
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        ds = FileDataset("stream.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.PatientName = "Test^Patient"
        ds.PatientID = "12345"

        stream = BytesIO()
        ds.save_as(stream, enforce_file_format=True)
        stream.seek(0)
        return stream

    def test_peek_restores_original_stream_position(self):
        stream = BytesIO(b"abcdef")
        stream.seek(2)

        with peek(stream) as handle:
            assert handle.read(2) == b"cd"

        assert stream.tell() == 2
        assert stream.read(2) == b"cd"

    def test_guess_type_detects_dicom_stream_without_consuming_it(self):
        stream = self._make_dicom_stream()

        mime_type, _ = guess_type(stream, force_magic=True)

        assert mime_type == "application/dicom"