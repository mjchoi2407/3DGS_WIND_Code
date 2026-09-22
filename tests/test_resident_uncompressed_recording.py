"""무압축 기록의 저장 경계·상태 값·기존 loader 호환성 확인."""
import unittest
import zipfile
import numpy as np
from wind3dgs.teacher.resident_uncompressed_recording import uncompressed_recording
from test_gpu_state_recording import StateRecordingTests

class UncompressedTests(StateRecordingTests):
    def test_transfer_only_at_interval_and_final_flush(self):
        original=np.savez_compressed
        with uncompressed_recording():
            self.assertIs(np.savez_compressed,np.savez)
            super().test_transfer_only_at_interval_and_final_flush()
        self.assertIs(np.savez_compressed,original)

    def test_zip_members_are_stored(self):
        import io
        data=np.arange(1000,dtype=np.float64);buffer=io.BytesIO()
        with uncompressed_recording():np.savez_compressed(buffer,u_hi=data,state_encoding=np.array('resident_pair_f64_v1'))
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:self.assertTrue(all(i.compress_type==zipfile.ZIP_STORED for i in archive.infolist()))
        buffer.seek(0)
        with np.load(buffer,allow_pickle=False) as z:np.testing.assert_array_equal(z['u_hi'],data)

if __name__=='__main__':unittest.main()
