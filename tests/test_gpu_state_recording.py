import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.teacher.gpu_recording import GPURecordingPolicy
from wind3dgs.teacher.gpu_state_recording import ResidentStateRecorder


class StateRecordingTests(unittest.TestCase):
    def test_transfer_only_at_interval_and_final_flush(self):
        device = wp.get_device(os.environ.get('WIND3DGS_TEST_DEVICE','cpu'))
        # 2 Hz × 2 substeps로 경계 동작만 검증. 물리 계산 간격의 기본값은 유지한다.
        policy = GPURecordingPolicy(fps=2,substeps=2)
        arrays = [wp.array(np.full((3,3),i+.125),dtype=wp.vec3d,device=device) for i in range(4)]
        with tempfile.TemporaryDirectory() as folder:
            writer = ResidentStateRecorder(folder,mesh='test',nodes=3,remaining_frames=5,
                                           first_frame=6,policy=policy,device=device)
            with patch.object(wp.array,'numpy',side_effect=AssertionError('조기 전송')):
                writer.start(arrays)
                for _ in range(7):
                    writer.append(arrays)
            self.assertEqual(list(Path(folder).iterdir()),[])
            writer.append(arrays)
            self.assertEqual(len(writer.files),1)
            with patch.object(wp.array,'numpy',side_effect=AssertionError('조기 전송')):
                writer.append(arrays)
                writer.append(arrays)
            writer.finish()
            self.assertEqual(len(writer.files),2)
            for path,expected_times in zip(writer.files,[np.arange(12,21)/4,np.arange(20,23)/4]):
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['time_s'],expected_times)
                    for i,name in enumerate(writer.names):
                        np.testing.assert_array_equal(z[name],np.full((len(expected_times),3,3),i+.125))
            with self.assertRaises(ValueError):
                writer.append(arrays)


if __name__ == '__main__':
    unittest.main()
