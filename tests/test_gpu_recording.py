import unittest
from wind3dgs.teacher.gpu_recording import GPURecordingPolicy


class GPURecordingTests(unittest.TestCase):
    def test_default_keeps_all_steps_and_flushes_short_final_chunk(self):
        policy = GPURecordingPolicy()
        self.assertEqual(policy.frames_per_chunk('reference_rectangle'), 120)
        self.assertEqual(policy.chunks('reference_rectangle', 167, 250), [(167,120),(287,120),(407,10)])
        self.assertEqual(policy.chunks('reference_rectangle', 167, 10), [(167,10)])
        self.assertEqual(policy.state_buffer_bytes('reference_rectangle', 7081, 10), 641*7081*96)

    def test_mesh_override_does_not_change_other_meshes_or_sample_rate(self):
        policy = GPURecordingPolicy(save_interval_by_mesh_s={'handkerchief': .5})
        self.assertEqual(policy.frames_per_chunk('handkerchief'), 30)
        self.assertEqual(policy.frames_per_chunk('triangular_flag'), 120)
        self.assertEqual(policy.substeps, 64)

    def test_invalid_interval_is_not_silently_rounded(self):
        for value in (0, -1, float('nan'), float('inf'), .001, .021, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                GPURecordingPolicy(save_interval_s=value)
        with self.assertRaises(ValueError):
            GPURecordingPolicy(save_interval_by_mesh_s={'handkerchief': 0})


if __name__ == '__main__':
    unittest.main()
