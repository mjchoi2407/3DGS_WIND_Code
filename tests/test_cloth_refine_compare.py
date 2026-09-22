"""청크 경계가 다른 저장 결과와 실패 프레임 분모를 검증한다."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from wind3dgs.evaluation.teacher_cloth_refine_compare import frame_times, trajectory_difference


class ComparisonTests(unittest.TestCase):
    def make_run(self, folder, ends, offset=0., old_low=0.):
        folder.mkdir(); (folder/'chunks').mkdir()
        report = {'chunks': []}
        begin = 0
        for i, end in enumerate(ends):
            steps = np.arange(begin*64, end*64+1)
            values = np.broadcast_to(steps[:, None, None]*1e-5, (len(steps), 2, 3)).copy()
            values[steps > 0] += offset
            low = np.full_like(values, old_low); low[steps == 0] = 0
            path = f'chunks/{i:04d}.npz'
            np.savez(folder/path, u_hi=values, u_lo=low, v_hi=2*values, v_lo=2*low,
                     time_s=steps/3840)
            report['chunks'].append({'path': path, 'begin_frame': begin, 'end_frame': end})
            begin = end
        return report

    def test_different_chunks_include_low_and_common_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = self.make_run(root/'old', [1, 3], old_low=1e-12)
            new = self.make_run(root/'new', [2, 3], offset=4e-12)
            result = trajectory_difference(root/'old', old, root/'new', new, 2)
            self.assertEqual(result['substeps'], 128)
            self.assertAlmostEqual(result['position_max_m'], np.sqrt(3)*3e-12, delta=1e-18)
            self.assertAlmostEqual(result['velocity_max_m_s'], 2*result['position_max_m'], delta=1e-18)

    def test_failed_and_unsaved_frames_are_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'timings.jsonl'
            rows = [{'frame': i, 'failed': i == 2, 'compute_audit_s': i+1.} for i in range(5)]
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(frame_times(path, 3), {0: 1., 1: 2.})
            path.write_text(json.dumps(rows[0])+'\n'+json.dumps(rows[0]))
            with self.assertRaisesRegex(ValueError, '중복'):frame_times(path, 1)

    def test_missing_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = self.make_run(root/'old', [1]); new = self.make_run(root/'new', [1])
            old['chunks'][0]['begin_frame'] = 1
            with self.assertRaisesRegex(ValueError, 'prefix'):
                trajectory_difference(root/'old', old, root/'new', new, 1)


class DiagnosticRecordingTests(unittest.TestCase):
    def test_warning_mask_preserves_flags_and_nonfinite_stops(self):
        import warp as wp
        from wind3dgs.teacher.resident_cloth_recording import stop_on_audit_masked
        for flag, mask, expected in ((16, 1, 0), (2, 1, 0), (32, 1, 0), (1, 1, 99), (16, 63, 99)):
            flags = wp.array([flag], dtype=wp.int32, device='cpu')
            failure = wp.zeros(1, dtype=wp.int32, device='cpu')
            bad = wp.array([1], dtype=wp.int32, device='cpu')
            enabled = wp.ones(1, dtype=wp.int32, device='cpu')
            wp.launch(stop_on_audit_masked, 1, [flags, 0, 1, bad, failure, enabled, mask], device='cpu')
            self.assertEqual(failure.numpy()[0], expected)
            self.assertEqual(flags.numpy()[0], flag)

    def test_recording_keeps_solver_warning_and_work_counts(self):
        import warp as wp
        from wind3dgs.teacher.resident_cloth_diagnostics import store
        def ints(a):return wp.array(a, dtype=wp.int32, device='cpu')
        def floats(a):return wp.array(a, dtype=wp.float64, device='cpu')
        counts = wp.zeros((1,29), dtype=wp.int32, device='cpu')
        values = wp.zeros((1,6), dtype=wp.float64, device='cpu')
        wp.launch(store, 29, [ints(list(range(17))), ints(list(range(10))), ints([0,128]),
                             floats([1.,2.,3.]), floats(list(range(10))), counts, values, 0], device='cpu')
        np.testing.assert_array_equal(counts.numpy()[0], list(range(17))+list(range(10))+[128,0])
        np.testing.assert_array_equal(values.numpy()[0], [1.,2.,3.,0.,1.,5.])


if __name__ == '__main__':unittest.main()
