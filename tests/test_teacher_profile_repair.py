import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from wind3dgs.evaluation.teacher_profile_repair import artifacts_valid,profile_command

class ProfileRepairTests(unittest.TestCase):
    def test_worker_success_without_report_is_not_profile_success(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'result.json').write_text('{}');(p/'device_phase_times.csv').write_text('header')
            self.assertFalse(artifacts_valid(p,p/'trace.nsys-rep'))
            (p/'trace.nsys-rep').write_bytes(b'x');self.assertTrue(artifacts_valid(p,p/'trace.nsys-rep'))
    def test_command_disables_cuda_injection(self):
        cmd=profile_command('nsys','python','timer.py','case','dest','report',{'volume':256})
        self.assertIn('--trace=nvtx',cmd)
        self.assertFalse(any('cuda-graph-trace' in x for x in cmd))
    def test_timer_wrap_preserves_action_and_records_calls(self):
        import warp as wp
        from wind3dgs.evaluation.teacher_profile_device_clock import Timer
        from warp.optim.linear import LinearOperator
        wp.config.kernel_cache_dir='/tmp/wind_profile_clock_cpu'
        operation=lambda *a,**kw:17
        op=LinearOperator((1,1),wp.float64,wp.get_device('cpu'),operation)
        s=SimpleNamespace(device='cpu',preconditioner=op,operator=LinearOperator((1,1),wp.float64,wp.get_device('cpu'),operation),
            gmres=SimpleNamespace(inner_product=operation,_orth=operation),coloring=SimpleNamespace(assemble=operation),
            current=SimpleNamespace(factor=operation),ops=SimpleNamespace(evaluate=operation),_line_search=operation)
        timer=Timer(s)
        self.assertEqual(s.operator.matvec(1,2,3,alpha=-1.,beta=1.),17)
        self.assertEqual(s.preconditioner.matvec(1,2,3),17)
        rows=timer.rows(0)
        self.assertEqual(rows[0]['calls'],1);self.assertEqual(rows[1]['calls'],1);self.assertEqual(rows[8]['calls'],1)
