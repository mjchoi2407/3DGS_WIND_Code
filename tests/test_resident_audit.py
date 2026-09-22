import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import warp as wp
from wind3dgs.evaluation.teacher_gpu_audit_io import trace_chunks


class AuditReaderTests(unittest.TestCase):
    def test_compressed_and_uncompressed_different_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            original = np.arange(11*4*6,dtype=np.float64).reshape(11,4,2,3)
            paths = []
            for k,(a,b) in enumerate(((0,4),(3,8),(7,11))):
                path = Path(folder)/f'{k}.npz'; paths.append(path)
                save = np.savez_compressed if k%2 else np.savez
                save(path,**{name:original[a:b,i] for i,name in enumerate(('u_hi','u_lo','v_hi','v_lo'))},
                     time_s=np.arange(a,b,dtype=float)/60,state_encoding=np.array('resident_pair_f64_v1'))
            for capacity in (1,3,5,64):
                blocks = list(trace_chunks(paths,2,capacity))
                actual = np.concatenate([a[0 if i==0 else 1:] for i,(a,t) in enumerate(blocks)])
                np.testing.assert_array_equal(actual,original.reshape(11,4,6))
                times = np.concatenate([t[0 if i==0 else 1:] for i,(a,t) in enumerate(blocks)])
                np.testing.assert_array_equal(times,np.arange(11)/60)

    def test_bad_archive_boundary_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for k in range(2):
                path = Path(folder)/f'{k}.npz'; paths.append(path)
                np.savez(path,**{name:np.full((3,2,3),float(k)) for name in ('u_hi','u_lo','v_hi','v_lo')},
                         time_s=np.arange(3,dtype=float),state_encoding=np.array('resident_pair_f64_v1'))
            with self.assertRaisesRegex(ValueError,'경계'): list(trace_chunks(paths,2,2))


@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class ResidentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wind3dgs.teacher.p3_shell import P3Shell
        cls.model = P3Shell(4); cls.n = len(cls.model.rest_positions)*3

    def audit(self,actual,reference=None,*,force=None,balances=None,times=None,reference_times=None,capacity=1):
        from wind3dgs.teacher.resident_audit import ResidentAudit
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
        steps = len(actual)-1; dt = 1/3840
        if reference is None: reference = actual.copy()
        if force is None: force = np.zeros((1,self.n//3,3))
        if balances is None: balances = np.zeros(steps)
        if times is None: times = np.arange(steps+1,dtype=float)*dt
        if reference_times is None: reference_times = times.copy()
        s = ResidentAudit(self.model,steps=steps,substeps=64,dt=dt,forces=force,balances=balances,
                          policy=ShellSolvePolicy(),chunk_steps=capacity)
        try:
            for start in range(0,steps,capacity):
                end = min(steps,start+capacity)+1
                count = s.upload(actual[start:end],reference[start:end],times[start:end],reference_times[start:end])
                with (patch.object(wp.array,'numpy',side_effect=AssertionError('중간 CPU 수치 조회')),
                      patch.object(CuDSSFactor,'execute',side_effect=AssertionError('중간 CPU 라이브러리 풀이'))):
                    s.submit(count)
            return s.result()
        finally: s.close()

    def test_device_input_prefix_and_gpu_failure_propagation(self):
        from wind3dgs.teacher.resident_audit import ResidentAudit
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher import resident_cloth_recording as k
        s = ResidentAudit(self.model, steps=2, substeps=64, dt=1/3840,
                          forces=np.zeros((1,self.n//3,3)), balances=np.zeros(2),
                          policy=ShellSolvePolicy(), chunk_steps=1, compare_reference=False)
        try:
            a = wp.zeros((2,4,self.n),dtype=wp.float64,device='cuda:0')
            with patch.object(wp.array,'numpy',side_effect=AssertionError('GPU 전달 중 CPU 조회')):
                s.submit(s.upload_device(a,origin_s=2.))
            first = s.result(1)
            self.assertFalse(first['flags'].any()); self.assertFalse(first['time_failed'])
            self.assertFalse(first['reference_comparison_enabled'])
            raw = np.zeros((2,4,self.n));raw[1,0,0]=np.nan
            a.assign(raw)
            s.submit(s.upload_device(a,origin_s=2.))
            failure=wp.zeros(1,dtype=wp.int32,device='cuda:0')
            enabled=wp.ones(1,dtype=wp.int32,device='cuda:0')
            bad=wp.array([2],dtype=wp.int32,device='cuda:0')
            wp.launch(k.stop_on_audit,dim=2,inputs=[s.flags,0,2,bad,failure,enabled],device='cuda:0')
            result=s.result()
            self.assertEqual(int(bad.numpy()[0]),1);self.assertEqual(int(failure.numpy()[0]),99)
            np.testing.assert_array_equal(result['history'][:1],first['history'])
            self.assertEqual(result['graph_inventory']['host_copies'],0)
        finally: s.close()

    def test_chunk_invariance_and_no_host_graph_nodes(self):
        a = np.zeros((4,4,self.n))
        one = self.audit(a,capacity=1); all_steps = self.audit(a,capacity=3)
        np.testing.assert_array_equal(one['history'],all_steps['history'])
        self.assertFalse(one['flags'].any()); self.assertFalse(one['time_failed'])
        self.assertEqual(one['graph_inventory']['host_copies'],0)
        self.assertEqual(one['graph_inventory']['host_callbacks'],0)
        self.assertGreater(one['graph_inventory']['children'],0)

    def test_force_position_energy_and_pin_failures(self):
        a = np.zeros((2,4,self.n)); force = np.zeros((1,self.n//3,3))
        free = np.flatnonzero(self.model.free)[0]; fixed = np.flatnonzero(~self.model.free)[0]
        force[0,free,1] = .01
        self.assertTrue(int(self.audit(a,force=force)['flags'][0]) & 2)
        self.assertTrue(int(self.audit(a,balances=np.array([1e-4]))['flags'][0]) & 8)
        a[1,0,3*fixed] = 1e-5
        flag = int(self.audit(a)['flags'][0]); self.assertTrue(flag & 32); self.assertTrue(flag & 4)
        a.fill(0); a[0,0,3*fixed] = 1e-5
        self.assertTrue(int(self.audit(a)['flags'][0]) & 32)

    def test_time_reference_and_nonfinite_failures(self):
        a = np.zeros((2,4,self.n)); times = np.array([0.,1/3840])
        self.assertTrue(self.audit(a,times=times,reference_times=times+1)['time_failed'])
        self.assertTrue(self.audit(a,times=np.array([0.,.1]))['time_failed'])
        ref = a.copy(); ref[1,0,0] = 1e-5
        self.assertEqual(self.audit(a,reference=ref)['comparison'][0,1],1)
        a[1,0,0] = np.nan
        self.assertTrue(int(self.audit(a)['flags'][0]) & 1)

    def test_comparison_keeps_low_parts(self):
        a = np.zeros((2,4,self.n)); ref = a.copy()
        j = 3*np.flatnonzero(self.model.free)[0]
        a[:,0,j] = 1e-4; ref[:,0,j] = 1e-4; a[1,1,j] = 2e-22
        result = self.audit(a,reference=ref)
        self.assertAlmostEqual(result['comparison'][0,0],2e-22,delta=1e-35)

    def test_polynomial_bounds_and_between_endpoint_failure(self):
        from wind3dgs.teacher.resident_audit_bounds import ResidentAuditBounds
        from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds
        m = self.model; gpu = ResidentAuditBounds(m); cpu = P3ShellBounds(m)
        x,y = m.xy.T; x = x-.25
        base = np.column_stack((.04*x*x*y,.2*x**3+.1*x*y*y,-.03*x*y))
        zero = np.zeros_like(base)
        for u0,v0,u1,dt in ((zero,4*base,zero,1.),(base,zero,base,0.),(zero,np.column_stack((-8*x,0*x,0*x)),zero,1.)):
            expected = cpu.interval(u0,v0,u1,dt)
            arrays = [wp.array(a.ravel(),dtype=wp.float64,device='cuda:0') for a in (u0,zero,v0,zero,u1,zero)]
            with patch.object(wp.array,'numpy',side_effect=AssertionError('기하 검사 중 CPU 조회')):
                result = gpu.evaluate(*arrays,dt)
            actual = result.numpy()
            np.testing.assert_allclose(actual[:4],[expected['projected_gradient_upper'],expected['strain_component_upper'],expected['engineering_curvature_component_upper_inv_m'],expected['roundoff_margin']],rtol=2e-12,atol=2e-12)
            self.assertEqual(actual[0]>=1,not expected['injectivity_sufficient_condition'])
        a = np.zeros((2,4,self.n))
        a[0,2,::3] = -8*x*3840
        self.assertTrue(int(self.audit(a)['flags'][0]) & 16)


if __name__ == '__main__': unittest.main()
