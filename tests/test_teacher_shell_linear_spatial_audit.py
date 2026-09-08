"""선형 normal 해, 공간 비교와 실패 보존 계약의 독립 검사."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg import expm

from wind3dgs.evaluation import teacher_shell_linear_spatial_audit as a


class LinearSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.arrays, cls.diagnostics = a._model(4, 'forward', a.t._Budget(60))
        cls.probes, cls.xyz = a._probes()
        cls.mapping, cls.mapping_report = a._mapping(cls.model, cls.probes, cls.xyz, 4)

    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.folder = Path(temp.name)

    def test_policy_rejects_extended_nonfinite_or_boolean_budget(self):
        for value in (0, -1, 1801, True, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                a.TeacherShellLinearSpatialPolicy(max_wall_time_s=value)
        self.assertEqual(a.TeacherShellLinearSpatialPolicy().max_wall_time_s, 1800)

    def test_modal_response_matches_first_order_matrix_exponential(self):
        arr = self.arrays; free = ~arr['pins']; mass = arr['mass_kg'][free]
        K = arr['stiffness_n_m'][np.ix_(free, free)]; N = len(mass)
        H = np.block([[np.zeros((N, N)), np.eye(N)], [-K/mass[:, None], np.zeros((N, N))]])
        initial = np.r_[arr['u0_m'][free], np.zeros(N)]
        times = [0., .004, .1]
        response = a._response(arr, times)
        for i, time in enumerate(times):
            ref = expm(H*time) @ initial
            np.testing.assert_allclose(response['u_m'][i, free], ref[:N], atol=1e-14, rtol=1e-9)
            np.testing.assert_allclose(response['v_m_s'][i, free], ref[N:], atol=1e-12, rtol=1e-8)

    def test_arbitrary_normal_initial_roundtrip_and_analytic_derivatives(self):
        arr = copy.deepcopy(self.arrays); free = ~arr['pins']
        rng = np.random.default_rng(18); arr['u0_m'][free] = rng.normal(size=free.sum())*1e-4
        arr['q0'] = arr['basis'].T @ (np.sqrt(arr['mass_kg'][free])*arr['u0_m'][free])
        x = a._response(arr, [0., .02-1e-7, .02, .02+1e-7])
        np.testing.assert_allclose(x['u_m'][0], arr['u0_m'], atol=2e-18)
        np.testing.assert_allclose((x['u_m'][3]-x['u_m'][1])/2e-7, x['v_m_s'][2], atol=1e-9, rtol=1e-6)
        np.testing.assert_allclose((x['v_m_s'][3]-x['v_m_s'][1])/2e-7, x['a_m_s2'][2], atol=1e-7, rtol=1e-6)

    def test_rigid_rotation_stationary_without_null_projection_removal(self):
        arr = copy.deepcopy(self.arrays); free = ~arr['pins']; arr['q0'][:] = 0.
        null = arr['omega_rad_s'] == 0; arr['q0'][null] = .001
        x = a._response(arr, [0., .4, 2.])
        np.testing.assert_array_equal(x['u_m'][0], x['u_m'][2])
        self.assertGreater(np.linalg.norm(x['u_m'][0]), 0)
        np.testing.assert_array_equal(x['v_m_s'], 0); np.testing.assert_array_equal(x['a_m_s2'], 0)
        self.assertEqual(self.diagnostics['normal_nullity'], 1)
        self.assertLess(self.diagnostics['rigid_null_error'], 1e-8)

    def test_sign_and_degenerate_basis_rotation_invariant(self):
        base = copy.deepcopy(self.arrays); free = ~base['pins']; base['omega_rad_s'][2:4] = 10.
        changed = copy.deepcopy(base)
        angle = .73; R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        changed['basis'][:, 2:4] = changed['basis'][:, 2:4] @ R
        changed['basis'][:, ::2] *= -1
        changed['q0'] = changed['basis'].T @ (np.sqrt(changed['mass_kg'][free])*changed['u0_m'][free])
        x, y = a._response(base, [0., .1]), a._response(changed, [0., .1])
        for key in ('u_m', 'v_m_s', 'a_m_s2'):
            np.testing.assert_allclose(x[key], y[key], atol=1e-13, rtol=1e-9)

    def test_energy_reaction_and_pins(self):
        arr = self.arrays; x = a._response(arr, np.linspace(0, a.PERIOD, 257))
        drift, eom = a._frame_check(x, x['energy_j'][0])
        self.assertLess(drift, 1e-8); self.assertLess(eom, 1e-8)
        for key in ('u_m', 'v_m_s', 'a_m_s2'): self.assertEqual(np.count_nonzero(x[key][:, arr['pins']]), 0)
        self.assertEqual(np.count_nonzero(x['reaction_n'][:, ~arr['pins']]), 0)
        self.assertGreater(np.max(abs(x['reaction_n'])), 0.)
        bad = copy.deepcopy(x); bad['energy_j'][1] *= 2
        with self.assertRaises(ValueError): a._frame_check(bad, x['energy_j'][0])

    def test_probes_quadrature_moments_and_positive_measure(self):
        self.assertEqual(len(self.xyz), 3072)
        p = self.probes.arrays(); areas = p['area_weights_m2']; x, y = self.xyz[:, :2].T
        for i, j in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            self.assertAlmostEqual(float(np.sum(areas*x**i*y**j)), 1/((i+1)*(j+1)), places=14)
        self.assertTrue(np.all(areas > 0)); self.assertAlmostEqual(p['mass_weights_kg'].sum(), .1)
        # 각 overlay 삼각형의 3점 moment: 정사각형 전체의 상쇄에 의존하지 않는다.
        for j in (0, 7, 15):
            for i in (0, 9, 15):
                corners = np.array([[i,j],[i+1,j],[i+1,j+1],[i,j+1]],dtype=float)/16
                for k in range(4):
                    v = np.array([corners.mean(0), corners[k], corners[(k+1)%4]])
                    points = self.xyz[((j*16+i)*4+k)*3:((j*16+i)*4+k)*3+3,:2]
                    expected = (np.outer(v.sum(0), v.sum(0))+v.T@v)/12
                    np.testing.assert_allclose(points.T@points/3, expected, atol=2e-16)

    def test_adapter_affine_roundtrip_and_p1_bound_all_n_diagonals(self):
        np.testing.assert_array_equal(a.ROTATION.T@a.ROTATION, np.eye(3)); self.assertEqual(np.linalg.det(a.ROTATION), 1.)
        for n in (4, 8, 16):
            for diagonal in a.DIAGONALS:
                model = a.t.previous._fixture(a.t.previous.TeacherShellDynamicsSpec(1e6,.3,.01,.1),n,diagonal,
                    mode='rest_linear_reference')
                mapping, report = a._mapping(model, self.probes, self.xyz, n)
                self.assertEqual(np.count_nonzero(~mapping['supported']), 0)
                self.assertLessEqual(report['initial_p1_error_m'], a.AMPLITUDE/(4*n*n)+1e-15)
                rest = model.structure.rest_positions_m
                mapped = a._map_scalar(rest[None,:,0],mapping)[0]
                np.testing.assert_allclose(mapped,self.xyz[:,0],atol=2e-16)

    def test_known_field_metric_and_constant_scalar_mapping(self):
        times=np.array([0.,1.,2.]); curve=np.array([0.,2.,0.])
        m=a._metric(curve,times,4.)
        self.assertEqual(m['normalized'],.5); self.assertEqual(m['max_time_s'],1.)
        self.assertAlmostEqual(m['time_rms_si'],np.sqrt(2))
        np.testing.assert_allclose(a._map_scalar(np.ones((3,25))*-2,self.mapping),-2,atol=1e-15)

    def test_gate_threshold_floor_order_and_incomplete_ladder(self):
        self.assertEqual(a._ladder(.02,.01)['status'],'passed')
        self.assertEqual(a._ladder(.01,.01)['status'],'failed')
        self.assertEqual(a._ladder(.02,.010001)['status'],'failed')
        self.assertEqual(a._ladder(0.,1e-10)['reason'],'below_algebra_floor')
        self.assertIsNone(a._ladder(0.,0.)['observed_order'])
        self.assertEqual(a._ladder(.04,.01)['observed_order'],2.)
        self.assertEqual(a._gates([])['spatial_response_check'],'not_assessed')

    def test_array_units_hash_nonfinite_rejected(self):
        x=a._response(self.arrays,[0.,.1]); ids=a._identities(x,a.UNITS)
        np.savez(self.folder/'x.npz',**x)
        a._load(self.folder/'x.npz',ids,a.UNITS)
        for key,value in (('unit','cm'),('sha256','0'*64),('shape',[2,24])):
            bad=copy.deepcopy(ids); bad['u_m'][key]=value
            with self.assertRaises(ValueError): a._load(self.folder/'x.npz',bad,a.UNITS)
        x['u_m'][0,0]=np.nan
        with self.assertRaises(ValueError): a._identities(x,a.UNITS)

    def test_duplicate_npz_and_strict_json_rejected(self):
        import io
        import warnings
        import zipfile
        payload=io.BytesIO(); np.save(payload,np.array([0.]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with zipfile.ZipFile(self.folder/'duplicate.npz','w') as z:
                z.writestr('time_s.npy',payload.getvalue()); z.writestr('time_s.npy',payload.getvalue())
        with self.assertRaises(ValueError): a._load(self.folder/'duplicate.npz',{},a.UNITS)
        for text in ('{"status":1,"status":2}', '{"value":NaN}'):
            (self.folder/'report.json').write_text(text)
            with self.assertRaises(ValueError): a._source(self.folder,a.t._Budget(10))

    def test_mid_case_interrupt_preserves_one_committed_chunk(self):
        response=a._response; calls=0
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3: raise KeyboardInterrupt()
            return response(*args,**kwargs)
        output=self.folder/'interrupted'
        with patch.object(a,'_source',return_value={}), patch.object(a,'_response',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                a.write_teacher_shell_linear_spatial_audit(self.folder/'source',output)
        m=a.t._json(output/'manifest.json'); report=a.t._json(output/'failure_report.json')
        self.assertEqual(m['status'],'interrupted'); self.assertEqual(m['last_checkpoint']['last_frame'],128)
        self.assertEqual(m['committed_chunks'],{'n4_forward':[0]})
        self.assertEqual(report['cases'][0]['sampled_frames'],129)
        self.assertEqual(m['pending_cases'],[c[0] for c in a.CASES])

    def test_chunk_checkpoint_and_missing_duplicate_order_hash_rejected(self):
        manifest={'outputs':{},'last_checkpoint':None}
        writer=a._Writer(self.folder/'run',manifest); case='n4_forward'
        initial=a._response(self.arrays,[0.]); writer.npz(f'cases/{case}/initial.npz',initial)
        x=a._response(self.arrays,a.PERIOD*np.arange(1,129)/a.SAMPLES)
        record=a._record(case,0,1,x,None); writer.chunk(record,x)
        self.assertEqual(a.t._json(writer.output/'manifest.json')['last_checkpoint']['last_frame'],128)
        row={'case_id':case,'vertex_count':25,'initial_arrays':a._identities(initial,a.UNITS),
             'chunks':[record],'sampled_frames':129,'last_chunk_sha256':record['chunk_sha256']}
        reader=a._read_frames(writer.output,row,a.t._Budget(10)); next(reader); next(reader)
        with self.assertRaises(ValueError): next(reader)  # 부분 prefix를 완료 case로 읽지 않는다.
        for changed in ({'index':1},{'first_frame':129},{'previous_chunk_sha256':'0'*64}):
            bad=copy.deepcopy(row); bad['chunks'][0].update(changed)
            with self.assertRaises(ValueError): list(a._read_frames(writer.output,bad,a.t._Budget(10)))
        bad=copy.deepcopy(row); bad['chunks']*=2
        with self.assertRaises(ValueError): list(a._read_frames(writer.output,bad,a.t._Budget(10)))

    def test_chunk_io_failure_keeps_previous_checkpoint(self):
        manifest={'outputs':{},'last_checkpoint':None}; writer=a._Writer(self.folder/'run',manifest)
        initial=(writer.output/'manifest.json').read_bytes()
        x=a._response(self.arrays,a.PERIOD*np.arange(1,129)/a.SAMPLES)
        with patch.object(writer,'checkpoint',side_effect=OSError('test disk failure')):
            with self.assertRaises(OSError): writer.chunk(a._record('n4_forward',0,1,x,None),x)
        self.assertIsNone(manifest['last_checkpoint'])
        self.assertEqual((writer.output/'manifest.json').read_bytes(),initial)
        self.assertTrue((writer.output/'cases/n4_forward/chunks/0000.npz').is_file())

    def test_source_timeout_interrupt_failure_preserved_and_no_overwrite(self):
        for index,error in enumerate((a.t._Limit('wall_time_limit'),KeyboardInterrupt(),ValueError('source tamper'))):
            output=self.folder/f'run{index}'
            with patch.object(a,'_source',side_effect=error):
                with self.assertRaises(type(error)): a.write_teacher_shell_linear_spatial_audit(self.folder/'source',output)
            m=a.t._json(output/'manifest.json')
            self.assertEqual(m['status'],'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed')
            self.assertEqual(len(m['pending_cases']),9); self.assertIsNone(m['last_checkpoint'])
            self.assertTrue((output/'failure_report.json').is_file())
            with self.assertRaises(FileExistsError): a.write_teacher_shell_linear_spatial_audit(self.folder/'source',output)


if __name__ == '__main__': unittest.main()
