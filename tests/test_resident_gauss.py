"""Gauss 실수 보조 풀이의 동등성과 명시적 GPU 상주/실패/재개 검사."""
import os
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix,kron,eye
from scipy.sparse.linalg import spsolve
from wind3dgs.teacher.p3_shell_gauss import tableau,coupled_preconditioner
from wind3dgs.teacher.resident_gauss import coupled_pattern,real_stage_transform

class RealCouplingTests(unittest.TestCase):
    def test_real_block_matches_original_complex_preconditioner(self):
        rng=np.random.default_rng(14); R=rng.normal(size=(7,7)); K=csr_matrix(R.T@R+np.eye(7))
        M=csr_matrix(np.diag(np.arange(1,8))); inv=tableau(3)[3]; h=.17
        stiffness,block,mapping,constant,T,Ti=coupled_pattern(K,M,inv,h)
        rhs=rng.normal(size=(3,7)); old,_=coupled_preconditioner(K,M,inv,h)
        result=(T@spsolve(block,(Ti@rhs).ravel()).reshape(3,7)).ravel()
        np.testing.assert_allclose(result,old@rhs.ravel(),rtol=2e-13,atol=1e-15)
        original=kron(eye(3),K)+kron(csr_matrix(np.asarray(inv,float)/h**2),M)
        np.testing.assert_allclose(original@result,rhs.ravel(),rtol=2e-13,atol=2e-13)
        # 다음 세대의 값 갱신. 구조만 재사용하고 수치 행렬은 바뀐다.
        updated=constant+np.where(mapping>=0,2*stiffness.data[np.maximum(mapping,0)],0)
        new=block.copy();new.data[:]=updated
        actual=(T@spsolve(new,(Ti@rhs).ravel()).reshape(3,7)).ravel()
        expected=spsolve((kron(eye(3),2*K)+kron(csr_matrix(np.asarray(inv,float)/h**2),M)).tocsr(),rhs.ravel())
        np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-15)

@unittest.skipUnless(os.environ.get('WIND3DGS_TEST_DEVICE')=='cuda:0','명시적 GPU 검증 전용')
class GaussGpuTests(unittest.TestCase):
    def test_gpu_audit_independent_buffers_and_ledger_rejection(self):
        import warp as wp
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
        from wind3dgs.teacher.resident_gauss_audit import ResidentGaussAudit
        m=P3Shell(4);zero=np.zeros_like(m.rest_positions);force=zero.copy();force[m.free,1]=1e-3
        s=ResidentGaussStepper(m,[zero]*4,force,dt=1/3840)
        try:
            s.step();self.assertEqual(int(s.failure.numpy()[0]),0)
            check=ResidentGaussAudit(m,s.policy,s.dt)
            initial=[wp.zeros(s.nfull,dtype=wp.float64,device='cuda:0') for _ in range(4)]
            acc=np.zeros((3,s.nodes,3));acc[:,m.free]=s.acc.numpy().reshape(3,-1,3)
            stage=[s.U,s.L,s.W,s.WL,wp.array(acc.reshape(3,-1),dtype=wp.float64,device='cuda:0')]
            ledger=wp.array(s.energy.numpy()[1:2],dtype=wp.float64,device='cuda:0')
            with patch.object(wp.array,'numpy',side_effect=AssertionError('검산 중 host 조회')):
                check.submit_device(initial,s.state,stage,s.held,ledger)
            self.assertEqual(int(check.failure.numpy()[0]),0)
            ledger.assign(np.array([1.]))
            check.submit_device(initial,s.state,stage,s.held,ledger)
            self.assertTrue(int(check.failure.numpy()[0])&4)
            bad=s.U.numpy();bad[0,3*np.flatnonzero(m.free)[0]]=np.nan
            stage[0]=wp.array(bad,dtype=wp.float64,device='cuda:0')
            check.submit_device(initial,s.state,stage,s.held,ledger)
            self.assertTrue(int(check.failure.numpy()[0])&8)
            check.close()
        finally:s.close()

    def test_linear_failure_preserves_committed_state(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
        from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
        m=P3Shell(4);zero=np.zeros_like(m.rest_positions);force=zero.copy();force[m.free,1]=.1
        s=ResidentGaussStepper(m,[zero]*4,force,dt=1/120,policy=ShellSolvePolicy(linear_restart=1,linear_cycles=1))
        try:
            s.step();self.assertEqual(int(s.failure.numpy()[0]),2)
            self.assertEqual(int(s.c.numpy()[6]),0)
            for a in s.state:np.testing.assert_array_equal(a.numpy(),0.)
        finally:s.close()

    def test_cpu_equivalence_host_free_and_restart(self):
        import warp as wp
        from wind3dgs.teacher.p3_shell import P3Shell
        from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
        from wind3dgs.teacher.p3_shell_gauss import gauss_step
        from wind3dgs.teacher.p3_shell_cudss import CuDSSFactor
        from wind3dgs.teacher.resident_gauss import ResidentGaussStepper
        from wind3dgs.teacher.resident_capture_audit import track_conditional_bodies
        from wind3dgs.teacher.resident_audit import device_graph_inventory
        m=P3Shell(4); zero=np.zeros_like(m.rest_positions); force=zero.copy();force[m.free,1]=1e-3
        raw=[zero.copy() for _ in range(4)]
        with track_conditional_bodies() as bodies:
            s=ResidentGaussStepper(m,raw,force,dt=1/3840,rebuild_every=2)
        try:
            inventory=device_graph_inventory(s.step_graph,conditional_bodies=bodies)
            self.assertTrue(inventory)
            with patch.object(wp.array,'numpy',side_effect=AssertionError('계산 중 host read')),patch.object(CuDSSFactor,'execute',side_effect=AssertionError('계산 중 CPU 수치 library')):
                s.step();s.step()
            self.assertEqual(int(s.failure.numpy()[0]),0)
            actual=[x.numpy().reshape(-1,3) for x in s.state]
            cpu=P3ShellWarpPrecisionStepper(P3ShellWarpPrecision(m,device='cuda:0',capture=True),policy=s.policy)
            state=cpu.state()
            for _ in range(2):state,_=gauss_step(cpu,state,force,1/3840,stages=3,preconditioner_kind='coupled')
            for i,expected in [(0,state.displacement_m),(2,state.velocity_m_s)]:
                np.testing.assert_allclose(actual[i].astype(np.longdouble)+actual[i+1],expected,rtol=1e-8,atol=2e-12 if i else 2e-14)
            s.step();expected=[x.numpy().copy() for x in s.state]
            s.set_state(actual);s.step()
            for a,b in zip(s.state,expected):np.testing.assert_allclose(a.numpy(),b,rtol=1e-8,atol=2e-12)
            from wind3dgs.teacher.gauss_independent_audit import GaussIndependentAudit
            s.set_state(raw,force);s.step()
            end=[x.numpy().reshape(-1,3).astype(np.longdouble) for x in s.state]
            U=(s.U.numpy().astype(np.longdouble)+s.L.numpy()).reshape(3,-1,3)
            W=(s.W.numpy().astype(np.longdouble)+s.WL.numpy()).reshape(3,-1,3)
            acc=np.zeros_like(U);acc[:,m.free]=s.acc.numpy().reshape(3,-1,3)
            checker=GaussIndependentAudit(m,s.policy)
            audit=checker.verify((zero,zero),(end[0]+end[1],end[2]+end[3]),U,W,acc,force,s.dt,float(s.energy.numpy()[1]))
            self.assertTrue(audit['passed'],audit)
            changed=acc.copy();changed[0,m.free,1]+=1.
            audit_bad=checker.verify((zero,zero),(end[0]+end[1],end[2]+end[3]),U,W,changed,force,s.dt,float(s.energy.numpy()[1]))
            self.assertFalse(audit_bad['passed'])
            # 실패 flag가 있는 호출은 확정 상태를 바꾸지 않는다.
            s.failure.assign(np.array([2],np.int32));s.step()
            for a,b in zip(s.state,end):np.testing.assert_array_equal(a.numpy().reshape(-1,3),b)
            s.set_state(raw,np.zeros_like(force));s.step()
            self.assertEqual(int(s.failure.numpy()[0]),0)
            for a in s.state:np.testing.assert_array_equal(a.numpy(),0.)
        finally:s.close()

if __name__=='__main__':unittest.main()
