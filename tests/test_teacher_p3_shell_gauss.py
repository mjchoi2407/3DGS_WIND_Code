import unittest
from types import SimpleNamespace
import numpy as np
from scipy.sparse import eye
from wind3dgs.teacher.p3_shell_gauss import gauss_step,A,A2,B,C,tableau
from wind3dgs.teacher.p3_shell_dynamics import ShellSolvePolicy
from wind3dgs.teacher.p3_shell_precision_state import extended_state

class Oscillator:
    def __init__(self,omega=2):
        self.omega2=omega**2
        self.M=eye(3,format='csc');self.K=self.omega2*self.M
        self.policy=ShellSolvePolicy(linear_cycles=3,linear_restart=240)
        self.model=SimpleNamespace(free=np.array([True]),mass=eye(1,format='csr'),evaluate_displacement=self.elastic)
    def elastic(self,u,*,direction=None):
        if direction is not None:return {'hvp_n':self.omega2*direction}
        return {'force_n':-self.omega2*u,'energy_j':float(self.omega2/2*np.sum(u*u))}
    def _make_state(self,u,v,t):return extended_state(u,v,t)
    def _validate_state(self,state):pass
    def kinetic_energy(self,v):return float(np.sum(v*v)/2)

class GaussTests(unittest.TestCase):
    def test_coefficients(self):
        np.testing.assert_allclose(B@A2,np.eye(2),rtol=0,atol=2e-18)
        np.testing.assert_allclose(C,[.5-np.sqrt(3)/6,.5+np.sqrt(3)/6],atol=1e-16)
    def test_order_four_and_linear_energy(self):
        errors=[]
        for steps in [5,10]:
            s=Oscillator();state=s._make_state([[1,0,0]],[[0,0,0]],0)
            for _ in range(steps):state,d=gauss_step(s,state,np.zeros((1,3)),1/steps)
            exact=np.array([np.cos(2),-2*np.sin(2)])
            errors.append(float(np.linalg.norm(np.array([state.displacement_m[0,0],state.velocity_m_s[0,0]])-exact)))
            self.assertAlmostEqual(s.elastic(state.displacement_m)['energy_j']+s.kinetic_energy(state.velocity_m_s),2.,delta=1e-11)
        self.assertGreater(errors[0]/errors[1],14)
        self.assertLess(errors[0]/errors[1],18)
    def test_order_six_and_collocation_controls(self):
        errors=[]
        for steps in [2,4]:
            s=Oscillator();state=s._make_state([[1,0,0]],[[0,0,0]],0)
            for _ in range(steps):
                state,d=gauss_step(s,state,np.zeros((1,3)),1/steps,stages=3)
                for node,u in zip(tableau(3)[2],d['stage_u_m']):
                    controls=d['position_controls_m'].copy()
                    while len(controls)>1:controls=(1-node)*controls[:-1]+node*controls[1:]
                    np.testing.assert_allclose(controls[0],u,rtol=0,atol=2e-15)
            errors.append(float(np.linalg.norm(np.array([state.displacement_m[0,0],state.velocity_m_s[0,0]])-[np.cos(2),-2*np.sin(2)])))
        self.assertGreater(errors[0]/errors[1],55)
        self.assertLess(errors[0]/errors[1],75)

    def test_order_eight(self):
        errors=[]
        for steps in [1,2]:
            s=Oscillator();state=s._make_state([[1,0,0]],[[0,0,0]],0)
            for _ in range(steps):state,d=gauss_step(s,state,np.zeros((1,3)),1/steps,stages=4)
            errors.append(float(np.linalg.norm(np.array([state.displacement_m[0,0],state.velocity_m_s[0,0]])-[np.cos(2),-2*np.sin(2)])))
        self.assertGreater(errors[0]/errors[1],200)
        self.assertLess(errors[0]/errors[1],300)
        for n in [4,5,6]:
            a,w,c,b,p=tableau(n)
            np.testing.assert_allclose(b@a@a,np.eye(n),atol=2e-15,rtol=0)
            np.testing.assert_allclose(a.sum(axis=1),c,atol=2e-17,rtol=0)
            self.assertAlmostEqual(float(w.sum()),1.)

    def test_failure_preserves_input(self):
        from wind3dgs.teacher.p3_shell_dynamics import ShellStepFailed
        s=Oscillator();s.policy=ShellSolvePolicy(max_newton=1,linear_cycles=3,linear_restart=240)
        state=s._make_state([[1,0,0]],[[0,0,0]],0)
        with self.assertRaises(ShellStepFailed):gauss_step(s,state,np.zeros((1,3)),.5,stages=4)
        np.testing.assert_array_equal(state.displacement_m,[[1,0,0]])
        np.testing.assert_array_equal(state.velocity_m_s,[[0,0,0]])

    def test_p3_checkpoint_replay(self):
        import tempfile
        from pathlib import Path
        from wind3dgs.teacher.p3_shell_execution import make_shell_stepper
        from wind3dgs.teacher.p3_shell_precision_state import save_checkpoint,load_checkpoint
        for stages in [4,6]:
            first,_=make_shell_stepper(4,device='cpu',backend='precision_hvp_graph')
            second,_=make_shell_stepper(4,device='cpu',backend='precision_hvp_graph')
            for s in [first,second]:s.policy=ShellSolvePolicy(linear_cycles=3,linear_restart=240)
            state=first.state();force=np.zeros_like(state.displacement_m);force[first.model.free,2]=1e-7
            middle,_=gauss_step(first,state,force,1/60,stages=stages,preconditioner_kind='coupled')
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.npz';save_checkpoint(path,middle);restored=load_checkpoint(path,second)
                np.testing.assert_array_equal(restored.displacement_m,middle.displacement_m)
                np.testing.assert_array_equal(restored.velocity_m_s,middle.velocity_m_s)
                a,_=gauss_step(first,middle,force,1/60,stages=stages,preconditioner_kind='coupled')
                b,_=gauss_step(second,restored,force,1/60,stages=stages,preconditioner_kind='coupled')
                np.testing.assert_allclose(a.displacement_m,b.displacement_m,rtol=0,atol=1e-16)
                np.testing.assert_allclose(a.velocity_m_s,b.velocity_m_s,rtol=0,atol=1e-12)

    def test_coupled_preconditioner_matches_block_system(self):
        from scipy.sparse import csr_matrix,kron
        from wind3dgs.teacher.p3_shell_gauss import coupled_preconditioner
        k=csr_matrix([[4.,1.,0.],[1.,5.,1.],[0.,1.,6.]])
        mass=csr_matrix(np.diag([1.,2.,3.]))
        for stages in [2,3,4,6]:
            inv=tableau(stages)[3]
            preconditioner,_=coupled_preconditioner(k,mass,inv,.1)
            matrix=kron(np.asarray(inv,dtype=float)/.01,mass)+kron(eye(stages),k)
            rhs=np.arange(3*stages,dtype=np.longdouble)
            np.testing.assert_allclose(matrix@(preconditioner@rhs),rhs,rtol=1e-11,atol=1e-11)
        s=Oscillator();state=s._make_state([[1,0,0]],[[0,0,0]],0)
        a,_=gauss_step(s,state,np.zeros((1,3)),.1,stages=4)
        b,_=gauss_step(s,state,np.zeros((1,3)),.1,stages=4,preconditioner_kind='coupled')
        np.testing.assert_allclose(a.displacement_m,b.displacement_m,rtol=0,atol=1e-14)
        np.testing.assert_allclose(a.velocity_m_s,b.velocity_m_s,rtol=0,atol=1e-12)

    def test_order_twelve(self):
        errors=[]
        for steps in [2,4]:
            s=Oscillator(omega=6);state=s._make_state([[1,0,0]],[[0,0,0]],0)
            for _ in range(steps):state,d=gauss_step(s,state,np.zeros((1,3)),1/steps,stages=6,preconditioner_kind='coupled')
            errors.append(float(np.linalg.norm(np.array([state.displacement_m[0,0],state.velocity_m_s[0,0]])-[np.cos(6),-6*np.sin(6)])))
        self.assertGreater(errors[0]/errors[1],2500)
        self.assertLess(errors[0]/errors[1],5000)
        a,w,c,b,p=tableau(6)
        for k in range(12):self.assertAlmostEqual(float(np.sum(w*c**k)),1/(k+1),delta=1e-15)

    def test_constant_force_exact_update_and_immutable_input(self):
        s=Oscillator();s.K=0*s.K
        s.model.evaluate_displacement=lambda u,direction=None: {'hvp_n':np.zeros_like(u)} if direction is not None else {'force_n':np.zeros_like(u),'energy_j':0.}
        state=s._make_state([[0,0,0]],[[1,0,0]],0)
        end,d=gauss_step(s,state,np.array([[2,0,0]]),.2)
        np.testing.assert_allclose(end.displacement_m,[[.24,0,0]],atol=1e-15)
        np.testing.assert_allclose(end.velocity_m_s,[[1.4,0,0]],atol=1e-15)
        np.testing.assert_array_equal(state.displacement_m,[[0,0,0]])
        self.assertLess(d['stage_update_error_m'],1e-15)
