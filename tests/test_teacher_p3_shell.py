import unittest

import numpy as np

from wind3dgs.teacher import p3_shell_kernels as k
from wind3dgs.teacher.p3_surface import P3SurfaceElement
from wind3dgs.teacher.shell_structure import ShellElasticMaterial


class P3ShellKernelTests(unittest.TestCase):
    def setUp(self):
        self.dm, self.db = ShellElasticMaterial(1e6, .3, .01).matrices()
        self.rng = np.random.default_rng(2093)
        self.F = np.tile([[1., .2, .03], [.01, -.07, -1.]], (7, 1, 1))
        self.H = self.rng.normal(size=(7, 3, 3))*.2

    def test_volume_energy_gradient_and_exact_tangent(self):
        dF = self.rng.normal(size=self.F.shape)*.03; dH = self.rng.normal(size=self.H.shape)*.03
        result = k.volume(self.F, self.H, self.dm, self.db, direction=(dF, dH))
        eps = 2e-5
        p = k.volume(self.F+eps*dF, self.H+eps*dH, self.dm, self.db)
        m = k.volume(self.F-eps*dF, self.H-eps*dH, self.dm, self.db)
        power = np.sum(result['gradient_F']*dF, axis=(1, 2))+np.sum(result['gradient_H']*dH, axis=(1, 2))
        np.testing.assert_allclose(result['directional_energy'], power, rtol=2e-13, atol=2e-13)
        np.testing.assert_allclose(power, (p['energy']-m['energy'])/(2*eps), rtol=3e-8, atol=1e-7)
        for name in ('F', 'H'):
            np.testing.assert_allclose(result['tangent_'+name],
                (p['gradient_'+name]-m['gradient_'+name])/(2*eps), rtol=2e-7, atol=1e-7)

    def test_volume_matches_existing_element_virtual_force(self):
        element = P3SurfaceElement.from_triangle([[0, 0], [1, 0], [0, 1]])
        x, y = element.rest_xy_m.T
        nodes = np.column_stack((x, .4*x*x+.1*x*y, -y))
        state = element.kinematics(nodes)
        H = state['second_inv_m']
        H = np.stack((H[:, 0, 0], H[:, 1, 1], 2*H[:, 0, 1]), axis=1)
        result = k.volume(state['tangent'], H, self.dm, self.db)
        C = element.hessian_inv_m2
        C = np.stack((C[:, :, 0, 0], C[:, :, 1, 1], 2*C[:, :, 0, 1]), axis=2)
        grad = (np.einsum('q,qia,qac->ic', element.weights_m2, element.gradient_inv_m, result['gradient_F'])
                +np.einsum('q,qia,qac->ic', element.weights_m2, C, result['gradient_H']))
        expected = element.stress_force(nodes, result['strain']@self.dm, result['curvature']@self.db)
        np.testing.assert_allclose(-grad, expected, rtol=3e-13, atol=5e-13)

    def test_edge_energy_gradient_tangent_and_side_orientation(self):
        F = [self.F, self.F+self.rng.normal(size=self.F.shape)*.1]
        H = [self.H, self.H+self.rng.normal(size=self.H.shape)*.1]
        direction = [(self.rng.normal(size=f.shape)*.03, self.rng.normal(size=h.shape)*.03) for f,h in zip(F,H)]
        mu = np.array([.6, .8]); penalty = 20.
        for boundary in (False, True):
            count = 1 if boundary else 2
            kw = {'fixed_normal': np.array([0., 1., 0.])} if boundary else {}
            a = k.edge(F[:count], H[:count], mu, penalty, self.db, directions=direction[:count], **kw)
            eps = 1e-5
            p = k.edge([F[i]+eps*direction[i][0] for i in range(count)],
                       [H[i]+eps*direction[i][1] for i in range(count)], mu, penalty, self.db, **kw)
            m = k.edge([F[i]-eps*direction[i][0] for i in range(count)],
                       [H[i]-eps*direction[i][1] for i in range(count)], mu, penalty, self.db, **kw)
            power = sum(np.sum(gf*df, axis=(1, 2))+np.sum(gh*dh, axis=(1, 2))
                        for (gf,gh),(df,dh) in zip(a['gradients'],direction))
            np.testing.assert_allclose(a['directional_energy'], power, rtol=5e-13, atol=1e-14)
            np.testing.assert_allclose(power, (p['energy']-m['energy'])/(2*eps), rtol=1e-8, atol=1e-10)
            for side in range(count):
                for component in (0,1):
                    np.testing.assert_allclose(a['tangents'][side][component],
                        (p['gradients'][side][component]-m['gradients'][side][component])/(2*eps),
                        rtol=2e-7, atol=1e-10)
        reverse = k.edge(F[::-1], H[::-1], -mu, penalty, self.db)
        forward = k.edge(F, H, mu, penalty, self.db)
        np.testing.assert_allclose(reverse['energy'], forward['energy'], rtol=1e-14)
        for a,b in zip(reverse['gradients'][::-1], forward['gradients']):
            for x,y in zip(a,b): np.testing.assert_allclose(x,y,atol=1e-14)

    def test_rigid_rotation_rest_and_deformed_energy(self):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec([.7, -1.1, .9]).as_matrix()
        a = k.volume(self.F, self.H, self.dm, self.db)
        b = k.volume(self.F@R.T, self.H@R.T, self.dm, self.db)
        np.testing.assert_allclose(a['energy'], b['energy'], rtol=2e-13)
        for key in ('gradient_F','gradient_H'):
            np.testing.assert_allclose(a[key]@R.T, b[key], rtol=1e-12, atol=2e-12)
        F = np.tile([[1.,0,0],[0,0,-1.]], (7,1,1))@R.T
        rest = k.volume(F, np.zeros_like(self.H), self.dm, self.db)
        self.assertLess(np.max(rest['energy']), 1e-25)
        for boundary in (False,True):
            Fs = [self.F] if boundary else [self.F,self.F*1.05]
            Hs = [self.H]*len(Fs)
            kw = {'fixed_normal': np.array([0.,1.,0.])} if boundary else {}
            a = k.edge(Fs,Hs,[1.,0.],20.,self.db,**kw)
            if boundary: kw['fixed_normal'] = R@kw['fixed_normal']
            b = k.edge([F@R.T for F in Fs],[H@R.T for H in Hs],[1.,0.],20.,self.db,**kw)
            np.testing.assert_allclose(a['energy'],b['energy'],atol=1e-13)


class P3ShellAssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from wind3dgs.teacher.p3_shell import P3Shell
        cls.model=P3Shell(4)

    def test_full_mass_and_linear_plate_limit(self):
        from wind3dgs.teacher.p3_wind_reset import make_p3
        m=self.model; old=make_p3(4)
        K=m.rest_stiffness(); normal=K[1::3,1::3].toarray()
        np.testing.assert_allclose(m.mass.toarray(),old.mass,rtol=1e-11,atol=1e-17)
        # Strong position trace에서 허용한 free 블록의 선형 극한이다.
        # Fixed row는 full normal 회전 경계의 접선 방향 미분도 포함한다.
        np.testing.assert_allclose(normal[np.ix_(m.free,m.free)],
                                   old.stiffness[np.ix_(m.free,m.free)],rtol=1e-11,atol=1e-10)
        rng=np.random.default_rng(77); d=rng.normal(size=m.rest_positions.shape)
        r=m.evaluate(m.rest_positions,direction=d)
        np.testing.assert_allclose(r['hvp_n'].ravel(),K@d.ravel(),rtol=1e-10,atol=2e-9)
        free=np.repeat(m.free,3)
        self.assertGreater(np.linalg.eigvalsh(K[free][:,free].toarray()).min(),0)

    def test_nonlinear_force_hvp_and_symmetry(self):
        m=self.model; rng=np.random.default_rng(78)
        x=m.rest_positions.copy(); s=m.xy[:,0]-.25
        x[:,0]-=.08*s*s; x[:,1]+=.25*s*s
        d,e=rng.normal(size=(2,)+x.shape)*.01; d[~m.free]=e[~m.free]=0
        r=m.evaluate(x,direction=d); other=m.evaluate(x,direction=e)
        eps=1e-5; plus,minus=m.evaluate(x+eps*d),m.evaluate(x-eps*d)
        self.assertAlmostEqual(-np.sum(r['force_n']*d),
                              (plus['energy_j']-minus['energy_j'])/(2*eps),delta=2e-7)
        np.testing.assert_allclose(r['hvp_n'],
            -(plus['force_n']-minus['force_n'])/(2*eps),rtol=2e-6,atol=2e-6)
        self.assertAlmostEqual(np.sum(e*r['hvp_n']),np.sum(d*other['hvp_n']),delta=1e-9)

    def test_global_rotation_and_translation(self):
        from scipy.spatial.transform import Rotation
        from wind3dgs.teacher.p3_shell import P3Shell
        m=self.model; R=Rotation.from_rotvec([1.3,-.2,.4]).as_matrix()
        rotated=P3Shell(4,rotation=R)
        x=m.rest_positions.copy(); x[:,1]=.7*(m.xy[:,0]-.25)**2
        a=m.evaluate(x); b=rotated.evaluate(x@R.T)
        self.assertAlmostEqual(a['energy_j'],b['energy_j'],delta=1e-10)
        np.testing.assert_allclose(a['force_n']@R.T,b['force_n'],rtol=1e-9,atol=3e-10)
        shifted=m.evaluate(x+[.2,.3,-.4])
        self.assertAlmostEqual(shifted['energy_j'],a['energy_j'],delta=1e-10)
        np.testing.assert_allclose(a['force_n'].sum(axis=0),0,atol=1e-10)

    def test_free_shell_force_torque_and_rigid_nullspace(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        m=P3Shell(4,clamp=False); x=m.rest_positions.copy(); x[:,1]=.3*x[:,0]**2
        a=m.evaluate(x)
        np.testing.assert_allclose(a['force_n'].sum(axis=0),0,atol=1e-10)
        np.testing.assert_allclose(np.cross(x,a['force_n']).sum(axis=0),0,atol=1e-10)
        values=np.linalg.eigvalsh(m.rest_stiffness().toarray())
        threshold=1e-10*values.max()
        self.assertEqual(int((abs(values)<threshold).sum()),6)
        self.assertGreater(values.min(),-threshold)

    def test_relative_wind_full_dof_power(self):
        m=self.model; x=m.rest_positions.copy(); x[:,1]=.6*(m.xy[:,0]-.25)**2
        v=np.random.default_rng(79).normal(size=x.shape)*.02; v[~m.free]=0
        r=m.aerodynamic_force(x,v,np.zeros(3))
        self.assertLess(r['power_w'],0)
        self.assertAlmostEqual(np.sum(v*r['force_n']),r['power_w'],delta=1e-16)
        zero=m.aerodynamic_force(x,v,np.zeros(3),active=False)
        np.testing.assert_array_equal(zero['force_n'],np.zeros_like(x))

    def test_clamp_normal_frame_couple_completes_torque_balance(self):
        m=self.model; u=np.zeros_like(m.rest_positions)
        u[:,1]=.25*(m.xy[:,0]-.25)**2
        r=m.evaluate_displacement(u)
        torque=np.cross(m.rest_positions+u,r['force_n']).sum(axis=0)
        self.assertGreater(np.linalg.norm(torque),1e-4)
        np.testing.assert_allclose(torque,r['fixed_normal_torque_on_shell_n_m'],rtol=1e-9,atol=1e-10)

    def test_signed_surface_map_cubic_reproduction_and_work(self):
        from wind3dgs.teacher.p3_shell import P3Shell
        rng=np.random.default_rng(82)
        xy=rng.uniform([.25,0.],[1.,1.],size=(73,2))
        xy=np.vstack((xy,[[.25,0.],[.25,1.],[1.,0.],[1.,1.],[.5,.5]]))
        for diagonal in ('forward','backward'):
            m=P3Shell(4,diagonal=diagonal); P=m.moving_surface_map(xy)
            for a in range(4):
                for b in range(4-a):
                    np.testing.assert_allclose(P@(m.xy[:,0]**a*m.xy[:,1]**b),xy[:,0]**a*xy[:,1]**b,atol=1e-14)
            self.assertLess(P.data.min(),0.)
            v=rng.normal(size=m.rest_positions.shape); f=rng.normal(size=(len(xy),3))
            self.assertAlmostEqual(np.sum((P@v)*f),np.sum(v*(P.T@f)),places=12)


if __name__ == '__main__': unittest.main()
