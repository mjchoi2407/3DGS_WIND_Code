import unittest

import numpy as np

from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_bounds import P3ShellBounds


class ShellPolynomialBoundsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=P3Shell(4);cls.bounds=P3ShellBounds(cls.model)

    def test_rigid_rotation_has_zero_strain_with_injective_projection(self):
        m=self.model;angle=.9;c,s=np.cos(angle),np.sin(angle)
        x=m.xy[:,0]-.25;u=np.column_stack(((c-1)*x,s*x,np.zeros_like(x)))
        r=self.bounds.interval(u,np.zeros_like(u),u,0)
        self.assertTrue(r['injectivity_sufficient_condition'])
        self.assertLess(r['strain_component_upper']-r['roundoff_margin'],1e-12)
        self.assertGreaterEqual(r['projected_gradient_upper'],1-c)
        self.assertLess(r['projected_gradient_upper']-r['roundoff_margin'],1-c+1e-12)
        self.assertLess(r['engineering_curvature_component_upper_inv_m'],1e-8)

    def test_quadratic_time_overshoot_is_not_hidden_by_valid_endpoints(self):
        m=self.model;zero=np.zeros_like(m.rest_positions);v=zero.copy();v[:,0]=-8*(m.xy[:,0]-.25)
        ends=self.bounds.interval(zero,zero,zero,0)
        interval=self.bounds.interval(zero,v,zero,1)
        self.assertTrue(ends['injectivity_sufficient_condition'])
        self.assertFalse(interval['injectivity_sufficient_condition'])
        self.assertGreaterEqual(interval['projected_gradient_upper'],4.)

    def test_bounds_cover_analytic_cubic_field_at_dense_space_time_points(self):
        m=self.model;s,t=m.xy.T;s=s-.25
        base=np.column_stack((.04*s*s*t,.2*s**3+.1*s*t*t,-.03*s*t))
        zero=np.zeros_like(base);v=4*base
        # u(tau)=4*tau*(1-tau)*base; endpoints0, midpoint base.
        r=self.bounds.interval(zero,v,zero,1)
        x,y=np.meshgrid(np.linspace(0,.75,73),np.linspace(0,1,71))
        for tau in np.linspace(0,1,31):
            w=2*tau*(1-tau)*2
            Fs=np.stack((1+w*.08*x*y,w*(.6*x*x+.1*y*y),-w*.03*y),axis=-1)
            Ft=np.stack((w*.04*x*x,w*.2*x*y,-1-w*.03*x),axis=-1)
            strain=np.stack(((np.sum(Fs*Fs,axis=-1)-1)/2,(np.sum(Ft*Ft,axis=-1)-1)/2,
                             np.sum(Fs*Ft,axis=-1)),axis=-1)
            self.assertLessEqual(float(abs(strain).max()),r['strain_component_upper'])
            Hss=np.stack((w*.08*y,w*1.2*x,np.zeros_like(x)),axis=-1)
            Htt=np.stack((np.zeros_like(x),w*.2*x,np.zeros_like(x)),axis=-1)
            Hst=np.stack((w*.16*x,w*.4*y,np.full_like(x,-w*.06)),axis=-1)
            normal=np.cross(Fs,Ft);normal/=np.linalg.norm(normal,axis=-1)[...,None]
            bending=np.stack([np.sum(h*normal,axis=-1) for h in (Hss,Htt,Hst)],axis=-1)
            self.assertLessEqual(float(abs(bending).max()),r['engineering_curvature_component_upper_inv_m'])
            for sign in (-1,1):
                self.assertLessEqual(float(abs(strain+sign*.005*bending).max()),
                                     r['linearized_fibre_strain_component_upper'])
        self.assertTrue(r['injectivity_sufficient_condition'])


if __name__=='__main__':unittest.main()
