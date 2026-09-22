"""보정 연산의 취소 오차 복구와 시험본 경계 검증."""
import os
import unittest
from decimal import Decimal, localcontext
import numpy as np
import warp as wp
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.p3_shell_warp_precision import P3ShellWarpPrecision,P3ShellWarpPrecisionStepper
from wind3dgs.teacher.p3_shell_dynamics import ShellState
from wind3dgs.teacher.p3_shell_warp_precision_kernels import two_sum, two_product


@wp.kernel
def arithmetic(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
               sums: wp.array(dtype=wp.vec2d), products: wp.array(dtype=wp.vec2d)):
    i=wp.tid()
    sums[i]=two_sum(a[i],b[i])
    products[i]=two_product(a[i],b[i])


wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


class PrecisionTests(unittest.TestCase):
    def test_error_terms_recover_exact_arithmetic(self):
        device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu')
        a=np.array([1.,1.+2**-27,1e-8,1e8])
        b=np.array([2**-54,1.-2**-27,-1e-8,1e-8])
        x=wp.array(a,dtype=wp.float64,device=device);y=wp.array(b,dtype=wp.float64,device=device)
        s=wp.zeros(4,dtype=wp.vec2d,device=device);p=wp.zeros_like(s)
        wp.launch(arithmetic,dim=4,inputs=[x,y,s,p],device=device)
        with localcontext() as context:
            context.prec=100
            for ai,bi,si,pi in zip(a,b,s.numpy(),p.numpy()):
                da,db=Decimal(float(ai)),Decimal(float(bi))
                self.assertEqual(sum(Decimal(float(v)) for v in si),da+db)
                self.assertEqual(sum(Decimal(float(v)) for v in pi),da*db)

    def test_rest_and_invalid_input(self):
        m=P3ShellWarpPrecision(P3Shell(4),device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu'))
        u=np.zeros_like(m.rest_positions,dtype=np.longdouble)
        np.testing.assert_array_equal(m.evaluate_displacement(u)['force_n'],0)
        with self.assertRaises(ValueError):m.evaluate_displacement(u,direction=u)
        with self.assertRaises(ValueError):m.evaluate_displacement(u[:-1])
        u[0,0]=np.nan
        with self.assertRaises(ValueError):m.evaluate_displacement(u)

    def test_step_and_reset_preserve_extended_state(self):
        m=P3ShellWarpPrecision(P3Shell(4),device=os.environ.get('WIND3DGS_P3_SHELL_DEVICE','cpu'))
        s=P3ShellWarpPrecisionStepper(m);initial=s.state()
        state,_=s.step(initial,np.zeros_like(m.rest_positions),1/600)
        self.assertEqual(state.displacement_m.dtype,np.dtype(np.longdouble))
        u=np.zeros_like(state.displacement_m)
        u[m.free,1]=np.longdouble(.001)+np.longdouble(2)**-65
        before=s.state(displacement=u);after,removed=s.reset_velocity(before)
        np.testing.assert_array_equal(after.displacement_m,u)
        self.assertEqual(removed,0.)
        with self.assertRaises(ValueError):
            s.step(ShellState(u.astype(float),u.astype(float),0.),np.zeros_like(u),1/600)
