import unittest
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.evaluation.teacher_p3_shell_comparison import interpolate_trace


class ShellTimeInterpolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=P3Shell(4);s=cls.m.xy[:,0]-.25
        cls.field=np.column_stack((np.zeros_like(s),s*s,np.zeros_like(s)))

    def test_exact_quadratic_and_linear_velocity_on_refined_clock(self):
        time=np.arange(13)/(60*4);U=time[:,None,None]**2*self.field;V=2*time[:,None,None]*self.field
        value,defect=interpolate_trace(self.m,{'u_m':U,'v_m_s':V},4,16)
        q=value['time_s'][:,None,None]
        np.testing.assert_allclose(value['u_m'],q*q*self.field,rtol=0,atol=1e-18)
        np.testing.assert_allclose(value['v_m_s'],2*q*self.field,rtol=0,atol=1e-17)
        self.assertLess(defect,8*np.finfo(float).eps)

    def test_reset_left_limit_is_retained_and_next_interval_uses_zero_velocity(self):
        time=np.arange(13)/(60*4);reset=1/60
        U=np.minimum(time,reset)[:,None,None]*self.field
        V=np.broadcast_to(self.field,U.shape).copy();V[4:]=0
        value,defect=interpolate_trace(self.m,{'u_m':U,'v_m_s':V},4,16,
                                      reset_frame=1,reset_velocity=self.field)
        np.testing.assert_array_equal(value['v_m_s'][16],self.field)
        np.testing.assert_array_equal(value['v_m_s'][17:],0)
        expected=np.broadcast_to(reset*self.field,value['u_m'][16:].shape)
        np.testing.assert_allclose(value['u_m'][16:],expected,rtol=0,atol=1e-18)
        # 시간 차분의 감산/나눗셈 roundoff를 허용한다. 물리 수렴 tolerance와 무관하다.
        self.assertLess(defect,8*np.finfo(float).eps)

    def test_missing_reset_limit_and_incompatible_clocks_are_rejected(self):
        t={'u_m':np.zeros((13,)+self.field.shape),'v_m_s':np.zeros((13,)+self.field.shape)}
        with self.assertRaises(ValueError):interpolate_trace(self.m,t,4,16,reset_frame=1)
        with self.assertRaises(ValueError):interpolate_trace(self.m,t,4,7)


if __name__=='__main__':unittest.main()
