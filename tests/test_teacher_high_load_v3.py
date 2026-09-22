import unittest
import numpy as np
from wind3dgs.evaluation.teacher_high_load_v3 import force_metrics,event_window,merge_windows,classify

class HighLoadTests(unittest.TestCase):
    def test_gravity_pins_and_cancelling_forces(self):
        f=np.array([[[1.,0,0],[-1.,0,0],[999.,0,0]],[[0,1.,0],[0,-1.,0],[999.,0,0]]])
        gravity=np.array([[0.,0,-9.],[0,0,-9.]])
        weights=np.array([2.,3.,4.])
        r=force_metrics(f+weights[None,:,None]*gravity[:,None,:],gravity,weights,np.array([0,1]),60)
        np.testing.assert_allclose(r['wind_free_l2_n'],np.sqrt(2))
        np.testing.assert_allclose(r['wind_local_node_max_n'],1)
        self.assertAlmostEqual(r['direction_angle_rad'][1],np.pi/2)
        self.assertAlmostEqual(r['direction_change_weighted_n_s'][1],120)
    def test_windows_and_censoring(self):
        w=event_window(np.arange(240),236)
        self.assertEqual(w['end'],240);self.assertTrue(w['right_censored'])
        rows=[dict(shape=s,begin=a,end=b,events=[{}]) for s,a,b in [('a',1,5),('a',4,8),('b',1,5)]]
        merged=merge_windows(rows);self.assertEqual(len(merged),2);self.assertEqual(merged[0]['end'],8)
    def test_classification(self):
        r=dict(passed=True,total_s=10)
        self.assertEqual(classify(dict(passed=False),None),'baseline_failure')
        self.assertEqual(classify(r,dict(passed=False)),'mixed_failure')
        self.assertEqual(classify(r,dict(passed=True,total_s=11)),'passed_without_speedup')
        self.assertEqual(classify(r,dict(passed=True,total_s=5,fallback_calls=1)),'passed_with_fp64_return_and_speedup')
        self.assertEqual(classify(r,dict(passed=True,total_s=5)),'fp32_centered_passed_with_speedup')
    def test_trace_kernels_cpu(self):
        import warp as wp
        from wind3dgs.teacher import resident_linear_trace_v3 as t
        wp.config.kernel_cache_dir='/tmp/wind_highload_cpu_cache';wp.init()
        def ia(x):return wp.array(x,dtype=wp.int32,device='cpu')
        def da(x):return wp.array(x,dtype=wp.float64,device='cpu')
        rows=wp.zeros((2,22),dtype=wp.float64,device='cpu');step=np.zeros(15,dtype=np.int32);step[14]=3
        wp.launch(t.trace_begin,1,inputs=[ia([0]),ia(step),da([.1]),da([4]),rows,-1,ia([0]),ia([2]),64,ia([10]),ia([7])],device='cpu')
        self.assertEqual(rows.numpy()[0,2],13);self.assertEqual(rows.numpy()[0,8],7)
        step[5]=1;step[6]=4
        wp.launch(t.record_line_start,1,inputs=[ia([1]),ia(step),rows],device='cpu')
        wp.launch(t.record_newton,1,inputs=[ia([1]),ia(step),rows],device='cpu')
        self.assertEqual(rows.numpy()[0,18],4)
