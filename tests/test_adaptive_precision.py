"""적응 정밀도의 GPU와 동일한 판정 커널을 실제 Warp CPU에서 확인한다."""
import unittest
import numpy as np
import warp as wp
from wind3dgs.teacher import resident_adaptive_precision as p


class AdaptiveDecisionTests(unittest.TestCase):
    def setUp(self):
        self.control=wp.array([0,0,0,0,1,1],dtype=wp.int32,device='cpu')
        self.counts=wp.zeros(10,dtype=wp.int32,device='cpu')
        self.gc=wp.zeros(9,dtype=wp.int32,device='cpu')
        self.gs=wp.zeros(10,dtype=wp.float64,device='cpu')
        self.history=wp.zeros(1,dtype=wp.float64,device='cpu')
        self.ratios=wp.zeros(2,dtype=wp.float64,device='cpu')
        self.step=wp.array([0,1]+[0]*15,dtype=wp.int32,device='cpu')

    def start(self,norm=1.,tol=1e-10):
        wp.launch(p.start,1,[self.control,self.counts,self.gc,self.gs,
            wp.array([norm*norm],dtype=wp.float64,device='cpu'),
            wp.array([tol],dtype=wp.float64,device='cpu'),self.step,self.history],device='cpu')

    def decide(self,norm,budget=2):
        wp.launch(p.decide,1,[self.control,self.counts,self.gc,self.gs,
            wp.array([norm*norm],dtype=wp.float64,device='cpu'),self.history,budget,wp.float64(.5),self.ratios],device='cpu')

    def test_accepts_only_original_target_and_zero_rhs(self):
        self.start();self.decide(1e-8)
        self.assertEqual(self.control.numpy()[0],1)
        self.decide(5e-11)
        self.assertEqual(self.control.numpy()[0],0)
        self.assertEqual(self.control.numpy()[1],0)
        self.assertLessEqual(self.gs.numpy()[5],self.gs.numpy()[1])
        self.start(norm=0.)
        self.assertFalse(np.any(self.control.numpy()[:4]))

    def test_budget_and_stagnation_disable_until_new_matrix(self):
        self.start();self.decide(.1);self.decide(.01)
        self.assertEqual(list(self.control.numpy()[[0,1,5]]),[0,1,0])
        self.start();self.assertEqual(self.control.numpy()[1],1)
        wp.launch(p.rebuilt,1,[self.control,self.counts,1],device='cpu')
        self.assertEqual(list(self.control.numpy()[[4,5]]),[0,1])
        self.start();self.decide(.75)
        self.assertEqual(self.counts.numpy()[7],1)

    def test_nonfinite_cannot_be_accepted(self):
        self.start();self.decide(float('nan'))
        self.assertEqual(self.control.numpy()[1],1)
        self.assertEqual(self.counts.numpy()[2],0)

    def test_rhs_change_resets_residual_and_iteration(self):
        self.start();self.decide(1e-11)
        self.start(norm=2.,tol=1e-4)
        self.assertEqual(self.control.numpy()[3],0)
        self.assertAlmostEqual(self.gs.numpy()[1],2e-4)
        self.assertEqual(self.history.numpy()[0],2.)

    def test_cast_and_correction_keep_fp64_accumulator(self):
        x=wp.array([1.,1e-10],dtype=wp.float64,device='cpu')
        residual=wp.array([1e-8,1e-12],dtype=wp.float64,device='cpu')
        low=wp.zeros(2,dtype=wp.float32,device='cpu')
        wp.launch(p.cast32,2,[residual,low],device='cpu')
        wp.launch(p.add32,2,[x,low],device='cpu')
        self.assertGreater(x.numpy()[0],1.)
        np.testing.assert_array_equal(x.numpy(),np.array([1.,1e-10])+low.numpy().astype(np.float64))


if __name__=='__main__':unittest.main()
