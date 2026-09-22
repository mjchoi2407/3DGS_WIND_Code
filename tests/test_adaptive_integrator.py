import unittest

from wind3dgs.teacher.resident_adaptive_integrator import branch_decision


class AdaptiveIntegratorDecisionTests(unittest.TestCase):
    def test_requires_three_early_retryable_failures(self):
        result=branch_decision(early_failures=2,completed=8,elapsed_s=20.,gauss_block_times_s=[.4,.4])
        self.assertFalse(result['switch'])
        self.assertEqual(result['reason'],'insufficient_evidence')

    def test_switches_only_when_observed_remaining_cost_is_larger(self):
        fast_gauss=branch_decision(early_failures=3,completed=8,elapsed_s=20.,gauss_block_times_s=[.4,.5,.45])
        self.assertTrue(fast_gauss['switch'])
        self.assertEqual(fast_gauss['reason'],'early_retry_density_and_cost')
        slow_gauss=branch_decision(early_failures=3,completed=16,elapsed_s=5.,gauss_block_times_s=[1.,1.,1.])
        self.assertFalse(slow_gauss['switch'])
        self.assertEqual(slow_gauss['reason'],'newmark_remaining_not_slower')

    def test_invalid_timing_never_forces_switch(self):
        result=branch_decision(early_failures=3,completed=0,elapsed_s=0.,gauss_block_times_s=[.4,.4,.4])
        self.assertFalse(result['switch'])
        self.assertEqual(result['reason'],'timing_unavailable')

    def test_direct_gauss_estimate_uses_one_eight_step_block_per_base_step(self):
        result=branch_decision(early_failures=3,completed=8,elapsed_s=20.,
                               gauss_block_times_s=[.4,.5,.6],steps=64)
        self.assertAlmostEqual(result['estimated_direct_gauss_s'],32.)


if __name__=='__main__':unittest.main()
