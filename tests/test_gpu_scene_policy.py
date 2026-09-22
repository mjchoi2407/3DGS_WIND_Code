import unittest
from wind3dgs.teacher.gpu_scene_policy import select

class GPUScenePolicyTests(unittest.TestCase):
    def test_known_hl01_boundary(self):
        self.assertEqual([select('NVIDIA GeForce GTX 1080 Ti',13000,'wind','reference_rectangle',f)['method'] for f in (214,215,239,240)],['M1','R64','R64','M1'])
    def test_nonwind_and_other_shapes(self):
        for phase in ('preload','calm'):
            self.assertEqual(select('GTX 1080 Ti',13000,phase,'handkerchief',0)['method'],'R64')
        self.assertEqual(select('GTX 1080 Ti',13000,'wind','handkerchief',225)['method'],'M1')
    def test_5070_environment_guard(self):
        with self.assertRaisesRegex(RuntimeError,'Graph'):select('RTX 5070',13040,'wind','handkerchief',0)
        value=select('RTX 5070',13000,'wind','handkerchief',0)
        self.assertEqual((value['method'],value['blocks']),('M2',[32,32,256]))
        self.assertFalse(value['production_enabled']);self.assertFalse(value['training_eligible'])
    def test_baseline_and_unmeasured_gpu(self):
        self.assertEqual(select('GTX 1080 Ti',13000,'wind','handkerchief',0,'baseline')['method'],'R64')
        for name in ('RTX 5070 Ti','RTX 5090','unknown'):
            self.assertEqual(select(name,13000,'wind','handkerchief',0)['method'],'R64')

    def test_adaptive_integrator_uses_device_specific_newmark_and_gauss(self):
        gtx=select('GTX 1080 Ti',13000,'wind','reference_rectangle',225,'adaptive_integrator')
        rtx=select('RTX 5070',13000,'wind','handkerchief',0,'adaptive_integrator')
        self.assertEqual((gtx['method'],gtx['direct_gauss']),('M1','R64'))
        self.assertEqual((rtx['method'],rtx['direct_gauss'],rtx['blocks']),('M2','MIXED32',[32,32,256]))
        self.assertEqual(select('GTX 1080 Ti',13000,'preload','handkerchief',0,'adaptive_integrator')['method'],'R64')

if __name__=='__main__':unittest.main()
