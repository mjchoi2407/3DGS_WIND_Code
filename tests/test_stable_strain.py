import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from wind3dgs.evaluation.teacher_precision_compare import specialize, GPU_MODULES
from wind3dgs.teacher.strain_precision_specialization import stable_strain_source

ROOT = Path(__file__).resolve().parents[1] / 'wind3dgs/teacher'


class StableStrainTests(unittest.TestCase):
    def function(self):
        s = stable_strain_source((ROOT/'p3_shell_warp_kernels.py').read_text(), 'p3_shell_warp_kernels')
        node = next(n for n in ast.parse(s).body if isinstance(n, ast.FunctionDef) and n.name == 'strain')
        node.decorator_list = []
        node.args.args[0].annotation = None
        env = {'wp': SimpleNamespace(float64=float), 'Triple': SimpleNamespace,
               'dot': lambda a,b: np.array([a.v@b.v, a.d@b.v+a.v@b.d])}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'strain', 'exec'), env)
        return env['strain']

    def geometry(self, d, direction):
        t = np.array([[1.,0.,0.],[0.,0.,-1.]])
        return SimpleNamespace(d0=SimpleNamespace(v=d[0],d=direction[0]),
            d1=SimpleNamespace(v=d[1],d=direction[1]),
            t0=SimpleNamespace(v=t[0],d=np.zeros(3)),t1=SimpleNamespace(v=t[1],d=np.zeros(3)))

    def test_finite_rotation_strain_and_directional_derivative(self):
        fn=self.function();rng=np.random.default_rng(7)
        t=np.array([[1.,0.,0.],[0.,0.,-1.]])
        angle=1.7;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        for d in [t@R.T-t, rng.normal(size=(2,3))*.2]:
            direction=rng.normal(size=(2,3));r=fn(self.geometry(d,direction))
            f=t+d;E=(f@f.T-np.eye(2))*.5
            np.testing.assert_allclose([r.a[0],r.b[0],r.c[0]],[E[0,0],E[1,1],2*E[0,1]],atol=5e-16)
            h=1e-5;plus=t+d+h*direction;minus=t+d-h*direction
            derivative=(plus@plus.T-minus@minus.T)/(4*h)
            np.testing.assert_allclose([r.a[1],r.b[1],r.c[1]],
                [derivative[0,0],derivative[1,1],2*derivative[0,1]],atol=1e-10)

    def test_small_extension_survives_fp32(self):
        d=np.zeros((2,3),np.float32);d[0,0]=1e-10
        r=self.function()(self.geometry(d,np.zeros((2,3),np.float32)))
        self.assertAlmostEqual(r.a[0]/1e-10,1.,places=6)
        self.assertEqual(np.float32(1.+float(d[0,0]))-np.float32(1.),0.)

    def test_all_generated_modules_and_reference_preservation(self):
        for lane in ('reference_hilo','fp64','fp32_hilo','fp32'):
            for name in GPU_MODULES:
                source=(ROOT/(name+'.py')).read_text()
                for variant in ('stable','stable_normal_pair','stable_geometry_pair','stable_metric_pair'):
                    result=specialize(source,name,lane,diagnostic=True,strain_formula=variant)
                    ast.parse(result)
                    if not lane.startswith('fp32'):
                        self.assertEqual(result,specialize(source,name,lane,diagnostic=True))

    def test_source_contract_fails_closed(self):
        with self.assertRaises(ValueError):stable_strain_source('pass','p3_shell_warp_kernels')


if __name__ == '__main__':unittest.main()
