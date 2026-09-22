import ast
from pathlib import Path
import unittest
from wind3dgs.evaluation.teacher_precision_compare import specialize,GPU_MODULES

class PrecisionSpecializationTests(unittest.TestCase):
    def test_specialization_is_explicit_and_syntactically_valid(self):
        root=Path(__file__).resolve().parents[1]/'wind3dgs/teacher'
        for lane in ('reference_hilo','fp32_hilo','fp64','fp32'):
            for name in GPU_MODULES:
                source=specialize((root/(name+'.py')).read_text(),name,lane)
                tree=ast.parse(source)
                if lane.startswith('fp32'):
                    self.assertFalse(any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='wp' and n.attr in ('float64','vec2d','vec3d','mat33d') for n in ast.walk(tree)),name)
    def test_plain_pair_math_does_not_compensate(self):
        root=Path(__file__).resolve().parents[1]/'wind3dgs/teacher'
        for lane in ('fp32','fp64'):
            tree=ast.parse(specialize((root/'p3_shell_warp_precision_kernels.py').read_text(),'p3_shell_warp_precision_kernels',lane))
            for n in tree.body:
                if isinstance(n,ast.FunctionDef) and n.name in ('two_sum','two_product','pair_add','pair_scale'):
                    self.assertEqual(len(n.body),1)
                    self.assertIsInstance(n.body[0],ast.Return)
                    self.assertEqual(n.body[0].value.args[1].args[0].value,0.)
    def test_cudss_dtype_descriptor_arity_is_preserved(self):
        root=Path(__file__).resolve().parents[1]/'wind3dgs/teacher'
        for lane in ('fp32','fp32_hilo'):
            source=specialize((root/'p3_shell_cudss.py').read_text(),'p3_shell_cudss',lane,diagnostic=True)
            calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and n.args and isinstance(n.args[0],ast.Constant)]
            csr=next(n for n in calls if n.args[0].value=='cudssMatrixCreateCsr')
            self.assertEqual(len(csr.args),14);self.assertEqual(csr.args[-4].value,0)

    def test_fail_closed_if_cudss_descriptor_contract_changes(self):
        with self.assertRaises(ValueError):specialize('pass','p3_shell_cudss','fp32')

class DiagnosticPolicyTests(unittest.TestCase):
    def function(self,name,fn):
        import types,numpy as np
        from wind3dgs.teacher.precision_diagnostic_policy import diagnostic_source
        root=Path(__file__).resolve().parents[1]/'wind3dgs/teacher'
        tree=ast.parse(diagnostic_source((root/(name+'.py')).read_text(),name))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
        node.decorator_list=[]
        for arg in node.args.args:arg.annotation=None
        node.returns=None
        def maximum(a,i,v):a[i]=max(a[i],v)
        def bit_or(a,i,v):a[i]|=v
        wp=types.SimpleNamespace(float64=float,isfinite=np.isfinite,max=max,atomic_max=maximum,atomic_or=bit_or)
        wp.vec2d=lambda a,b:np.array([a,b]);wp.abs=abs;wp.tid=lambda:0
        env={'wp':wp,'pair_add':lambda a,b:a+b,'pair_scale':lambda a,b:a*b};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'diagnostic','exec'),env)
        return env[fn]
    def test_finite_assembly_error_warns_but_nonfinite_stops(self):
        check=self.function('resident_coloring','check_error')
        f=[0,0];check([1e-14],[1.],f);self.assertEqual(f,[0,128])
        f=[0,0];check([float('nan')],[1.],f);self.assertEqual(f,[7,0])
    def test_bounded_nonconvergence_retains_warning(self):
        check=self.function('resident_step_kernels','end_attempt')
        for code in (1,2,3):
            c=[0]*17;c[7]=code;f=[0,0];check(c,f,32)
            self.assertEqual(f,[0,1<<code]);self.assertEqual(c[11],0)
        c=[0]*17;c[7]=4;f=[0,0];check(c,f,32);self.assertEqual(f,[4,0])
    def test_update_tolerance_warns_and_nonfinite_stops(self):
        check=self.function('resident_step_kernels','audit_update')
        args=[[0.],[0.],[0.],[0.],[.1],[0.],[0.],[0.],[0],.01]
        f=[0,0];check(*args,f);self.assertEqual(f,[0,256])
        args[4]=[float('nan')];f=[0,0];check(*args,f);self.assertEqual(f,[8,0])
    def test_warning_is_reset_per_step_without_clearing_fatal(self):
        check=self.function('resident_step_kernels','begin_step')
        c=[0]*17;f=[4,128];check(c,[],f)
        self.assertEqual(f,[4,0]);self.assertEqual(c[11],0)

    def test_fp32_hilo_keeps_compensation_and_correct_splitter(self):
        root=Path(__file__).resolve().parents[1]/'wind3dgs/teacher'
        s=specialize((root/'p3_shell_warp_precision_kernels.py').read_text(),'p3_shell_warp_precision_kernels','fp32_hilo')
        self.assertIn('4097.0',s);self.assertNotIn('134217729.0',s)
        tree=ast.parse(s);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='two_product')
        self.assertGreater(len(fn.body),1)

if __name__=='__main__':unittest.main()
