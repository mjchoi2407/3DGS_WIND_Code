import ast
from pathlib import Path
import unittest
import numpy as np
from wind3dgs.evaluation.teacher_precision_v3 import select_systems
from wind3dgs.teacher.resident_linear_trace_v3 import specialize_stepper

class V3ContractTests(unittest.TestCase):
    def test_selection_distinct_and_not_success_cherry_pick(self):
        rows=[dict(solve_id=i,iterations=n,target=t,rhs_l2=1.,linear_failure=0,true_residual=0.) for i,n,t in [(0,1,1e-4),(1,100,1e-4),(2,20,1e-10),(3,30,1e-3)]]
        selected=select_systems(rows)
        self.assertEqual(selected['L1']['solve_id'],1)
        self.assertEqual(selected['L2']['solve_id'],2)
        self.assertEqual(len({v['solve_id'] for v in selected.values()}),3)
        with self.assertRaises(ValueError):select_systems(rows[:2])
    def test_explicit_stepper_only_linear_hooks(self):
        original=Path('code/wind3dgs/teacher/p3_shell_resident_stepper.py').read_text()
        tree=ast.parse(original)
        for mode in ('R64','M1','M2'):
            new=ast.parse(specialize_stepper(original,mode,12))
            original_methods={n.name:ast.dump(n) for n in tree.body[-1].body if isinstance(n,ast.FunctionDef)}
            methods={n.name:ast.dump(n) for n in new.body[-1].body if isinstance(n,ast.FunctionDef)}
            for name in ('action','_evaluate','_line_search','_attempt','_finalize','_commit'):
                self.assertEqual(methods[name],original_methods[name])
    def test_inner_precision_boundary(self):
        source=Path('code/wind3dgs/teacher/resident_inner32_v3.py').read_text()
        self.assertIn('self.V = wp.zeros((self.restart + 1, self.n), dtype=wp.float32',source)
        self.assertIn('self.H = wp.zeros((self.restart, self.restart + 1), dtype=wp.float64',source)
        self.assertIn('self.dotter.compute(self.dot_a64, self.dot_b64)',source)
        self.assertIn('wp.capture_if(self.c[2:3]',source)
    def test_reference_audit_untouched(self):
        source=Path('code/wind3dgs/teacher/resident_precision_v3.py').read_text()
        self.assertNotIn('force_atol',source)
        self.assertNotIn('line_decide',source)
        self.assertIn('self.g.A.matvec(self.g.x,self.g.b,self.r,alpha=-1.,beta=1.)',source)

if __name__=='__main__':unittest.main()

class PipelineGateTests(unittest.TestCase):
    def test_failed_linear_candidates_never_reach_integration(self):
        import argparse,json,tempfile
        from unittest.mock import patch
        from wind3dgs.evaluation import teacher_precision_v3 as v
        calls=[]
        rows=[dict(solve_id=i,iterations=cost,target=t,rhs_l2=1.,true_residual=0.,linear_failure=0,frame=0,substep=i,newton=0,current_P=1) for i,cost,t in [(0,1,1e-4),(1,100,1e-4),(2,20,1e-10),(3,30,1e-3)]]
        selected=v.select_systems(rows)
        def prepare(a):
            for name in ('C0','W1'):(a.out/'cases'/name).mkdir(parents=True)
        def command(logs,name,cmd,env=None):
            calls.append(name);out=Path(cmd[cmd.index('--out')+1]);out.mkdir()
            stage=cmd[cmd.index('--stage')+1];method=cmd[cmd.index('--method')+1]
            if stage=='environment':result={'gpu':'RTX 5070'}
            elif stage=='linear':result=dict(passed=method not in ('M1','M2'),counts=[0]*6,rows=[])
            else:
                result=dict(passed=True,rows=[{'retries':0}],setup_s=0,preprocess_s=0,compute_audit_wall_s=1,solver_s=.9,audit_s=.1,transfer_s=0,save_s=0)
                v.write(out/'linear_trace.json',[dict(overflow=False,rows=rows)])
                if name.startswith('capture_'):
                    label=name.split('_')[1];np.savez(out/'linear_snapshot.npz',fixture=[1])
                    v.write(out/'linear_snapshot.json',dict(selected=selected[label]))
            v.write(out/'result.json',result)
            return {'returncode':0,'process_wall_s':1.}
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(out=Path(temp)/'fixture',gpu='rtx5070',prepare_only=False)
            with patch.object(v,'prepare',prepare),patch.object(v,'prepare_runtime',lambda *x:None),patch.object(v,'run_command',command),patch.object(v,'telemetry',lambda *x:None),patch.object(v,'package',lambda *x:'fixture_only'),patch.object(v.shutil,'which',lambda *x:None):v.run(args)
            self.assertIn('L1_M1',calls);self.assertIn('L1_M2',calls)
            self.assertNotIn('L0_M1',calls);self.assertNotIn('C0_M1_screen',calls)
            self.assertFalse(any('W30' in x for x in calls))
            self.assertIsNone(v.read(args.out/'summary.json')['selected'])
