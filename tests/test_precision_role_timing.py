"""CUDA 없이 계측 회계/분류 함수를 원본 AST에서 검사한다."""
import ast
from pathlib import Path
import unittest
from wind3dgs.evaluation.teacher_extended_precision_probe import collect_roles,write
import tempfile

source=Path('code/wind3dgs/teacher/resident_precision_role_timing.py').read_text()
tree=ast.parse(source)
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('category','aggregate')]
selected += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROLES' for t in n.targets)]
space={};exec(compile(ast.Module(body=selected,type_ignores=[]),'timing_pure','exec'),space)


class RoleTimingTest(unittest.TestCase):
    def test_nested_accounting(self):
        report={'regions':[dict(path='nonlinear._step',inclusive_s=10.,exclusive_s=2.,calls=1),
          dict(path='nonlinear._step/probe.build',inclusive_s=8.,exclusive_s=3.,calls=1),
          dict(path='nonlinear._step/probe.build/P32.factor',inclusive_s=5.,exclusive_s=5.,calls=1)]}
        d=space['aggregate'](report,12.)
        self.assertTrue(d['valid_accounting'])
        self.assertEqual(sum(r['time_s'] for r in d['rows']),12.)
        self.assertEqual(next(r['time_s'] for r in d['rows'] if r['role']=='P_assembly'),3.)

    def test_assembly_and_anchor_hvp_are_not_inner_hvp(self):
        classify=space['category']
        self.assertEqual(classify('nonlinear._step/probe.build/A32.matvec'),'P_assembly')
        self.assertEqual(classify('nonlinear._step/mixed.true_residual/A64.action'),'master_true_residual')
        self.assertEqual(classify('audit.submit/P64.matvec'),'independent_audit')
        self.assertEqual(classify('nonlinear._step/P64.matvec'),'nonlinear_force_state')

    def test_overlap_not_hidden(self):
        d=space['aggregate']({'regions':[dict(path='root',inclusive_s=3.,exclusive_s=3.,calls=1)]},2.)
        self.assertFalse(d['valid_accounting'])
        self.assertEqual(d['rows'][-1]['time_s'],-1.)

    def test_fp64_return_is_non_additive_per_role(self):
        report={'regions':[
            dict(path='nonlinear._step',inclusive_s=5.,exclusive_s=1.,calls=1),
            dict(path='nonlinear._step/probe.rejected_assembly',inclusive_s=4.,exclusive_s=1.,calls=1),
            dict(path='nonlinear._step/probe.rejected_assembly/coloring.assemble',inclusive_s=3.,exclusive_s=3.,calls=1)]}
        d=space['aggregate'](report,5.)
        self.assertEqual(d['fp64_return_by_role_s']['P_assembly'],4.)
        self.assertEqual(sum(r['time_s'] for r in d['rows']),5.)

    def test_gauss_retry_is_separate_fp64_role(self):
        self.assertEqual(space['category']('gauss.retry_fp64/P64.factor'),'gauss_retry_fp64')

    def test_report_has_requested_columns(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);write(root/'selection.json',[dict(name='sample',frame=189)])
            write(root/'config.json',dict(full_frame=True))
            for lane in ('M2','M2_extended32'):
                p=root/'runs'/lane;p.mkdir(parents=True)
                write(p/'result.json',dict(case='sample',lane=lane,repeat=0,passed=True))
                write(p/'role_times.json',dict(valid_accounting=True,fp64_return_by_role_s={'P_assembly':.25},
                      rows=[dict(role='P_assembly',time_s=1.,wall_fraction=1.,instrumented=True)]))
            collect_roles(root)
            self.assertIn('M2 평균(s)',(root/'role_report.md').read_text())
            self.assertIn('보조 행렬 조립',(root/'role_report.md').read_text())
            self.assertIn('(FP64 전환 0.250000)',(root/'role_report.md').read_text())


if __name__=='__main__':unittest.main()
