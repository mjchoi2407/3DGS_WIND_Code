"""Gauss 역할별 GPU 계측의 순수 분류와 회계를 검사한다."""
import ast
from pathlib import Path
import unittest

source=Path('code/wind3dgs/teacher/resident_gauss_role_timing.py').read_text()
tree=ast.parse(source)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('category','aggregate')]
nodes += [n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROLES' for t in n.targets)]
space={};exec(compile(ast.Module(body=nodes,type_ignores=[]),'gauss_timing_pure','exec'),space)


class GaussRoleTimingTest(unittest.TestCase):
    def test_factor_inside_build_is_separate(self):
        classify=space['category']
        self.assertEqual(classify('mixed32.block/mixed._build/P32.build/P32.factor'),'P_factor_apply')
        self.assertEqual(classify('mixed32.block/mixed._build/P32.build'),'P_assembly')
        self.assertEqual(classify('mixed32.block/mixed._build/P32.build/coloring.assemble/A32.action'),'P_assembly')

    def test_fallback_overlay_is_inclusive_in_role_total(self):
        report={'regions':[
            dict(path='mixed32.block',inclusive_s=3.,exclusive_s=3.,calls=1),
            dict(path='fallback64.block',inclusive_s=2.,exclusive_s=1.,calls=1),
            dict(path='fallback64.block/P64.matvec',inclusive_s=1.,exclusive_s=1.,calls=1),
        ]}
        value=space['aggregate'](report,5.)
        self.assertTrue(value['valid_accounting'])
        self.assertEqual(sum(x['time_s'] for x in value['rows']),5.)
        self.assertEqual(value['fp64_return_by_role_s']['P_factor_apply'],1.)
        self.assertEqual(value['fp64_return_by_role_s']['other_device'],1.)

    def test_audit_wins_over_nested_factor(self):
        self.assertEqual(space['category']('mixed32.block/audit.submit/P64.matvec'),'independent_audit')


if __name__=='__main__':unittest.main()
