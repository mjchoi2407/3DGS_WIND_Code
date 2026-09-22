"""GPU 실행 없이 입력 선정과 부모 import 경계를 확인한다."""
import ast
from pathlib import Path
import unittest
from wind3dgs.evaluation.teacher_highload_precision_bench import pick


class SelectionTest(unittest.TestCase):
    def test_selection_uses_cost_iterations_and_distinct_median(self):
        cases=[dict(frame=i,original_timing=dict(compute_audit_s=i+1,gmres_iterations=100 if i==1 else i,attempts=[])) for i in range(7)]
        selected=pick(cases)
        self.assertEqual([c['frame'] for c in selected],[6,1,3])
        self.assertEqual(len({c['frame'] for c in selected}),3)

    def test_parent_has_no_cuda_import(self):
        source=Path('code/wind3dgs/evaluation/teacher_highload_precision_bench.py').read_text()
        imports=[n for n in ast.parse(source).body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertTrue(all('warp' not in ast.unparse(n) and 'cupy' not in ast.unparse(n) for n in imports))


if __name__=='__main__':
    unittest.main()
