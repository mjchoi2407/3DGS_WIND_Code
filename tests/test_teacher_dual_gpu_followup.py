"""GPU를 실행하지 않고 v2 일정·결과 집계·forcing 경계를 검증한다."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wind3dgs.evaluation import teacher_dual_gpu_followup as v


class FollowupTests(unittest.TestCase):
    def test_schedule_and_aggregation_fixture_only(self):
        calls=[]
        def prepare(source,root,case):
            root.mkdir(parents=True)
            return {'fps':60}
        def command(logs,name,cmd,env):
            calls.append(cmd)
            out=Path(cmd[cmd.index('--out')+1]);out.mkdir()
            stage=cmd[cmd.index('--stage')+1]
            if stage=='environment':result={'gpu':'RTX 5070'}
            elif stage=='frame':
                result=dict(passed=True,setup_s=1,preprocess_s=1,transfer_s=0,save_s=0,compute_audit_wall_s=1,solver_s=.8,audit_s=.2,
                    rows=[{'physical_start_s':3.,'attempts':[]}],graph=[],graph_error=None,launches=[])
                np.savez(out/'audit_0000.npz',dt_s=[1/60],method=[0],flags=[0],checks=np.zeros((1,6)))
                np.savez(out/'snapshot_0000.npz',time_s=3.,free=[True],u_hi=np.zeros((1,3)),u_lo=np.zeros((1,3)),v_hi=np.zeros((1,3)),v_lo=np.zeros((1,3)),force=np.zeros((1,3)),elastic_j=0.,kinetic_j=0.)
            else:
                mode=cmd[cmd.index('--measurement-mode')+1]
                result=dict(finite=True,force_status=0,rows=[dict(role='assembled_force',precision='fixture',block_dim=256,measurement_mode=mode,repeat_id=i,gpu_us_per_call=1.) for i in range(5)])
                np.savez(out/'outputs.npz',force=np.zeros((1,3)),volume_geometry=np.ones((1,1,4)),free=[True])
            v.write(out/'result.json',result)
            return {'returncode':0,'process_wall_s':1.}
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(out=Path(temp)/'run',gpu_profile='rtx5070',worker_id='fixture',source_run=Path('unused'),wind_source=Path('unused'),prepare_only=False)
            with patch.object(v,'prepare_case',prepare),patch.object(v,'run_command',command),patch.object(v,'telemetry',lambda *x:None),patch.object(v,'package',lambda p:'fixture_only'):
                v.run(args)
            summary=v.read(args.out/'summary.json')
            self.assertEqual(summary['status'],'measurement_attempts_complete')
            self.assertEqual(summary['cases']['C0']['stats']['A']['n'],3)
            self.assertEqual(summary['cases']['W1']['stats']['B']['n'],6)
            self.assertEqual(summary['cases']['W1']['decision']['numerical_regression'],'budget_not_defined')
            fixed=[c for c in calls if c[c.index('--stage')+1]=='fixed']
            self.assertEqual(len(fixed),24)
            self.assertTrue(all(c[c.index('--calls')+1]=='20' and c[c.index('--repeats')+1]=='5' for c in fixed))
            names=[Path(c[c.index('--out')+1]).name for c in calls]
            i=names.index('pair00_A')
            self.assertEqual(names[i:i+4],['pair00_A','pair00_B','pair01_B','pair01_A'])

if __name__=='__main__':unittest.main()
