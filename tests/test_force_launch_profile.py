import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from wind3dgs.teacher.force_launch_profile import BASELINE, select, save_profile, role_of, force_launches
from wind3dgs.evaluation.teacher_dual_gpu import difference, verified_graph


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'profile.json'
        self.identity={'gpu':'uuid-a','precision':'fp64_hilo','mesh':'mesh-a','build':'build-a'}

    def test_default_and_missing(self):
        self.assertEqual(select('baseline',None,None,self.identity),(BASELINE,'baseline'))
        self.assertEqual(select('cached',None,self.path,self.identity)[0],BASELINE)

    def test_hit_and_identity_changes(self):
        blocks={'volume':32,'interior_edge':64,'boundary_edge':128}
        save_profile(self.path,self.identity,blocks,{'samples':3},regression_status='passed',local_audit_status='passed',budget_source='fixture_only',fixture_only=True)
        self.assertEqual(select('cached',None,self.path,self.identity,allow_fixture=True),(blocks,'cache_hit'))
        for key in self.identity:
            identity=dict(self.identity);identity[key]='changed'
            self.assertEqual(select('cached',None,self.path,identity),(BASELINE,'stale_identity'))

    def test_invalid_and_tampered_profile(self):
        for field,value in [('schema_version',9),('blocks',{'volume':1024}),('local_audit_status','failed'),('precision_policy','fp32')]:
            data=save_profile(self.path,self.identity,BASELINE,{'samples':3},regression_status='passed',local_audit_status='passed',budget_source='fixture_only',fixture_only=True);data[field]=value
            self.path.write_text(json.dumps(data))
            self.assertEqual(select('cached',None,self.path,self.identity)[0],BASELINE)

    def test_unapproved_and_fixture_not_production(self):
        with self.assertRaises(ValueError):
            save_profile(self.path,self.identity,BASELINE,{},regression_status='budget_not_defined',local_audit_status='passed',budget_source=None)
        save_profile(self.path,self.identity,BASELINE,{'fixture':True},regression_status='passed',local_audit_status='passed',budget_source='fixture_only',fixture_only=True)
        self.assertEqual(select('cached',None,self.path,self.identity)[1],'unapproved_or_fixture_profile')

    def test_judgement_nonexact_and_missing(self):
        from wind3dgs.evaluation.teacher_launch_judgement import judgement
        a={'n':6,'median':2.,'mad':.01};b={'n':6,'median':1.,'mad':.01}
        result=judgement(a,b,True,True,[{'status':'finite','linf':1e-14}])
        self.assertEqual(result['performance'],'improved')
        self.assertEqual(result['physical_audit'],'passed')
        self.assertFalse(result['exact_equality_observed'])
        self.assertFalse(result['production_enabled'])
        self.assertEqual(judgement(a,b,True,True,[])['comparison_status'],'missing_or_invalid')

    def test_only_force_roles(self):
        def kernel(key,module='x.p3_shell_warp_precision_kernels'):
            return types.SimpleNamespace(key=key,module=types.SimpleNamespace(name=module))
        args=[None]*14;args[13]=0
        self.assertEqual(role_of(kernel('edge_kernel'),args),'interior_edge')
        args[13]=1
        self.assertEqual(role_of(kernel('edge_kernel'),args),'boundary_edge')
        self.assertEqual(role_of(kernel('volume_kernel'),[]),'volume')
        self.assertIsNone(role_of(kernel('edge_hvp_kernel'),args))
        self.assertIsNone(role_of(kernel('edge_kernel','unrelated'),args))

    def test_actual_graph_dimensions(self):
        r={'graph_error':None,'launches':[{'logical_shape':[5,7],'block_dim':32,'kernel':'volume_kernel'}],
           'graph':[{'kernel':'volume_kernel_hash','block':[32,1,1],'grid':[2,1,1]}]}
        self.assertTrue(verified_graph(r))
        r['graph'][0]['block'][0]=256
        self.assertFalse(verified_graph(r))

    def test_scoped_launch_override_and_restore(self):
        calls=[]
        def launch(kernel,**kwargs):calls.append(kwargs)
        fake=types.SimpleNamespace(launch=launch)
        kernel=types.SimpleNamespace(key='volume_kernel',module=types.SimpleNamespace(name='x.p3_shell_warp_precision_kernels',options={'block_dim':256}))
        hvp=types.SimpleNamespace(key='volume_hvp_kernel',module=kernel.module)
        with patch.dict('sys.modules',{'warp':fake}):
            with force_launches({'volume':32,'interior_edge':64,'boundary_edge':128},[]):
                fake.launch(kernel,inputs=[],dim=(5,7));fake.launch(hvp,inputs=[],dim=(5,7))
            self.assertIs(fake.launch,launch)
        self.assertEqual(calls[0]['block_dim'],32)
        self.assertNotIn('block_dim',calls[1])

    def test_zero_relative_and_nonfinite(self):
        self.assertIsNone(difference([0.,0.],[1.,0.])['rel2'])
        self.assertEqual(difference([float('nan')],[1.])['status'],'nonfinite')


if __name__=='__main__':unittest.main()
