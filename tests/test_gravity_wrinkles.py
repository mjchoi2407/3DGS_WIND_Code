import unittest
import tempfile
import signal
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
from wind3dgs.evaluation.teacher_gravity_wrinkles import smooth_ramp,settings,valid,may_branch
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.resident_gravity import gravity_load

class GravityTests(unittest.TestCase):
    def test_bending_variant_preserves_membrane_and_mass_parameters(self):
        from wind3dgs.evaluation.teacher_gravity_wrinkles import with_bending_ratio
        from wind3dgs.teacher.shell_structure import ShellElasticMaterial
        base={'case':'bend_001','bending_ratio':.01,'material':{'E_pa':1e7,'h_m':.001,'nu':.3,'area_density_kg_m2':.1}}
        changed=with_bending_ratio(base,1/300)
        def scales(p):
            m=p['material'];return ShellElasticMaterial(m['E_pa'],m['nu'],m['h_m']).scales()
        old,new=scales(base),scales(changed)
        self.assertAlmostEqual(new[0]/old[0],1.)
        self.assertAlmostEqual(new[1]/old[1],1/3)
        self.assertEqual(base['material']['area_density_kg_m2'],changed['material']['area_density_kg_m2'])
        self.assertEqual(base['bending_ratio'],.01)
        for target in (0.,-1.,float('nan'),.02):
            with self.assertRaises(ValueError):with_bending_ratio(base,target)

    def test_consistent_gravity_force_and_work(self):
        m=P3Shell(4);g=np.array([0.,0.,-9.81]);f=gravity_load(m,g)
        np.testing.assert_allclose(f.sum(0),float(m.mass.sum())*g,rtol=1e-14,atol=1e-14)
        delta=np.random.default_rng(3).normal(size=f.shape)*.001
        expected=float(np.sum((m.mass@delta)*g))
        self.assertAlmostEqual(float(np.sum(f*delta)),expected,places=14)
        self.assertEqual(float(np.max(abs(f[:,0:2]))),0.)
    def test_ramp_monotonic_and_full_strength(self):
        r=smooth_ramp(300,60);self.assertTrue(np.all(np.diff(r)>=0));self.assertLess(r[0],.001)
        np.testing.assert_array_equal(r[59:],1.)
    def test_invalid_schedule_rejected(self):
        c=settings();c['preload_frames']=10
        with self.assertRaises(ValueError):valid(c)
    def test_preload_completion_not_speed_controls_branch(self):
        self.assertFalse(may_branch({'status':'complete','preload_complete':False},settings()))
        self.assertFalse(may_branch({'status':'interrupted','preload_complete':True},settings()))
        self.assertTrue(may_branch({'status':'complete','preload_complete':True,'last_rms_speed_m_s':10.},settings()))
    def test_interrupt_stops_owned_worker_and_preserves_saved_prefix(self):
        from wind3dgs.evaluation import teacher_gravity_wrinkles as run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run.write(root/'manifest.json',{});run.write(root/'config.json',settings(shape='handkerchief'))
            for phase in run.PHASES:
                folder=root/phase/'handkerchief';folder.mkdir(parents=True)
                run.write(folder/'report.json',{'status':'ready','completed_frames':0})
            handlers={};proc=Mock(pid=123456)
            def install(sig,handler):
                previous=handlers.get(sig,signal.SIG_DFL);handlers[sig]=handler;return previous
            def output():
                run.write(root/'preload/handkerchief/report.json',{'status':'running','completed_frames':120})
                handlers[signal.SIGINT](signal.SIGINT,None)
                yield ''
            proc.stdout=output()
            with patch.object(run.signal,'signal',side_effect=install),patch.object(run.os,'killpg') as kill,patch.object(run.subprocess,'Popen',return_value=proc) as popen:
                with self.assertRaises(KeyboardInterrupt):run.controller(root)
                kill.assert_called_once_with(proc.pid,signal.SIGTERM)
                self.assertTrue(popen.call_args.kwargs['start_new_session'])
            report=run.read(root/'preload/handkerchief/report.json')
            self.assertEqual(report['status'],'interrupted');self.assertEqual(report['completed_frames'],120)
            self.assertEqual(run.read(root/'wind/handkerchief/report.json')['status'],'ready')

if __name__=='__main__':unittest.main()
