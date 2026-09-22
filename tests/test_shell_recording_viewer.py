"""표시용 삼각분할과 저장 시간 선택 검사. 시뮬레이션을 실행하지 않는다."""
import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from wind3dgs.evaluation.view_shell_recording import display_faces, frame_at_time, prepare, sha
from wind3dgs.evaluation.teacher_plate_cubic import NODES
from wind3dgs.evaluation.teacher_scene_model import build_scene_model


class RecordingViewerTests(unittest.TestCase):
    def test_recording_uses_plan_resolution_instead_of_old_32(self):
        import wind3dgs
        package = Path(wind3dgs.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'run';folder = root/'reference_rectangle'
            (folder/'frames').mkdir(parents=True)
            plan = {'scene_model_schema':'material_resolution_v1', 'reference_rectangle_resolution':4,
                    'material':{'E_pa':1e7,'nu':.3,'h_m':.001,'area_density_kg_m2':.1}, 'fps':60}
            (root/'plan.json').write_text(json.dumps(plan))
            model = build_scene_model(root, plan, 'reference_rectangle')
            zeros = np.zeros((2,len(model.xy),3), dtype=np.float32)
            trace = folder/'frames/000.npz'
            np.savez(trace, u_hi=zeros, u_lo=zeros, time_s=np.array([0.,1/60]), wind_m_s=np.zeros(3))
            (folder/'report.json').write_text(json.dumps({'status':'complete','completed_frames':1,
                'frames':[{'frame':0,'trace_sha256':sha(trace)}]}))
            frozen = {'runtime/code/wind3dgs/'+str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}
            frozen['plan.json'] = sha(root/'plan.json')
            (root/'manifest.json').write_text(json.dumps(frozen))
            cache = prepare(root, Path(directory)/'cache', 'reference_rectangle')
            positions = np.load(cache/'positions.npy')
            self.assertEqual(positions.shape, (2,len(model.xy),3))
            np.testing.assert_allclose(positions[0], model.rest_positions, rtol=1e-7)
            self.assertEqual(prepare(root, Path(directory)/'cache', 'reference_rectangle'), cache)

    def test_gpu_chunks_select_frame_endpoints(self):
        import wind3dgs
        package=Path(wind3dgs.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';folder=root/'reference_rectangle';folder.mkdir(parents=True)
            (root/'inputs').mkdir()
            plan={'scene_model_schema':'material_resolution_v1','reference_rectangle_resolution':4,
                  'material':{'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1},'fps':60,'substeps':2}
            (root/'plan.json').write_text(json.dumps(plan));model=build_scene_model(root,plan,'reference_rectangle')
            wind=root/'inputs/wind.npz';np.savez(wind,wind_m_s=np.ones((2,3)))
            chunks=[]
            for frame in range(2):
                hi=np.zeros((3,len(model.xy),3));hi[:,model.free,1]=np.arange(frame*2,frame*2+3)[:,None]*.001
                path=folder/f'{frame}.npz';np.savez(path,u_hi=hi,u_lo=np.zeros_like(hi),time_s=np.arange(frame*2,frame*2+3)/120)
                chunks.append({'begin_frame':frame,'end_frame':frame+1,'path':path.name,'files':{path.name:sha(path)}})
            (folder/'report.json').write_text(json.dumps({'status':'complete','completed_frames':2,'backend':'resident_cloth_gpu_v2','chunks':chunks}))
            frozen={'runtime/code/wind3dgs/'+str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}
            frozen.update({'plan.json':sha(root/'plan.json'),'inputs/wind.npz':sha(wind)})
            (root/'manifest.json').write_text(json.dumps(frozen))
            cache=prepare(root,Path(directory)/'cache','reference_rectangle');positions=np.load(cache/'positions.npy')
            np.testing.assert_allclose(positions[-1,model.free,1],model.rest_positions[model.free,1]+.004,atol=1e-8)
            with np.load(cache/'geometry.npz') as z:np.testing.assert_array_equal(z['wind'][1:],np.ones((2,3)))

    def test_gauss_frame_records_are_not_indexed_by_integration_steps(self):
        import wind3dgs
        package=Path(wind3dgs.__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'run';folder=root/'reference_rectangle';folder.mkdir(parents=True)
            (root/'inputs').mkdir()
            plan={'scene_model_schema':'material_resolution_v1','reference_rectangle_resolution':4,
                  'material':{'E_pa':1e6,'nu':.3,'h_m':.01,'area_density_kg_m2':.1},'fps':60,'substeps':512}
            (root/'plan.json').write_text(json.dumps(plan));model=build_scene_model(root,plan,'reference_rectangle')
            wind=root/'inputs/wind.npz';np.savez(wind,wind_m_s=np.ones((1,3)))
            hi=np.zeros((2,len(model.xy),3));hi[1,model.free,1]=.001
            path=folder/'0.npz';np.savez(path,u_hi=hi,u_lo=np.zeros_like(hi),time_s=np.array([0.,1/60]))
            (folder/'report.json').write_text(json.dumps({'status':'complete','completed_frames':1,
                'backend':'resident_gauss_gpu_v1','recorded_substeps':1,
                'chunks':[{'begin_frame':0,'end_frame':1,'path':path.name,'files':{path.name:sha(path)}}]}))
            frozen={'runtime/code/wind3dgs/'+str(p.relative_to(package)):sha(p) for p in package.rglob('*.py')}
            frozen.update({'plan.json':sha(root/'plan.json'),'inputs/wind.npz':sha(wind)})
            (root/'manifest.json').write_text(json.dumps(frozen))
            cache=prepare(root,Path(directory)/'cache','reference_rectangle')
            positions=np.load(cache/'positions.npy')
            self.assertEqual(len(positions),2)
            np.testing.assert_allclose(positions[-1,model.free,1],model.rest_positions[model.free,1]+.001,atol=1e-8)

    def test_p3_display_triangles_cover_element_once(self):
        faces = display_faces(np.arange(10)[None])
        self.assertEqual(faces.shape, (9, 3))
        self.assertEqual(set(faces.ravel()), set(range(10)))
        p = NODES[:, 1:][faces]
        a, b = p[:,1]-p[:,0], p[:,2]-p[:,0]
        signed = a[:,0]*b[:,1]-a[:,1]*b[:,0]
        self.assertTrue(np.all(signed > 0))
        self.assertAlmostEqual(float(signed.sum()/2), .5)
        dofs = np.arange(10,20)[None]
        np.testing.assert_array_equal(display_faces(dofs), faces+10)

    def test_saved_frame_selection_including_final_time(self):
        times = np.array([0., .2, .4, .6])
        self.assertEqual(frame_at_time(times, -.1), 0)
        self.assertEqual(frame_at_time(times, .21), 1)
        self.assertEqual(frame_at_time(times, .39), 2)
        self.assertEqual(frame_at_time(times, .6), 3)
        self.assertEqual(frame_at_time(times, 1.), 3)


if __name__ == '__main__':
    unittest.main()
