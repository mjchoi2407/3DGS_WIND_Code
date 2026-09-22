import unittest
import numpy as np
from wind3dgs.teacher.p3_shell import P3Shell
from wind3dgs.teacher.sample_meshes import SampleClothMesh,SampleMeshKind,make_sample_mesh

class SampleShellTests(unittest.TestCase):
    def test_original_rectangle_equivalence(self):
        old=P3Shell(4);xy=old.vertex_xy
        v=np.column_stack((xy[:,0],np.zeros(len(xy)),.5-xy[:,1])).astype(np.float32)
        pinned=xy[:,0]==.25
        mesh=SampleClothMesh(SampleMeshKind.RECTANGULAR_FLAG,v,old.triangles.astype(np.int32),
            np.column_stack(((xy[:,0]-.25)/.75,xy[:,1])).astype(np.float32),pinned,pinned.astype(np.int8),{})
        new=P3Shell(sample_mesh=mesh)
        np.testing.assert_array_equal(new.free,old.free)
        np.testing.assert_array_equal(new.rest_positions,old.rest_positions)
        np.testing.assert_allclose(new.mass.toarray(),old.mass.toarray(),rtol=0,atol=0)
        u=np.random.default_rng(12).normal(size=old.rest_positions.shape)*1e-5;u[~old.free]=0
        for k in ['force_n','energy_j']:
            np.testing.assert_allclose(new.evaluate_displacement(u)[k],old.evaluate_displacement(u)[k],rtol=0,atol=0)
    def test_sample_rest_mass_and_tangent(self):
        for kind in ['triangular_flag','handkerchief']:
            with self.subTest(kind=kind):
                mesh=make_sample_mesh(kind,resolution=(4,4));m=P3Shell(sample_mesh=mesh)
                u=np.zeros_like(m.rest_positions);e=m.evaluate_displacement(u)
                self.assertLess(np.max(abs(e['force_n'])),1e-10)
                v=mesh.vertices[mesh.faces].astype(float);area=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1).sum()/2
                self.assertAlmostEqual(m.mass.sum(),area*m.density,places=12)
                np.testing.assert_array_equal(~m.free[:len(mesh.vertices)],mesh.pinned)
                self.assertGreater(np.count_nonzero(~m.free),0)
                rng=np.random.default_rng(19);u=rng.normal(size=u.shape)*1e-5;d=rng.normal(size=u.shape);d[~m.free]=0;u[~m.free]=0
                h=m.evaluate_displacement(u,direction=d)['hvp_n'];eps=1e-7
                fd=-(m.evaluate_displacement(u+eps*d)['force_n']-m.evaluate_displacement(u-eps*d)['force_n'])/(2*eps)
                np.testing.assert_allclose(h,fd,atol=2e-5,rtol=2e-5)
                with self.assertRaises(ValueError):m.moving_surface_map([[.5,.5]])
if __name__=='__main__':unittest.main()
