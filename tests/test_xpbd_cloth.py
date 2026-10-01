"""새 CG 후보의 최소 독립 계산·접촉·핀·CPU/GPU 동등성 검사."""
import numpy as np
import pytest
from wind3dgs.teacher.xpbd_cloth import Cloth,XPBD,colors,hinge_data,membrane_data


def triangle(pinned=None):
    return Cloth(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),np.array([[0,1,2]]),np.zeros(3,bool) if pinned is None else pinned,.1,np.diag([100.,100.,40.]),.001)


def test_membrane_gradient_and_material_energy():
    c=triangle();x=c.rest+np.array([[.02,.01,.03],[.03,.02,.1],[.0,.03,.02]])
    strain,jac=membrane_data(x,c.grad[0]);rng=np.random.default_rng(7);d=rng.normal(size=x.shape)
    eps=1e-6;fd=(membrane_data(x+eps*d,c.grad[0])[0]-membrane_data(x-eps*d,c.grad[0])[0])/(2*eps)
    np.testing.assert_allclose(fd,np.einsum('ijk,ik->j',jac,d),rtol=1e-8,atol=1e-9)
    assert strain@c.dm@strain*c.area[0]>0


@pytest.mark.parametrize('boundary',[False,True])
def test_hinge_gradient_and_rigid_invariance(boundary):
    x=np.array([[0.,0.,0.],[1.,0.,.1],[.2,1.,.1],[.3,-1.,.2]])
    fixed=np.array([0.,0.,1.]) if boundary else None
    angle,grad=hinge_data(x,fixed);eps=1e-6
    for i in range(3 if boundary else 4):
        for j in range(3):
            d=np.zeros_like(x);d[i,j]=eps
            fd=(hinge_data(x+d,fixed)[0]-hinge_data(x-d,fixed)[0])/(2*eps)
            np.testing.assert_allclose(grad[i,j],fd,rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(grad.sum(0),0,atol=1e-14)
    if not boundary:np.testing.assert_allclose(np.cross(x,grad).sum(0),0,atol=1e-14)


def test_color_no_shared_write_and_mass():
    c=Cloth(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]]),np.array([[0,1,2],[1,3,2]]),np.zeros(4,bool),.1,np.eye(3),.001)
    for rows in (c.faces,c.hinges):
        for group in colors(rows):
            nodes=rows[group].ravel();nodes=nodes[nodes>=0];assert len(nodes)==len(set(nodes))
    assert c.mass.sum()==pytest.approx(.1)


@pytest.mark.parametrize('tau',[0.,.005])
def test_one_step_block_matches_independent_numpy(tau):
    c=triangle(np.array([True,False,False]));x=c.rest.copy();x[1]*=1.02;old=x.astype(np.float32).astype(float)
    vel=np.array([[0.,0.,0.],[.2,.1,0.],[-.1,.1,0.]])
    h=1/60;pred=old+h*vel;strain,jac=membrane_data(pred,c.grad[0]);gamma=tau/h
    mat=(1+gamma)*sum(w*j@j.T for w,j in zip(c.inv_mass,jac))+c.compliance[0]/h**2
    rhs=-strain-gamma*np.einsum('ijk,ik->j',jac,pred-old)
    dl=np.linalg.solve(mat,rhs);expected=pred+np.array([w*j.T@dl for w,j in zip(c.inv_mass,jac)])
    op=XPBD(c,x,vel,substeps=1,membrane_tau=tau,contact_sweeps=0,device='cpu')
    actual,_,status,_=op.step(gravity=(0,0,0))
    assert status==0;np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=2e-7)


def test_free_rigid_translation_and_pin():
    c=triangle();vel=np.tile([.2,.1,.3],(3,1));op=XPBD(c,velocities=vel,substeps=4,device='cpu')
    x,v,status,_=op.step(wind=vel[0],gravity=(0,0,0))
    assert status==0;np.testing.assert_allclose(x,c.rest+vel/60,atol=5e-7)
    c=triangle(np.ones(3,bool));op=XPBD(c,substeps=2,device='cpu')
    x,v,status,_=op.step(wind=(0,0,3))
    assert status==0;np.testing.assert_array_equal(x,c.rest);np.testing.assert_array_equal(v,0)


def test_two_patches_discrete_contact_separates():
    base=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    rest=np.concatenate([base,base+np.array([0,0,.1])]);x=rest.copy();x[3:,2]=.0004
    c=Cloth(rest,np.array([[0,1,2],[3,4,5]]),np.zeros(6,bool),.1,np.eye(3),.001)
    op=XPBD(c,x,substeps=1,membrane_tau=0,thickness=.001,contact_sweeps=4,device='cpu')
    out,_,status,hits=op.step(gravity=(0,0,0));assert status==0 and np.all(hits>0)
    assert np.mean(out[3:,2]-out[:3,2])>.00095


@pytest.mark.parametrize('steps',[1,4])
def test_cpu_cuda_same_small_state(steps):
    import warp as wp
    if not wp.is_cuda_available():pytest.skip('CUDA 없는 환경')
    c=triangle(np.array([True,False,False]));x=c.rest.copy();x[1,0]+=.02
    a=XPBD(c,x,substeps=steps,device='cpu').step(wind=(0,0,1))
    b=XPBD(c,x,substeps=steps,device='cuda:0').step(wind=(0,0,1))
    assert a[2]==b[2]==0
    np.testing.assert_allclose(a[0],b[0],rtol=2e-5,atol=3e-6)
    np.testing.assert_allclose(a[1],b[1],rtol=2e-4,atol=3e-4)


@pytest.mark.parametrize('boundary',[False,True])
def test_bending_kernel_matches_numpy_update(boundary):
    import warp as wp
    from wind3dgs.teacher import xpbd_cloth as m
    x=np.array([[0.,0.,0.],[1.,0.,.1],[.2,1.,.1],[.3,-1.,.2]],dtype=np.float32)
    fixed=np.array([0.,0.,1.]);angle,g=hinge_data(x.astype(float),fixed if boundary else None)
    weights=np.array([0.,0.,2.,3.]);h=.002;compliance=.003;tau=.001
    dl=-angle/((1+tau/h)*np.sum(weights[:,None]*g*g)+compliance/h**2)
    expected=x+weights[:,None]*g*dl
    for device in ('cpu','cuda:0'):
        if device.startswith('cuda') and not wp.is_cuda_available():continue
        op=lambda a,t:wp.array(a,dtype=t,device=device)
        xx=op(x,wp.vec3);status=wp.zeros(1,dtype=int,device=device)
        wp.launch(m.bend,dim=1,inputs=[xx,op(x,wp.vec3),op(weights,wp.float32),op([[0,1,2,-1 if boundary else 3]],wp.vec4i),op([compliance],wp.float32),op([fixed],wp.vec3),op([0],wp.int32),h,tau,status],device=device)
        assert status.numpy()[0]==0
        np.testing.assert_allclose(xx.numpy(),expected,rtol=1e-5,atol=3e-7)


def test_failed_frame_is_saved_but_not_counted_as_accepted(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from wind3dgs.evaluation import teacher_xpbd_pilot as p
    c=triangle()
    class Fixture:
        def __init__(self,*a,**kw):
            self.device=SimpleNamespace(name='fixture');self.x=SimpleNamespace(assign=lambda a:None);self.v=self.x
        def step(self,*a):return c.rest.astype(np.float32),np.zeros_like(c.rest,dtype=np.float32),5,np.zeros(2,int)
    monkeypatch.setattr(p,'XPBD',Fixture)
    r,x,v=p.run_case(tmp_path,'failure',c,c.rest,np.zeros_like(c.rest),np.zeros((2,3)),np.zeros((2,3)),1,'cpu',0.,0.)
    assert r['status']=='failed' and r['completed_frames']==0 and r['recorded_frames']==1
    assert len(r['frames'])==1 and len(x)==2 and (tmp_path/'failure/trajectory.npz').exists()
