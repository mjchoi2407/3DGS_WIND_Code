"""Gauss의60Hz 외력 갱신과 각 적분 단계의 독립 GPU 검산 연결."""
import numpy as np
import warp as wp
from .resident_gauss import ResidentGaussStepper
from .resident_gauss_audit import ResidentGaussAudit
from .resident_gravity import add_gravity
from . import resident_aero
from . import p3_shell_warp_kernels as aero_k

@wp.kernel
def expand_acc(acc:wp.array2d(dtype=wp.float64),ids:wp.array(dtype=wp.int32),full:wp.array2d(dtype=wp.float64)):
    s,i=wp.tid();full[s,ids[i]]=acc[s,i]

@wp.kernel
def propagate(flag:wp.array(dtype=wp.int32),failure:wp.array(dtype=wp.int32)):
    if flag[0]!=0:wp.atomic_max(failure,0,99)

class GaussSequence:
    def __init__(self,model,initial,wind,gravity,*,policy,dt,substeps=512,rebuild_every=64):
        self.s=ResidentGaussStepper(model,initial,np.zeros_like(initial[0]),dt=dt,policy=policy,rebuild_every=rebuild_every)
        self.audit=ResidentGaussAudit(model,policy,dt)
        self.wind=wp.array(wind,dtype=wp.vec3d,device='cuda:0');self.gravity=wp.array(gravity,dtype=wp.vec3d,device='cuda:0')
        self.weights=wp.array(np.asarray(model.mass@np.ones(len(model.rest_positions))),dtype=wp.float64,device='cuda:0')
        self.control=wp.zeros(17,dtype=wp.int32,device='cuda:0')
        self.previous=[wp.empty_like(x) for x in self.s.state]
        self.acc=wp.zeros((3,self.s.nfull),dtype=wp.float64,device='cuda:0')
        self.ledger=wp.array(ptr=self.s.energy.ptr+8,shape=(1,),dtype=wp.float64,device='cuda:0')
        self.checks=wp.zeros(substeps*11,dtype=wp.float64,device='cuda:0');self.flags=wp.zeros(substeps,dtype=wp.int32,device='cuda:0')
        wp.load_module(module=__name__,device='cuda:0')
        wp.load_module(module=resident_aero,device='cuda:0')
        wp.load_module(module=aero_k,device='cuda:0')
        self._forcing()
        with wp.ScopedCapture() as cap:self._forcing()
        self.frame_graph=cap.graph
    def _forcing(self):
        s=self.s;m=s.ops.model;b=m._volume;m._force.zero_();m._hvp.zero_()
        wp.launch(resident_aero.aero,dim=b.host.weights.shape,inputs=[s.vec(s.state[0]),s.vec(s.state[2]),b.ids,b.N,b.G,b.H,b.weight,m._t0,m._t1,self.wind,self.control,m._qforce,m._power,m._valid],device='cuda:0')
        wp.launch(aero_k.assemble_aero,dim=(b.elements,10),inputs=[b.N,m._qforce,b.points,b.local,b.dlocal],device='cuda:0')
        b.gather(m._force,m._hvp,'cuda:0');wp.copy(s.held,s.flat_force)
        wp.launch(resident_aero.check,dim=b.host.weights.shape,inputs=[m._valid,s.failure],device='cuda:0')
        wp.launch(add_gravity,dim=len(self.weights),inputs=[self.weights,self.gravity,self.control,s.held],device='cuda:0')
    def start_frame(self,frame):
        control=np.zeros(17,dtype=np.int32);control[16]=frame;self.control.assign(control)
        wp.capture_launch(self.frame_graph)
    def step(self,j):
        for dst,src in zip(self.previous,self.s.state):wp.copy(dst,src)
        self.s.step()
        wp.launch(expand_acc,dim=(3,self.s.n),inputs=[self.s.acc,self.s.ids,self.acc],device='cuda:0')
        self.audit.submit_device(self.previous,self.s.state,[self.s.U,self.s.L,self.s.W,self.s.WL,self.acc],self.s.held,self.ledger)
        wp.launch(propagate,dim=1,inputs=[self.audit.failure,self.s.failure],device='cuda:0')
        wp.copy(self.checks,self.audit.checks,dest_offset=j*11,count=11)
        wp.copy(self.flags,self.audit.failure,dest_offset=j,count=1)
    def close(self):
        wp.synchronize_device('cuda:0');self.frame_graph=None
        self.audit.close();self.s.close()
