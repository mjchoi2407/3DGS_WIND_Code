"""P3 shell의 Warp float64 연산자와 기존 CPU Newmark의 연결.

구조·공력 point 연산, element 적분과 순서 고정 nodal gather는 Warp device에서 수행한다.
총에너지의 마지막 합산·진단과 consistent-mass 희소 풀이/Newton 제어는 CPU에 유지한다.
한 instance의 작업 buffer는 동기 호출용이며 동시에 여러 thread에서 공유하지 않는다.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import warp as wp

from .p3_shell import P3Shell
from .p3_shell_dynamics import P3ShellStepper
from .p3_surface import _array,_positive
from . import p3_shell_warp_kernels as kernels


BACKEND='warp_float64_p3_shell_deterministic_gather_v1'


class _Batch:
    def __init__(self,host,model,device):
        self.host=host;self.elements,self.points=host.weights.shape
        def array(value,dtype):return wp.array(np.ascontiguousarray(value),dtype=dtype,device=device)
        self.ids=array(host.ids.astype(np.int32),wp.int32)
        self.N=array(host.N,wp.float64);self.G=array(host.G,wp.float64);self.H=array(host.H,wp.float64)
        self.weight=array(host.weights,wp.float64)
        self.A=wp.zeros((self.elements,self.points,2),dtype=wp.vec3d,device=device)
        self.C=wp.zeros((self.elements,self.points,3),dtype=wp.vec3d,device=device)
        self.dA=wp.zeros_like(self.A);self.dC=wp.zeros_like(self.C)
        self.local=wp.zeros(self.elements*10,dtype=wp.vec3d,device=device);self.dlocal=wp.zeros_like(self.local)
        flat=host.ids.ravel();order=np.argsort(flat,kind='stable').astype(np.int32)
        counts=np.bincount(flat,minlength=len(model.xy))
        self.row=array(np.r_[0,np.cumsum(counts)].astype(np.int32),wp.int32)
        self.columns=array(order,wp.int32)

    def geometry_inputs(self):return [self.ids,self.G,self.H]
    def outputs(self):return [self.A,self.C,self.dA,self.dC]

    def assemble(self,force,hvp,device):
        wp.launch(kernels.assemble_batch,dim=(self.elements,10),inputs=[self.G,self.H,self.weight,self.points,
            *self.outputs(),self.local,self.dlocal],device=device)
        self.gather(force,hvp,device)

    def gather(self,force,hvp,device):
        wp.launch(kernels.gather_batch,dim=len(force),inputs=[self.row,self.columns,self.local,self.dlocal,force,hvp],device=device)


class P3ShellWarp:
    """P3Shell의 동일한 물리식. 명시적으로 선택한 cpu/cuda device에서 float64로 평가한다."""
    def __init__(self,reference,*,device='cuda:0'):
        if type(reference) is not P3Shell:raise ValueError('동일 topology/material의 P3Shell 기준이 필요합니다')
        self.reference=reference
        cache=os.environ.get('WARP_CACHE_PATH',str(Path(__file__).resolve().parents[2]/'outputs/warp-cache'))
        wp.config.kernel_cache_dir=cache
        self.device=wp.get_device(device)
        if str(device).startswith('cuda') and not self.device.is_cuda:
            raise ValueError('요청한 CUDA device로 실행할 수 없습니다')
        self._volume=_Batch(reference.volume,reference,self.device)
        self.edges=[]
        for host,mu,penalty,boundary in reference.edge_groups:
            batches=[_Batch(b,reference,self.device) for b in host];e,q=host[0].weights.shape
            self.edges.append({'batches':batches,'boundary':boundary,
                'mu':wp.array(np.ascontiguousarray(mu[:,0,:]),dtype=wp.vec2d,device=self.device),
                'penalty':wp.array(np.ascontiguousarray(penalty[:,0]),dtype=wp.float64,device=self.device),
                'diagnostic':wp.zeros((e,q,5),dtype=wp.float64,device=self.device),
                'valid':wp.zeros((e,q),dtype=wp.int32,device=self.device)})
        self._u=wp.zeros(len(reference.xy),dtype=wp.vec3d,device=self.device)
        self._d=wp.zeros_like(self._u);self._force=wp.zeros_like(self._u);self._hvp=wp.zeros_like(self._u)
        shape=reference.volume.weights.shape
        self._diagnostic=wp.zeros((*shape,4),dtype=wp.float64,device=self.device)
        self._valid=wp.zeros(shape,dtype=wp.int32,device=self.device)
        self._qforce=wp.zeros(shape,dtype=wp.vec3d,device=self.device)
        self._power=wp.zeros(shape,dtype=wp.float64,device=self.device)
        self._t0=wp.vec3d(*reference.rest_tangents[0]);self._t1=wp.vec3d(*reference.rest_tangents[1])
        self._normal=wp.vec3d(*reference.rest_normal)
        self._dm=wp.mat33d(*reference.dm.ravel());self._db=wp.mat33d(*reference.db.ravel())

    def __getattr__(self,name):
        # Geometry/map/mass/policy의 authority는 CPU 기준 모델에 그대로 둔다.
        return getattr(self.reference,name)

    def evaluate(self,positions,*,direction=None):
        x=_array(positions,self.rest_positions.shape,'shell position')
        return self.evaluate_displacement(x-self.rest_positions,direction=direction)

    def evaluate_displacement(self,displacement,*,direction=None):
        u=_array(displacement,self.rest_positions.shape,'shell displacement')
        d=np.zeros_like(u) if direction is None else _array(direction,u.shape,'shell direction')
        return self._evaluate_displacement_arrays(u,d)

    def _evaluate_displacement_arrays(self,u,d,geometry_kernels=kernels):
        self._u.assign(u);self._d.assign(d);self._force.zero_();self._hvp.zero_()
        b=self._volume
        wp.launch(geometry_kernels.volume_kernel,dim=b.host.weights.shape,inputs=[self._u,self._d,*b.geometry_inputs(),
            self._t0,self._t1,self._dm,self._db,*b.outputs(),self._diagnostic,self._valid],device=self.device)
        b.assemble(self._force,self._hvp,self.device)
        for item in self.edges:
            a=item['batches'][0];other=item['batches'][-1]
            wp.launch(geometry_kernels.edge_kernel,dim=a.host.weights.shape,inputs=[self._u,self._d,
                *a.geometry_inputs(),*other.geometry_inputs(),self._t0,self._t1,self._normal,item['mu'],
                item['penalty'],int(item['boundary']),self._db,*a.outputs(),*other.outputs(),
                item['diagnostic'],item['valid']],device=self.device)
            for side in item['batches']:side.assemble(self._force,self._hvp,self.device)
        wp.synchronize_device(self.device)
        if not self._valid.numpy().all() or any(not item['valid'].numpy().all() for item in self.edges):
            raise ValueError('shell Warp 구적점의 현재 면적 비율이 유효하지 않습니다')
        diag=self._diagnostic.numpy();weights=b.host.weights
        membrane=float(np.sum(weights*diag[:,:,0]));bending=float(np.sum(weights*diag[:,:,1]))
        edge_energy=0.;jump=0.;boundary_jump=0.;torque=np.zeros(3)
        for item in self.edges:
            values=item['diagnostic'].numpy();weight=item['batches'][0].host.weights
            edge_energy+=float(np.sum(weight*values[:,:,0]));current=float(values[:,:,1].max())
            if item['boundary']:
                boundary_jump=max(boundary_jump,current);torque+=np.sum(weight[...,None]*values[:,:,2:],axis=(0,1))
            else:jump=max(jump,current)
        force=np.array(self._force.numpy(),copy=True);hvp=np.array(self._hvp.numpy(),copy=True)
        result={'energy_j':membrane+bending+edge_energy,'membrane_energy_j':membrane,
                'bending_volume_energy_j':bending,'edge_energy_j':edge_energy,
                'force_n':force,'hvp_n':hvp,'max_normal_jump':jump,'max_boundary_normal_jump':boundary_jump,
                'fixed_normal_torque_on_shell_n_m':torque,'min_area_ratio':float(diag[:,:,2].min()),
                'max_strain_component':float(diag[:,:,3].max())}
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError('shell Warp 에너지/미분의 유한 범위 이탈')
        return result

    def aerodynamic_force(self,positions,velocities,wind,**kwargs):
        x=_array(positions,self.rest_positions.shape,'shell position')
        return self.aerodynamic_force_displacement(x-self.rest_positions,velocities,wind,**kwargs)

    def aerodynamic_force_displacement(self,displacement,velocities,wind,*,kappa=.6,guard=10000.,active=True):
        u=_array(displacement,self.rest_positions.shape,'shell displacement')
        v=_array(velocities,u.shape,'shell velocity');wind=_array(wind,(3,),'wind')
        kappa=_positive(kappa,'kappa',zero=True);guard=_positive(guard,'guard')
        if type(active) is not bool:raise ValueError('공력 활성 상태는 bool이어야 합니다')
        fixed_vn=float(wind@self.rest_normal)
        fixed_tau=kappa*fixed_vn*abs(fixed_vn)*self.rest_normal if active and self.clamp else np.zeros(3)
        if not np.isfinite(fixed_tau).all() or np.linalg.norm(fixed_tau)>guard:
            raise ValueError('면적 곱 전 traction guard 발생')
        self._u.assign(u);self._d.assign(v);self._force.zero_();self._hvp.zero_();b=self._volume
        wp.launch(kernels.aero_kernel,dim=b.host.weights.shape,inputs=[self._u,self._d,b.ids,b.N,b.G,b.H,
            b.weight,self._t0,self._t1,wp.vec3d(*wind),wp.float64(kappa),wp.float64(guard),int(active),
            self._qforce,self._power,self._valid],device=self.device)
        wp.launch(kernels.assemble_aero,dim=(b.elements,10),inputs=[b.N,self._qforce,b.points,b.local,b.dlocal],device=self.device)
        b.gather(self._force,self._hvp,self.device);wp.synchronize_device(self.device)
        if not self._valid.numpy().all():raise ValueError('shell Warp 공력 구적/traction guard 실패')
        force=np.array(self._force.numpy(),copy=True);power=float(self._power.numpy().sum())
        if not np.isfinite(force).all() or not np.isfinite(power):raise ValueError('shell Warp 공력의 유한 범위 이탈')
        return {'force_n':force,'total_force_n':force.sum(axis=0)+.25*fixed_tau,
                'power_w':power,'fixed_region_force_n':.25*fixed_tau,'guard_activations':0}


class P3ShellWarpStepper(P3ShellStepper):
    """변경하지 않은 Newmark 식·허용오차/CPU sparse solver에 Warp 연산자를 연결한다."""
    def __init__(self,model,*,policy=None):
        if type(model) is not P3ShellWarp:raise ValueError('P3ShellWarp 모델이 필요합니다')
        super().__init__(model.reference,policy=policy)
        self.model=model
