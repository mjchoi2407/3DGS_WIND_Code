"""왼쪽 0.25m 고정의 P3 유한 회전 Koiter/StVK shell. Training-only CPU reference 후보."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix, csr_matrix

from wind3dgs.evaluation import teacher_plate_cubic as cubic
from wind3dgs.teacher.p3_surface import P3SurfaceElement, _array, _positive
from wind3dgs.teacher.p3_wind_reset import rectangular_mesh
from wind3dgs.teacher.shell_structure import ShellElasticMaterial
from . import p3_shell_kernels as kernels


LAW = 'flat_p3_koiter_stvk_normal_jump_v1'


@dataclass
class _Batch:
    ids: np.ndarray
    N: np.ndarray
    G: np.ndarray
    H: np.ndarray
    weights: np.ndarray

    def geometry(self, x):
        local = x[self.ids]; local = local-local[:, :1]
        return np.einsum('eqia,eic->eqac', self.G, local), np.einsum('eqik,eic->eqkc', self.H, local)

    def adjoint(self, A, C):
        return (np.einsum('eq,eqia,eqac->eic', self.weights, self.G, A)
                +np.einsum('eq,eqik,eqkc->eic', self.weights, self.H, C))


def _topology(xy, tri):
    nodes = list(xy.copy()); edges = {}; dofs = []
    for face_index, face in enumerate(tri):
        local = list(face)
        for i,j in ((0,1),(1,2),(2,0)):
            key = tuple(sorted((int(face[i]),int(face[j]))))
            if key not in edges:
                edges[key] = {'ids': [len(nodes),len(nodes)+1], 'sides': []}
                nodes.extend(((2*xy[key[0]]+xy[key[1]])/3, (xy[key[0]]+2*xy[key[1]])/3))
            row = edges[key]; row['sides'].append((face_index,i,j))
            local.extend(row['ids'] if face[i] == key[0] else row['ids'][::-1])
        local.append(len(nodes)); nodes.append(xy[face].mean(axis=0)); dofs.append(local)
    return np.array(nodes), np.array(dofs,dtype=np.int64), edges


def _hessian(H):
    return np.stack((H[...,0,0],H[...,1,1],2*H[...,0,1]),axis=-1)


class P3Shell:
    """3D position/velocity DOF, full consistent scalar M와 fixed normal 경계의 에너지 모델."""
    def __init__(self, resolution=4, *, diagonal='forward', quadrature_order=6, edge_order=4,
                 material=None, area_density_kg_m2=.1, clamp=True, rotation=None, sample_mesh=None):
        self.material = material or ShellElasticMaterial(1e6,.3,.01)
        if type(self.material) is not ShellElasticMaterial or type(clamp) is not bool:
            raise ValueError('ShellElasticMaterial과 bool clamp가 필요합니다')
        if type(edge_order) is not int or not 3 <= edge_order <= 12:
            raise ValueError('Edge 구적 차수는 3~12 정수여야 합니다')
        self.density = _positive(area_density_kg_m2, '면밀도')
        R = np.eye(3) if rotation is None else _array(rotation,(3,3),'proper rotation')
        if not np.allclose(R.T@R,np.eye(3),atol=1e-12,rtol=0) or np.linalg.det(R) <= 0:
            raise ValueError('proper rotation이 필요합니다')
        self.resolution, self.diagonal = resolution, diagonal
        self.quadrature_order, self.edge_order = quadrature_order, edge_order
        self.sample_mesh_kind = None
        if sample_mesh is None:
            xy, tri = rectangular_mesh(resolution, diagonal)
        else:
            from .sample_meshes import validate_sample_mesh
            validate_sample_mesh(sample_mesh)
            if not np.all(sample_mesh.vertices[:,1] == 0):
                raise ValueError('평평한 XZ 샘플만 지원합니다')
            xy=np.column_stack((sample_mesh.vertices[:,0].astype(float), .5-sample_mesh.vertices[:,2].astype(float)))
            tri=sample_mesh.faces.astype(np.int64,copy=True)
            self.sample_mesh_kind=sample_mesh.kind.value
        self.xy, self.dofs, edges = _topology(xy,tri)
        self.vertex_xy, self.triangles = xy, tri
        self.rest_positions = np.column_stack((self.xy[:,0],np.zeros(len(self.xy)),.5-self.xy[:,1]))@R.T
        self.rest_normal = R@np.array([0.,1.,0.])
        self.rest_tangents = np.array([[1.,0.,0.],[0.,0.,-1.]])@R.T
        self.free = self.xy[:,0] > .25 if clamp else np.ones(len(self.xy),dtype=bool)
        if sample_mesh is not None and clamp:
            pinned=np.zeros(len(self.xy),dtype=bool)
            pinned[:len(xy)]=sample_mesh.pinned
            for key,row in edges.items():
                pinned[row['ids']]=bool(np.all(sample_mesh.pinned[list(key)]))
            for face,ids in zip(tri,self.dofs):
                pinned[ids[-1]]=bool(np.all(sample_mesh.pinned[face]))
            self.free=~pinned
        self.clamp = clamp
        elements = [P3SurfaceElement.from_triangle(xy[face],quadrature_order=quadrature_order) for face in tri]
        self.volume = _Batch(self.dofs,np.stack([e.shape for e in elements]),
            np.stack([e.gradient_inv_m for e in elements]),np.stack([_hessian(e.hessian_inv_m2) for e in elements]),
            np.stack([e.weights_m2 for e in elements]))
        local_m = np.stack([e.consistent_mass(self.density) for e in elements])
        row = np.broadcast_to(self.dofs[:,:,None],local_m.shape).ravel()
        col = np.broadcast_to(self.dofs[:,None,:],local_m.shape).ravel()
        self.mass = coo_matrix((local_m.ravel(),(row,col)),shape=(len(self.xy),len(self.xy))).tocsr()
        self.dm,self.db = self.material.matrices()
        z,w = leggauss(edge_order); z=(z+1)/2
        self.edge_groups = []
        for boundary in (False,True):
            records = [(key,r) for key,r in edges.items() if (len(r['sides']) == 1) == boundary
                       and (not boundary or clamp and (np.all(xy[list(key),0] == .25) if sample_mesh is None else np.all(sample_mesh.pinned[list(key)])))]
            if not records: continue
            batches = []; mus=[]; penalties=[]
            for side in range(1 if boundary else 2):
                ids_list=[]; Ns=[]; Gs=[]; Hs=[]; weights=[]
                for key,record in records:
                    fi,i,j = record['sides'][side]
                    face = tri[fi]; points = xy[face]; origin=points[0]
                    inv = np.linalg.inv(np.column_stack((np.ones(3),points-origin)))
                    samples = (1-z[:,None])*xy[key[0]]+z[:,None]*xy[key[1]]
                    bary = np.column_stack((np.ones(edge_order),samples-origin))@inv
                    N = cubic.shape_values(bary); G,H = cubic._derivatives(bary,inv)
                    N[:,0] += 1-N.sum(axis=1); G[:,0] -= G.sum(axis=1); H[:,0] -= H.sum(axis=1)
                    length = np.linalg.norm(xy[key[1]]-xy[key[0]])
                    if side == 0:
                        tangent = points[j]-points[i]
                        mus.append(np.array([tangent[1],-tangent[0]])/length)
                        hs=[max(np.linalg.norm(xy[tri[f,a]]-xy[tri[f,b]]) for a,b in ((0,1),(1,2),(2,0)))
                            for f,_,_ in record['sides']]
                        penalties.append(cubic.PENALTY_FACTOR*self.material.young_modulus_pa
                                         *self.material.thickness_m**3/np.mean(hs))
                    ids_list.append(self.dofs[fi]); Ns.append(N); Gs.append(G); Hs.append(_hessian(H)); weights.append(w*length/2)
                batches.append(_Batch(np.array(ids_list),np.array(Ns),np.array(Gs),np.array(Hs),np.array(weights)))
            self.edge_groups.append((batches,np.array(mus)[:,None,:],np.array(penalties)[:,None],boundary))

    def evaluate(self, positions, *, direction=None):
        x = _array(positions,self.rest_positions.shape,'shell position')
        return self.evaluate_displacement(x-self.rest_positions,direction=direction)

    def evaluate_displacement(self, displacement, *, direction=None):
        """작은 dt에서도 rest 좌표 차분을 반복하지 않는 solver용 상태 표현."""
        u = _array(displacement,self.rest_positions.shape,'shell displacement')
        d = None if direction is None else _array(direction,u.shape,'shell direction')
        V=self.volume; F,H=V.geometry(u); F+=self.rest_tangents
        result=kernels.volume(F,H,self.dm,self.db,direction=None if d is None else V.geometry(d))
        grad=np.zeros_like(u); hvp=np.zeros_like(u)
        np.add.at(grad,V.ids,V.adjoint(result['gradient_F'],result['gradient_H']))
        if d is not None: np.add.at(hvp,V.ids,V.adjoint(result['tangent_F'],result['tangent_H']))
        energies={key:float(np.sum(V.weights*result[key])) for key in ('membrane_energy','bending_energy')}
        edge_energy=0.; max_jump=0.; boundary_jump=0.; frame_torque=np.zeros(3)
        for batches,mu,penalty,boundary in self.edge_groups:
            geometry=[b.geometry(u) for b in batches]
            for F,H in geometry: F+=self.rest_tangents
            dirs=None if d is None else [b.geometry(d) for b in batches]
            edge=kernels.edge([g[0] for g in geometry],[g[1] for g in geometry],mu,penalty,self.db,
                directions=dirs,fixed_normal=self.rest_normal if boundary else None)
            edge_energy += float(np.sum(batches[0].weights*edge['energy']))
            jump=float(np.max(np.linalg.norm(edge['normal_jump'],axis=-1)))
            if boundary:
                boundary_jump=max(boundary_jump,jump)
                frame_torque+=np.sum(batches[0].weights[...,None]
                    *np.cross(self.rest_normal,edge['fixed_normal_gradient']),axis=(0,1))
            else: max_jump=max(max_jump,jump)
            for index,b in enumerate(batches):
                np.add.at(grad,b.ids,b.adjoint(*edge['gradients'][index]))
                if d is not None: np.add.at(hvp,b.ids,b.adjoint(*edge['tangents'][index]))
        energy=sum(energies.values())+edge_energy
        if not np.isfinite(energy) or not np.isfinite(grad).all() or not np.isfinite(hvp).all():
            raise ValueError('shell 조립 에너지/미분의 유한 범위 이탈')
        return {'energy_j':energy,'membrane_energy_j':energies['membrane_energy'],
                'bending_volume_energy_j':energies['bending_energy'],'edge_energy_j':edge_energy,
                'force_n':-grad,'hvp_n':hvp,'max_normal_jump':max_jump,'max_boundary_normal_jump':boundary_jump,
                'fixed_normal_torque_on_shell_n_m':frame_torque,
                'min_area_ratio':float(result['area_ratio'].min()),'max_strain_component':float(abs(result['strain']).max())}

    def aerodynamic_force(self, positions, velocities, wind, *, kappa=.6, guard=10000., active=True):
        x=_array(positions,self.rest_positions.shape,'shell position')
        return self.aerodynamic_force_displacement(x-self.rest_positions,velocities,wind,
                                                   kappa=kappa,guard=guard,active=active)

    def aerodynamic_force_displacement(self, displacement, velocities, wind, *, kappa=.6, guard=10000., active=True):
        u=_array(displacement,self.rest_positions.shape,'shell displacement')
        v=_array(velocities,u.shape,'shell velocity'); wind=_array(wind,(3,),'wind')
        kappa=_positive(kappa,'kappa',zero=True); guard=_positive(guard,'guard')
        if type(active) is not bool: raise ValueError('공력 활성 상태는 bool이어야 합니다')
        F,_=self.volume.geometry(u); F+=self.rest_tangents
        c=np.cross(F[:,:,0],F[:,:,1]); J=np.linalg.norm(c,axis=-1)
        if not np.isfinite(J).all() or np.any(J <= 1e-8): raise ValueError('현재 면적 비율 실패')
        n=c/J[...,None]; qv=np.einsum('eqi,eic->eqc',self.volume.N,v[self.volume.ids])
        vn=np.sum((wind-qv)*n,axis=-1)
        traction=kappa*(vn*abs(vn))[...,None]*n if active else np.zeros_like(n)
        fixed_vn=float(wind@self.rest_normal)
        fixed_tau=kappa*fixed_vn*abs(fixed_vn)*self.rest_normal if active and self.clamp else np.zeros(3)
        if (not np.isfinite(traction).all() or np.any(np.linalg.norm(traction,axis=-1)>guard)
                or not np.isfinite(fixed_tau).all() or np.linalg.norm(fixed_tau)>guard):
            raise ValueError('면적 곱 전 traction guard 발생')
        qf=traction*(self.volume.weights*J)[...,None]
        local=np.einsum('eqi,eqc->eic',self.volume.N,qf); force=np.zeros_like(u)
        np.add.at(force,self.volume.ids,local)
        return {'force_n':force,'total_force_n':force.sum(axis=0)+.25*fixed_tau,
                'power_w':float(np.sum(qf*qv)),'fixed_region_force_n':.25*fixed_tau,'guard_activations':0}

    def rest_stiffness(self):
        """선형 극한의 sparse K. 비선형 solver의 preconditioner 및 독립 극한 검사에 쓴다."""
        rows=[]; cols=[]; data=[]
        def add(ids,values):
            rows.extend(np.broadcast_to(ids[:,:,None],values.shape).ravel())
            cols.extend(np.broadcast_to(ids[:,None,:],values.shape).ravel()); data.extend(values.ravel())
        V=self.volume; t=self.rest_tangents; normal=self.rest_normal
        B=np.stack((V.G[...,0,None]*t[0],V.G[...,1,None]*t[1],
                    V.G[...,1,None]*t[0]+V.G[...,0,None]*t[1]),axis=2).reshape(len(V.ids),len(V.weights[0]),3,30)
        C=(V.H[...,None]*normal).transpose(0,1,3,2,4).reshape(len(V.ids),len(V.weights[0]),3,30)
        values=np.einsum('eq,eqai,ab,eqbj->eij',V.weights,B,self.dm,B)
        values+=np.einsum('eq,eqai,ab,eqbj->eij',V.weights,C,self.db,C)
        add((3*V.ids[...,None]+np.arange(3)).reshape(-1,30),values)
        for batches,mu,penalty,boundary in self.edge_groups:
            count=len(batches); J=[]; moments=[]; ids=[]
            if boundary:
                b=batches[0]
                coefficients=np.stack((mu[...,0,None]*self.db[0]+mu[...,1,None]*self.db[2],
                                       mu[...,0,None]*self.db[2]+mu[...,1,None]*self.db[1]),axis=-2)
                flux=np.einsum('eqak,eqik->eqia',coefficients,b.H)
                scalar=(-np.einsum('eq,eqia,eqja->eij',b.weights,b.G,flux)
                        -np.einsum('eq,eqia,eqja->eij',b.weights,flux,b.G)
                        +np.einsum('eq,eqia,eqja->eij',b.weights*penalty,b.G,b.G))
                values=np.einsum('eij,a,b->eiajb',scalar,normal,normal).reshape(len(b.ids),30,30)
                add((3*b.ids[...,None]+np.arange(3)).reshape(len(b.ids),30),values)
                continue
            for side,b in enumerate(batches):
                J.append(np.sum(b.G*mu[:,:,None,:],axis=-1)*(1 if side==0 else -1))
                coefficient=np.stack((mu[...,0]**2,mu[...,1]**2,2*mu[...,0]*mu[...,1]),axis=-1)@self.db
                moments.append(np.einsum('eqa,eqia->eqi',coefficient,b.H)/count)
                ids.append(b.ids)
            J=np.concatenate(J,axis=-1); moment=np.concatenate(moments,axis=-1)
            weights=batches[0].weights
            scalar=(-np.einsum('eq,eqi,eqj->eij',weights,J,moment)
                    -np.einsum('eq,eqi,eqj->eij',weights,moment,J)
                    +np.einsum('eq,eqi,eqj->eij',weights*penalty,J,J))
            ids=np.concatenate(ids,axis=1)
            values=np.einsum('eij,a,b->eiajb',scalar,normal,normal).reshape(len(ids),count*30,count*30)
            add((3*ids[...,None]+np.arange(3)).reshape(len(ids),-1),values)
        size=self.rest_positions.size
        return coo_matrix((data,(rows,cols)),shape=(size,size)).tocsr()

    def moving_surface_map(self, xy):
        """자유 사각형의 signed P3 평가. 고정 경계 DOF도 포함하는 full shape map이다."""
        if self.sample_mesh_kind is not None:
            raise ValueError('샘플 메시의 공간 비교 map은 아직 연결하지 않았습니다')
        xy=np.asarray(xy,dtype=float)
        if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
                or np.any(xy[:,0] < .25) or np.any(xy[:,0] > 1)
                or np.any(xy[:,1] < 0) or np.any(xy[:,1] > 1)):
            raise ValueError('[.25,1]×[0,1] material 좌표가 필요합니다')
        n=self.resolution
        cx=np.minimum(((xy[:,0]-.25)*n).astype(int),3*n//4-1)
        cy=np.minimum((xy[:,1]*n).astype(int),n-1)
        sx=(xy[:,0]-.25)*n-cx; sy=xy[:,1]*n-cy
        side=sy>sx if self.diagonal == 'forward' else sx+sy>1
        faces=2*(cy*(3*n//4)+cx)+side.astype(int)
        points=self.vertex_xy[self.triangles[faces]]
        centered=points-points[:,:1]
        inverse=np.linalg.inv(np.concatenate((np.ones((len(xy),3,1)),centered),axis=-1))
        bary=np.einsum('pi,pij->pj',np.column_stack((np.ones(len(xy)),xy-points[:,0])),inverse)
        if np.any(bary < -1e-12): raise ValueError('P3 요소 support 밖의 좌표입니다')
        N=cubic.shape_values(bary); N[:,0]+=1-N.sum(axis=1)
        return csr_matrix((N.ravel(),(np.repeat(np.arange(len(xy)),10),self.dofs[faces].ravel())),
                          shape=(len(xy),len(self.xy)))
