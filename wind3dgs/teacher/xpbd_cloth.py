"""CG 후보: P1 StVK strain block + dihedral Small Steps XPBD.

FP32·lumped mass·색상별 Gauss-Seidel. 기존 P3/Newmark의 대체 승인이 아니다.
접촉은 이산 VF/EE Jacobi 보정이며 IPC barrier/연속 CCD 보장은 제공하지 않는다.
"""
from dataclasses import dataclass
import numpy as np
import warp as wp

LAW='p1_stvk_dihedral_smallsteps_xpbd_v1'
wp.set_module_options({'enable_backward':False,'fast_math':False})


def colors(rows):
    groups=[];used=[]
    for i,row in enumerate(rows):
        nodes=set(int(v) for v in row if v>=0)
        for j,seen in enumerate(used):
            if not nodes & seen:break
        else:j=len(groups);groups.append([]);used.append(set())
        groups[j].append(i);used[j].update(nodes)
    return [np.asarray(g,dtype=np.int32) for g in groups]


@dataclass
class Cloth:
    rest:np.ndarray
    faces:np.ndarray
    pinned:np.ndarray
    density:float
    dm:np.ndarray
    bending:float

    def __post_init__(self):
        self.rest=np.asarray(self.rest,dtype=float);self.faces=np.asarray(self.faces,dtype=np.int32)
        self.pinned=np.asarray(self.pinned,dtype=bool)
        if self.rest.ndim!=2 or self.rest.shape[1]!=3 or not np.isfinite(self.rest).all():raise ValueError('rest 오류')
        if self.faces.ndim!=2 or self.faces.shape[1]!=3 or self.faces.min()<0 or self.faces.max()>=len(self.rest):raise ValueError('faces 오류')
        if self.pinned.shape!=(len(self.rest),) or self.density<=0 or self.bending<=0 or np.min(np.linalg.eigvalsh(self.dm))<=0:raise ValueError('물성/핀 오류')
        local=self.rest[self.faces];e=local[:,1]-local[:,0];f=local[:,2]-local[:,0]
        cross=np.cross(e,f);self.area=np.linalg.norm(cross,axis=1)/2
        if np.any(self.area<1e-12):raise ValueError('퇴화 rest')
        self.normal=cross/(2*self.area[:,None]);t=e/np.linalg.norm(e,axis=1)[:,None];s=np.cross(self.normal,t)
        coords=np.stack([np.zeros((len(e),2)),np.stack([np.sum(e*t,1),np.sum(e*s,1)],1),np.stack([np.sum(f*t,1),np.sum(f*s,1)],1)],1)
        inv=np.linalg.inv(np.concatenate([np.ones((len(e),3,1)),coords],axis=2))
        self.grad=inv[:,1:,:].transpose(0,2,1)
        self.mass=np.zeros(len(self.rest));np.add.at(self.mass,self.faces.ravel(),np.repeat(self.density*self.area/3,3))
        if np.any(self.mass<=0):raise ValueError('질량0 노드')
        self.inv_mass=np.where(self.pinned,0.,1/self.mass)
        self.compliance=np.linalg.inv(self.dm)[None,:,:]/self.area[:,None,None]
        records={};near=[{i} for i in range(len(self.rest))]
        for fi,face in enumerate(self.faces):
            for a,b,c in (face,face[[1,2,0]],face[[2,0,1]]):
                records.setdefault(tuple(sorted((int(a),int(b)))),[]).append((int(a),int(b),int(c),fi))
                near[a].add(int(b));near[b].add(int(a))
        self.edges=np.array(list(records),dtype=np.int32);hinges=[];stiff=[];normals=[]
        for key,items in records.items():
            if len(items)>2:raise ValueError('nonmanifold')
            a,b,c,fi=items[0];length=np.linalg.norm(self.rest[b]-self.rest[a])
            if len(items)==2:
                a1,b1,d,fj=items[1]
                if (a1,b1)!=(b,a):raise ValueError('orientation 불일치')
                hinges.append((a,b,c,d));stiff.append(3*self.bending*length**2/(self.area[fi]+self.area[fj]));normals.append(self.normal[fi])
            elif self.pinned[a] and self.pinned[b]:
                hinges.append((a,b,c,-1));stiff.append(3*self.bending*length**2/self.area[fi]);normals.append(self.normal[fi])
        self.hinges=np.asarray(hinges,dtype=np.int32).reshape(-1,4);self.bend_compliance=1/np.asarray(stiff)
        self.fixed_normals=np.asarray(normals).reshape(-1,3)
        self.tri_colors=colors(self.faces);self.bend_colors=colors(self.hinges)
        # 고정 topology 후보. 작은 테스트 전용 O(N²), 인접1-ring 제외.
        vf=[(i,*face) for i in range(len(self.rest)) for face in self.faces if not near[i].intersection(face)]
        ee=[(*e0,*e1) for i,e0 in enumerate(self.edges) for e1 in self.edges[i+1:] if not (near[e0[0]]|near[e0[1]]).intersection(e1)]
        self.vf=np.asarray(vf,dtype=np.int32).reshape(-1,4);self.ee=np.asarray(ee,dtype=np.int32).reshape(-1,4)


def membrane_data(x,grad):
    F=(x-x[:1]).T@grad
    strain=np.array([.5*(F[:,0]@F[:,0]-1),.5*(F[:,1]@F[:,1]-1),F[:,0]@F[:,1]])
    jac=np.stack([np.stack([g[0]*F[:,0],g[1]*F[:,1],g[0]*F[:,1]+g[1]*F[:,0]]) for g in grad])
    return strain,jac


def hinge_data(x,fixed=None):
    a,b,c=x[:3];e=b-a;r=c-a;l=np.linalg.norm(e);t=e/l
    n0=np.cross(e,r);l0=np.linalg.norm(n0);n0/=l0
    if fixed is None:n1=np.cross(x[3]-a,e);l1=np.linalg.norm(n1);n1/=l1
    else:n1=np.array(fixed);l1=1.
    sn=t@np.cross(n0,n1);cs=n0@n1;den=sn*sn+cs*cs
    at=cs*np.cross(n0,n1)/den;at=(at-t*(t@at))/l
    z=(cs*np.cross(n1,t)-sn*n1)/den;z=(z-n0*(z@n0))/l0
    gc=np.cross(z,e);gb=np.cross(r,z)+at;gd=np.zeros(3)
    if fixed is None:
        z1=(cs*np.cross(t,n0)-sn*n0)/den;z1=(z1-n1*(z1@n1))/l1
        gb+=np.cross(z1,x[3]-a);gd=np.cross(e,z1)
    return np.arctan2(sn,cs),np.array([-gb-gc-gd,gb,gc,gd])


@wp.kernel
def aero(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),faces:wp.array(dtype=wp.vec3i),wind:wp.array(dtype=wp.vec3),force:wp.array(dtype=wp.vec3),status:wp.array(dtype=int)):
    f=faces[wp.tid()];cross=wp.cross(x[f[1]]-x[f[0]],x[f[2]]-x[f[0]]);length=wp.length(cross)
    if length<1.e-12:wp.atomic_max(status,0,1)
    else:
        n=cross/length;vn=wp.dot(wind[0]-(v[f[0]]+v[f[1]]+v[f[2]])/3.,n);traction=.6*vn*wp.abs(vn)
        if wp.abs(traction)>10000.:wp.atomic_max(status,0,2)
        q=(traction*length/6.)*n
        for j in range(3):wp.atomic_add(force,f[j],q)


@wp.kernel
def predict(x:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),w:wp.array(dtype=float),force:wp.array(dtype=wp.vec3),gravity:wp.array(dtype=wp.vec3),h:float):
    i=wp.tid();old[i]=x[i]
    if w[i]>0.:v[i]+=h*(gravity[0]+w[i]*force[i]);x[i]+=h*v[i]
    else:x[i]=rest[i];v[i]=wp.vec3(0.)


@wp.func
def strain_jac(g:wp.vec3,F0:wp.vec3,F1:wp.vec3):
    a=g[0]*F0;b=g[1]*F1;c=g[0]*F1+g[1]*F0
    return wp.mat33(a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2])


@wp.kernel
def stretch(x:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),w:wp.array(dtype=float),faces:wp.array(dtype=wp.vec3i),grad:wp.array(dtype=wp.mat33),comp:wp.array(dtype=wp.mat33),ids:wp.array(dtype=int),h:float,tau:float,status:wp.array(dtype=int)):
    k=ids[wp.tid()];f=faces[k];g=grad[k];p=x[f[0]];q=x[f[1]];r=x[f[2]]
    F0=(q-p)*g[1,0]+(r-p)*g[2,0];F1=(q-p)*g[1,1]+(r-p)*g[2,1]
    C=wp.vec3(.5*(wp.dot(F0,F0)-1.),.5*(wp.dot(F1,F1)-1.),wp.dot(F0,F1))
    A=strain_jac(g[0],F0,F1);B=strain_jac(g[1],F0,F1);D=strain_jac(g[2],F0,F1)
    gamma=tau/h;mat=(1.+gamma)*(w[f[0]]*A*wp.transpose(A)+w[f[1]]*B*wp.transpose(B)+w[f[2]]*D*wp.transpose(D))+comp[k]/(h*h)
    rhs=-C-gamma*(A*(p-old[f[0]])+B*(q-old[f[1]])+D*(r-old[f[2]]))
    dl=wp.inverse(mat)*rhs
    x[f[0]]=p+w[f[0]]*wp.transpose(A)*dl;x[f[1]]=q+w[f[1]]*wp.transpose(B)*dl;x[f[2]]=r+w[f[2]]*wp.transpose(D)*dl
    if not wp.isfinite(dl[0]) or not wp.isfinite(dl[1]) or not wp.isfinite(dl[2]):wp.atomic_max(status,0,3)


@wp.struct
class Hinge:
    angle:float
    a:wp.vec3
    b:wp.vec3
    c:wp.vec3
    d:wp.vec3
    valid:int


@wp.func
def hinge(a:wp.vec3,b:wp.vec3,c:wp.vec3,d:wp.vec3,fixed:wp.vec3,boundary:int):
    out=Hinge();e=b-a;r=c-a;l=wp.length(e);cross0=wp.cross(e,r);l0=wp.length(cross0)
    cross1=wp.cross(d-a,e);l1=wp.length(cross1)
    if l>1.e-8 and l0>1.e-10 and (boundary!=0 or l1>1.e-10):
        t=e/l;n0=cross0/l0;n1=fixed
        if boundary==0:n1=cross1/l1
        sn=wp.dot(t,wp.cross(n0,n1));cs=wp.dot(n0,n1);den=sn*sn+cs*cs
        out.angle=wp.atan2(sn,cs);out.valid=1
        at=cs*wp.cross(n0,n1)/den;at=(at-t*wp.dot(t,at))/l
        z=(cs*wp.cross(n1,t)-sn*n1)/den;z=(z-n0*wp.dot(n0,z))/l0
        out.c=wp.cross(z,e);out.b=wp.cross(r,z)+at
        if boundary==0:
            z1=(cs*wp.cross(t,n0)-sn*n0)/den;z1=(z1-n1*wp.dot(n1,z1))/l1
            out.b+=wp.cross(z1,d-a);out.d=wp.cross(e,z1)
        out.a=-out.b-out.c-out.d
    return out


@wp.kernel
def bend(x:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),w:wp.array(dtype=float),hinges:wp.array(dtype=wp.vec4i),comp:wp.array(dtype=float),normals:wp.array(dtype=wp.vec3),ids:wp.array(dtype=int),h:float,tau:float,status:wp.array(dtype=int)):
    k=ids[wp.tid()];f=hinges[k];d=wp.max(f[3],0);boundary=int(f[3]<0)
    z=hinge(x[f[0]],x[f[1]],x[f[2]],x[d],normals[k],boundary)
    if z.valid==0:wp.atomic_max(status,0,4)
    else:
        rate=wp.dot(z.a,x[f[0]]-old[f[0]])+wp.dot(z.b,x[f[1]]-old[f[1]])+wp.dot(z.c,x[f[2]]-old[f[2]])
        mass=w[f[0]]*wp.dot(z.a,z.a)+w[f[1]]*wp.dot(z.b,z.b)+w[f[2]]*wp.dot(z.c,z.c)
        if boundary==0:rate+=wp.dot(z.d,x[d]-old[d]);mass+=w[d]*wp.dot(z.d,z.d)
        gamma=tau/h;dl=(-z.angle-gamma*rate)/((1.+gamma)*mass+comp[k]/(h*h))
        x[f[0]]+=w[f[0]]*dl*z.a;x[f[1]]+=w[f[1]]*dl*z.b;x[f[2]]+=w[f[2]]*dl*z.c
        if boundary==0:x[d]+=w[d]*dl*z.d


@wp.func
def barycentric_closest(p:wp.vec3,a:wp.vec3,b:wp.vec3,c:wp.vec3):
    ab=b-a;ac=c-a;ap=p-a;d1=wp.dot(ab,ap);d2=wp.dot(ac,ap)
    if d1<=0. and d2<=0.:return wp.vec3(1.,0.,0.)
    bp=p-b;d3=wp.dot(ab,bp);d4=wp.dot(ac,bp)
    if d3>=0. and d4<=d3:return wp.vec3(0.,1.,0.)
    vc=d1*d4-d3*d2
    if vc<=0. and d1>=0. and d3<=0.:
        v=d1/(d1-d3);return wp.vec3(1.-v,v,0.)
    cp=p-c;d5=wp.dot(ab,cp);d6=wp.dot(ac,cp)
    if d6>=0. and d5<=d6:return wp.vec3(0.,0.,1.)
    vb=d5*d2-d1*d6
    if vb<=0. and d2>=0. and d6<=0.:
        v=d2/(d2-d6);return wp.vec3(1.-v,0.,v)
    va=d3*d6-d5*d4
    if va<=0. and d4-d3>=0. and d5-d6>=0.:
        v=(d4-d3)/(d4-d3+d5-d6);return wp.vec3(0.,1.-v,v)
    denom=va+vb+vc
    if wp.abs(denom)<1.e-20:return wp.vec3(1.,0.,0.)
    return wp.vec3(1.-(vb+vc)/denom,vb/denom,vc/denom)


@wp.func
def segment_parameters(a:wp.vec3,b:wp.vec3,c:wp.vec3,d:wp.vec3):
    e=b-a;f=d-c;r=a-c;aa=wp.dot(e,e);bb=wp.dot(e,f);cc=wp.dot(f,f);dd=wp.dot(e,r);ee=wp.dot(f,r)
    s=float(0.);t=float(0.);den=aa*cc-bb*bb
    if aa>1.e-16 and cc>1.e-16:
        if den>1.e-16:s=wp.clamp((bb*ee-cc*dd)/den,0.,1.)
        t=(bb*s+ee)/cc
        if t<0.:t=0.;s=wp.clamp(-dd/aa,0.,1.)
        elif t>1.:t=1.;s=wp.clamp((bb-dd)/aa,0.,1.)
    return wp.vec2(s,t)


@wp.kernel
def contact(x:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),w:wp.array(dtype=float),pairs:wp.array(dtype=wp.vec4i),kind:int,thickness:float,delta:wp.array(dtype=wp.vec3),count:wp.array(dtype=int),hits:wp.array(dtype=int)):
    f=pairs[wp.tid()];a=x[f[0]];b=x[f[1]];c=x[f[2]];d=x[f[3]];weights=wp.vec4(0.)
    lo=wp.vec3(0.);hi=wp.vec3(0.);lo1=wp.vec3(0.);hi1=wp.vec3(0.)
    if kind==0:lo=a;hi=a;lo1=wp.min(b,wp.min(c,d));hi1=wp.max(b,wp.max(c,d))
    else:lo=wp.min(a,b);hi=wp.max(a,b);lo1=wp.min(c,d);hi1=wp.max(c,d)
    if lo[0]>hi1[0]+thickness or lo1[0]>hi[0]+thickness or lo[1]>hi1[1]+thickness or lo1[1]>hi[1]+thickness or lo[2]>hi1[2]+thickness or lo1[2]>hi[2]+thickness:return
    if kind==0:
        bar=barycentric_closest(a,b,c,d);weights=wp.vec4(1.,-bar[0],-bar[1],-bar[2])
    else:
        st=segment_parameters(a,b,c,d);weights=wp.vec4(1.-st[0],st[0],st[1]-1.,-st[1])
    diff=weights[0]*a+weights[1]*b+weights[2]*c+weights[3]*d;length=wp.length(diff)
    if length>=thickness:return
    normal=wp.vec3(0.,1.,0.)
    if length>1.e-9:normal=diff/length
    else:
        prev=wp.vec3(0.)
        for j in range(4):prev+=weights[j]*old[f[j]]
        if wp.length(prev)>1.e-9:normal=wp.normalize(prev)
    denom=float(0.)
    for j in range(4):denom+=w[f[j]]*weights[j]*weights[j]
    if denom>0.:
        dl=(thickness-length)/denom
        for j in range(4):
            if w[f[j]]>0. and wp.abs(weights[j])>1.e-8:
                wp.atomic_add(delta,f[j],w[f[j]]*weights[j]*dl*normal);wp.atomic_add(count,f[j],1)
        wp.atomic_add(hits,kind,1)


@wp.kernel
def apply_contacts(x:wp.array(dtype=wp.vec3),delta:wp.array(dtype=wp.vec3),count:wp.array(dtype=int)):
    i=wp.tid()
    if count[i]>0:x[i]+=delta[i]/float(count[i])


@wp.kernel
def velocity(x:wp.array(dtype=wp.vec3),old:wp.array(dtype=wp.vec3),v:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),w:wp.array(dtype=float),h:float,status:wp.array(dtype=int)):
    i=wp.tid()
    if w[i]==0.:x[i]=rest[i];v[i]=wp.vec3(0.)
    else:v[i]=(x[i]-old[i])/h
    for j in range(3):
        if not wp.isfinite(x[i][j]) or not wp.isfinite(v[i][j]) or wp.abs(x[i][j])>100.:wp.atomic_max(status,0,5)


class XPBD:
    def __init__(self,cloth,positions=None,velocities=None,*,substeps=32,fps=60,membrane_tau=.005,bending_tau=0.,thickness=.001,contact_sweeps=2,device='cuda:0'):
        if type(substeps) is not int or substeps<1 or fps<=0 or min(membrane_tau,bending_tau,thickness,contact_sweeps)<0:raise ValueError('XPBD 설정 오류')
        self.cloth=cloth;self.device=wp.get_device(device);self.h=1/(fps*substeps);self.substeps=substeps;self.membrane_tau=membrane_tau;self.bending_tau=bending_tau;self.thickness=thickness;self.contact_sweeps=contact_sweeps
        def arr(a,dtype):return wp.array(np.ascontiguousarray(a),dtype=dtype,device=self.device)
        self.rest=arr(cloth.rest,wp.vec3);self.x=arr(cloth.rest if positions is None else positions,wp.vec3);self.v=arr(np.zeros_like(cloth.rest) if velocities is None else velocities,wp.vec3)
        if self.x.shape!=self.rest.shape or self.v.shape!=self.rest.shape:raise ValueError('상태 shape 오류')
        self.old=wp.empty_like(self.x);self.force=wp.zeros_like(self.x);self.w=arr(cloth.inv_mass,wp.float32)
        self.faces=arr(cloth.faces,wp.vec3i);self.hinges=arr(cloth.hinges,wp.vec4i);self.normals=arr(cloth.fixed_normals,wp.vec3)
        g=np.pad(cloth.grad,((0,0),(0,0),(0,1)));self.grad=arr(g,wp.mat33);self.comp=arr(cloth.compliance,wp.mat33);self.bc=arr(cloth.bend_compliance,wp.float32)
        self.tc=[arr(a,wp.int32) for a in cloth.tri_colors];self.hc=[arr(a,wp.int32) for a in cloth.bend_colors]
        self.vf=arr(cloth.vf,wp.vec4i);self.ee=arr(cloth.ee,wp.vec4i)
        self.delta=wp.zeros_like(self.x);self.count=wp.zeros(len(cloth.rest),dtype=int,device=self.device);self.hits=wp.zeros(2,dtype=int,device=self.device);self.status=wp.zeros(1,dtype=int,device=self.device)
        self.wind=wp.zeros(1,dtype=wp.vec3,device=self.device);self.gravity=wp.zeros_like(self.wind)
        wp.load_module(module=__name__,device=self.device)
        self.graph=None
        if self.device.is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:self._frame()
            self.graph=capture.graph

    def launch(self,fn,args,n):
        if n:wp.launch(fn,dim=n,inputs=args,device=self.device)

    def _contacts(self):
        self.delta.zero_();self.count.zero_()
        self.launch(contact,[self.x,self.old,self.w,self.vf,0,self.thickness,self.delta,self.count,self.hits],len(self.vf))
        self.launch(contact,[self.x,self.old,self.w,self.ee,1,self.thickness,self.delta,self.count,self.hits],len(self.ee))
        self.launch(apply_contacts,[self.x,self.delta,self.count],len(self.x))

    def _frame(self):
        self.status.zero_();self.hits.zero_();self.force.zero_()
        self.launch(aero,[self.x,self.v,self.faces,self.wind,self.force,self.status],len(self.faces))
        for _ in range(self.substeps):
            self.launch(predict,[self.x,self.v,self.old,self.rest,self.w,self.force,self.gravity,self.h],len(self.x))
            for c in self.tc:self.launch(stretch,[self.x,self.old,self.w,self.faces,self.grad,self.comp,c,self.h,self.membrane_tau,self.status],len(c))
            for c in self.hc:self.launch(bend,[self.x,self.old,self.w,self.hinges,self.bc,self.normals,c,self.h,self.bending_tau,self.status],len(c))
            for _ in range(self.contact_sweeps):self._contacts()
            self.launch(velocity,[self.x,self.old,self.v,self.rest,self.w,self.h,self.status],len(self.x))

    def step(self,wind=(0,0,0),gravity=(0,0,-9.81)):
        self.wind.assign(np.asarray([wind],dtype=np.float32));self.gravity.assign(np.asarray([gravity],dtype=np.float32))
        if self.graph is None:self._frame()
        else:wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        return self.x.numpy(),self.v.numpy(),int(self.status.numpy()[0]),self.hits.numpy()
