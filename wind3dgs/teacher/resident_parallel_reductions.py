"""개발용 고정 순서 GPU 병렬 합산. 원본 solver의 해당 kernel 호출만 교체한다."""
from contextlib import contextmanager
import warp as wp
from .p3_shell_warp_precision_kernels import pair_add

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

@wp.kernel
def norm_terms(m:wp.array(dtype=wp.float64),elastic:wp.array(dtype=wp.float64),force:wp.array(dtype=wp.float64),
               ids:wp.array(dtype=wp.int32),u:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),out:wp.array2d(dtype=wp.float64)):
    i=wp.tid();j=ids[i]
    out[i,0]=m[i]*m[i];out[i,1]=elastic[j]*elastic[j];out[i,2]=force[j]*force[j]
    out[i,3]=u[j]*u[j];out[i,4]=rhs[i]*rhs[i]

@wp.kernel
def square_terms(a:wp.array(dtype=wp.float64),out:wp.array2d(dtype=wp.float64)):
    i=wp.tid();out[i,0]=a[i]*a[i]

@wp.kernel
def work_terms(u0:wp.array(dtype=wp.float64),u0l:wp.array(dtype=wp.float64),u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),
               force:wp.array(dtype=wp.float64),out:wp.array2d(dtype=wp.float64)):
    i=wp.tid()
    difference=pair_add(wp.vec2d(u[i],ul[i]),wp.vec2d(-u0[i],-u0l[i]))
    out[i,0]=force[i]*(difference[0]+difference[1])

@wp.kernel
def reduce_pairs(src:wp.array2d(dtype=wp.float64),dst:wp.array2d(dtype=wp.float64),n:int,kind:int):
    i,j=wp.tid();a=src[2*i,j]
    if 2*i+1<n:
        b=src[2*i+1,j]
        if kind==1 and j==2:a=wp.min(a,b)
        elif (kind==1 and j>=3) or (kind==2 and j==1):a=wp.max(a,b)
        else:a=a+b
    dst[i,j]=a

@wp.kernel
def finish_norms(a:wp.array2d(dtype=wp.float64),s:wp.array(dtype=wp.float64),atol:wp.float64,rtol:wp.float64,uatol:wp.float64,urtol:wp.float64):
    s[0]=wp.sqrt(a[0,4]);s[1]=atol+rtol*wp.sqrt(wp.max(a[0,0],wp.max(a[0,1],a[0,2])))
    s[3]=uatol+urtol*wp.sqrt(a[0,3])

@wp.kernel
def finish_linear(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),lc:wp.array(dtype=wp.int32),
                  ls:wp.array(dtype=wp.float64),a:wp.array2d(dtype=wp.float64),coef:wp.float64):
    c[8]=wp.max(c[8],lc[7]);c[9]+=lc[7];c[5]=0;s[2]=coef*wp.sqrt(a[0,0])
    if lc[8]!=0 or not wp.isfinite(s[2]) or ls[5]>ls[1]:c[7]=2;c[4]=0
    elif s[0]<=s[1] and s[2]<=s[3]:c[4]=0
    else:c[5]=1;c[6]=0;s[4]=wp.float64(1.);s[5]=s[0]

@wp.kernel
def finish_energy(a:wp.array2d(dtype=wp.float64),elastic:wp.array(dtype=wp.float64),kinetic:wp.array(dtype=wp.float64),
                  energy:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    work=a[0,0];energy[2]=work;energy[1]=(elastic[0]+wp.float64(.5)*kinetic[0])-energy[0]-work
    if not wp.isfinite(energy[1]):wp.atomic_max(failure,0,9)

@wp.kernel
def finish_volume(a:wp.array2d(dtype=wp.float64),out:wp.array(dtype=wp.float64)):
    out[0]=a[0,0]+a[0,1];out[1]=a[0,0];out[2]=a[0,1];out[3]=a[0,2];out[4]=a[0,3];out[5]=a[0,4]

@wp.kernel
def finish_edge(a:wp.array2d(dtype=wp.float64),out:wp.array(dtype=wp.float64)):
    out[0]+=a[0,0];out[5]=wp.max(out[5],a[0,1])

class ParallelReductions:
    def __init__(self,capacity,device):
        self.device=device
        self.a=wp.empty((capacity,5),dtype=wp.float64,device=device)
        self.b=wp.empty_like(self.a)
        wp.load_module(module=__name__,device=device)

    def launch(self,kernel,args,dim=1):
        wp.launch(kernel,dim=dim,inputs=args,device=self.device)

    def reduce(self,src,n,cols,kind=0):
        if n<1:raise ValueError('빈 집계는 지원하지 않습니다.')
        while n>1:
            dst=self.b if src.ptr==self.a.ptr else self.a
            count=(n+1)//2
            self.launch(reduce_pairs,[src,dst,n,kind],(count,cols))
            src=dst;n=count
        return src

    def replace(self,kernel,args):
        from . import resident_step_kernels as k
        from . import p3_shell_resident_kernels as r
        if kernel is k.norms:
            m,e,f,ids,u,rhs,s,*limits=args
            self.launch(norm_terms,[m,e,f,ids,u,rhs,self.a],len(ids))
            total=self.reduce(self.a,len(ids),5)
            self.launch(finish_norms,[total,s,*limits])
        elif kernel is k.after_linear:
            c,s,lc,ls,delta,coef=args
            self.launch(square_terms,[delta,self.a],len(delta))
            total=self.reduce(self.a,len(delta),1)
            self.launch(finish_linear,[c,s,lc,ls,total,coef])
        elif kernel is k.energy_balance:
            u0,u0l,u,ul,force,elastic,kinetic,energy,failure=args
            self.launch(work_terms,[u0,u0l,u,ul,force,self.a],len(u))
            total=self.reduce(self.a,len(u),1)
            self.launch(finish_energy,[total,elastic,kinetic,energy,failure])
        elif kernel is r.reduce_volume:
            partial,out=args
            total=self.reduce(partial,partial.shape[0],5,1)
            self.launch(finish_volume,[total,out])
        elif kernel is r.reduce_edge:
            partial,out=args
            total=self.reduce(partial,partial.shape[0],2,2)
            self.launch(finish_edge,[total,out])
        else:return False
        return True

@contextmanager
def parallel_reductions():
    """단일 solver·단일 stream의 별도 worker 전용. 초기화 때만 workspace 할당."""
    from .p3_shell_resident_stepper import ResidentShellStepper
    original_init,original_launch=ResidentShellStepper.__init__,wp.launch
    workspace=None
    def init(self,model,*args,**kwargs):
        nonlocal workspace
        workspace=ParallelReductions(max(3*len(model.rest_positions),3*len(model.triangles)),kwargs.get('device','cuda:0'))
        original_init(self,model,*args,**kwargs)
    def launch(kernel,*args,**kwargs):
        if workspace is not None and 'inputs' in kwargs and workspace.replace(kernel,kwargs['inputs']):return
        return original_launch(kernel,*args,**kwargs)
    ResidentShellStepper.__init__,wp.launch=init,launch
    try:yield
    finally:ResidentShellStepper.__init__,wp.launch=original_init,original_launch
