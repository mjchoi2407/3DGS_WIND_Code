"""GPU Newmark/Newton 상태·잔차·분기. 계산 결과는 device 배열에만 남긴다."""
import warp as wp
from .p3_shell_warp_precision_kernels import pair_add,pair_scale
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def pack_sum(a:wp.array(dtype=wp.float64),b:wp.array(dtype=wp.float64),ids:wp.array(dtype=wp.int32),out:wp.array(dtype=wp.float64)):
    i=wp.tid();out[i]=a[ids[i]]+b[ids[i]]

@wp.kernel
def scatter(a:wp.array(dtype=wp.float64),ids:wp.array(dtype=wp.int32),out:wp.array(dtype=wp.float64)):
    i=wp.tid();out[ids[i]]=a[i]

@wp.kernel
def tangent_result(m:wp.array(dtype=wp.float64),h:wp.array(dtype=wp.float64),ids:wp.array(dtype=wp.int32),
                   y:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64),c:wp.float64,alpha:wp.float64,beta:wp.float64):
    i=wp.tid();value=alpha*(m[i]+c*h[ids[i]])
    if beta!=wp.float64(0.):value+=beta*y[i]
    out[i]=value

@wp.kernel
def fail_from_status(status:wp.array(dtype=wp.int32),failure:wp.array(dtype=wp.int32),code:int):
    if status[0]!=0:wp.atomic_max(failure,0,code)

@wp.kernel
def predict(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),
            a0:wp.array(dtype=wp.float64),ids:wp.array(dtype=wp.int32),dt:wp.float64,uh:wp.array(dtype=wp.float64),lo:wp.array(dtype=wp.float64)):
    i=wp.tid();j=ids[i]
    pos=wp.vec2d(u[j],ul[j]);velocity=pair_scale(wp.vec2d(v[j],vl[j]),dt)
    accel=pair_scale(wp.vec2d(a0[i],wp.float64(0.)),wp.float64(.5)*dt*dt)
    value=pair_add(pair_add(pos,velocity),accel);uh[j]=value[0];lo[j]=value[1]

@wp.kernel
def residual(m:wp.array(dtype=wp.float64),elastic:wp.array(dtype=wp.float64),force:wp.array(dtype=wp.float64),
             ids:wp.array(dtype=wp.int32),rhs:wp.array(dtype=wp.float64)):
    i=wp.tid();rhs[i]=elastic[ids[i]]+force[ids[i]]-m[i]

@wp.kernel
def norms(m:wp.array(dtype=wp.float64),elastic:wp.array(dtype=wp.float64),force:wp.array(dtype=wp.float64),
          ids:wp.array(dtype=wp.int32),u:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),s:wp.array(dtype=wp.float64),
          atol:wp.float64,rtol:wp.float64,uatol:wp.float64,urtol:wp.float64):
    nm=wp.float64(0.);ne=wp.float64(0.);nf=wp.float64(0.);nu=wp.float64(0.);nr=wp.float64(0.)
    for i in range(ids.shape[0]):
        j=ids[i];nm+=m[i]*m[i];ne+=elastic[j]*elastic[j];nf+=force[j]*force[j];nu+=u[j]*u[j];nr+=rhs[i]*rhs[i]
    s[0]=wp.sqrt(nr);s[1]=atol+rtol*wp.sqrt(wp.max(nm,wp.max(ne,nf)));s[3]=uatol+urtol*wp.sqrt(nu)

@wp.kernel
def begin_frame(c:wp.array(dtype=wp.int32)):
    c[1]=0;c[2]=0;c[14]=0

@wp.kernel
def begin_step(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    c[10]=0;c[11]=0;c[7]=0
    if failure[0]==0:c[11]=1

@wp.kernel
def begin_attempt(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64)):
    c[0]=0;c[7]=0;c[8]=0;c[4]=1;s[2]=wp.float64(0.);s[7]=wp.float64(0.);s[8]=wp.float64(0.)

@wp.kernel
def decide_newton(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32),maximum:int):
    c[4]=1
    if not wp.isfinite(s[0]) or failure[0]!=0:c[7]=4;c[4]=0
    elif s[0]<=s[1] and s[2]<=s[3]:c[4]=0
    elif c[0]>=maximum:c[4]=0;c[7]=1

@wp.kernel
def select_preconditioner(c:wp.array(dtype=wp.int32),rebuild_every:int):
    c[3]=0
    if c[1]!=0:
        if c[2]%rebuild_every==0:c[3]=1
        c[2]+=1

@wp.kernel
def built(c:wp.array(dtype=wp.int32)):
    c[15]+=1

@wp.kernel
def ew_tolerance(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),tol:wp.array(dtype=wp.float64),floor:wp.float64,cap:wp.float64):
    eta=cap
    if c[0]>0 and s[7]>wp.float64(0.):
        eta=wp.float64(.9)*wp.pow(s[0]/wp.max(s[7],wp.float64(1e-300)),wp.float64(1.5))
        safeguard=wp.float64(.9)*wp.pow(s[8],wp.float64(1.5))
        if safeguard>wp.float64(.1):eta=wp.max(eta,safeguard)
    eta=wp.max(floor,wp.min(cap,eta));tol[0]=eta;s[7]=s[0];s[8]=eta

@wp.kernel
def after_linear(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),lc:wp.array(dtype=wp.int32),
                 ls:wp.array(dtype=wp.float64),delta:wp.array(dtype=wp.float64),coef:wp.float64):
    c[8]=wp.max(c[8],lc[7]);c[9]+=lc[7];c[5]=0
    norm=wp.float64(0.)
    for i in range(delta.shape[0]):norm+=delta[i]*delta[i]
    s[2]=coef*wp.sqrt(norm)
    if lc[8]!=0 or not wp.isfinite(s[2]) or ls[5]>ls[1]:c[7]=2;c[4]=0
    elif s[0]<=s[1] and s[2]<=s[3]:c[4]=0
    else:c[5]=1;c[6]=0;s[4]=wp.float64(1.);s[5]=s[0]

@wp.kernel
def trial(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),a:wp.array(dtype=wp.float64),
          delta:wp.array(dtype=wp.float64),ids:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),coef:wp.float64,
          tu:wp.array(dtype=wp.float64),tl:wp.array(dtype=wp.float64),ta:wp.array(dtype=wp.float64)):
    i=wp.tid();j=ids[i]
    value=pair_add(wp.vec2d(u[j],ul[j]),pair_scale(wp.vec2d(delta[i],wp.float64(0.)),s[4]*coef))
    tu[j]=value[0];tl[j]=value[1];ta[i]=a[i]+s[4]*delta[i]

@wp.kernel
def line_decide(c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),trial_s:wp.array(dtype=wp.float64),
                status:wp.array(dtype=wp.int32),accept:wp.array(dtype=wp.int32),maximum:int):
    accept[0]=0
    if status[0]==0 and wp.isfinite(trial_s[0]) and trial_s[0]<=(wp.float64(1.)-wp.float64(1e-4)*s[4])*s[5]:
        accept[0]=1;c[5]=0;s[2]*=s[4]
    else:
        c[6]+=1;s[4]*=wp.float64(.5)
        if c[6]>=maximum:c[5]=0;c[4]=0;c[7]=3

@wp.kernel
def next_newton(c:wp.array(dtype=wp.int32)):
    if c[7]==0:c[0]+=1

@wp.kernel
def end_attempt(c:wp.array(dtype=wp.int32),failure:wp.array(dtype=wp.int32),switch:int):
    if c[7]==2 and c[1]==0 and c[10]==0:
        c[1]=1;c[2]=0;c[10]=1;c[11]=1
    else:
        c[11]=0
        if c[7]!=0:wp.atomic_max(failure,0,c[7])
        elif c[1]==0 and c[8]>=switch:c[1]=1;c[2]=0

@wp.kernel
def finish_velocity(v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),a0:wp.array(dtype=wp.float64),a:wp.array(dtype=wp.float64),
                    ids:wp.array(dtype=wp.int32),dt:wp.float64,out:wp.array(dtype=wp.float64),lo:wp.array(dtype=wp.float64)):
    i=wp.tid();j=ids[i]
    acc=pair_add(wp.vec2d(a0[i],wp.float64(0.)),wp.vec2d(a[i],wp.float64(0.)))
    value=pair_add(wp.vec2d(v[j],vl[j]),pair_scale(acc,wp.float64(.5)*dt));out[j]=value[0];lo[j]=value[1]

@wp.kernel
def step_success(c:wp.array(dtype=wp.int32),failure:wp.array(dtype=wp.int32),accept:wp.array(dtype=wp.int32)):
    accept[0]=0
    if failure[0]==0:accept[0]=1;c[13]+=1
    c[12]+=1;c[14]+=1

@wp.kernel
def frame_done(c:wp.array(dtype=wp.int32)):
    c[16]+=1

@wp.kernel
def audit_update(u0:wp.array(dtype=wp.float64),u0l:wp.array(dtype=wp.float64),v0:wp.array(dtype=wp.float64),v0l:wp.array(dtype=wp.float64),
                 u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),
                 ids:wp.array(dtype=wp.int32),dt:wp.float64,failure:wp.array(dtype=wp.int32)):
    i=wp.tid();j=ids[i]
    average=pair_scale(pair_add(wp.vec2d(v0[j],v0l[j]),wp.vec2d(v[j],vl[j])),wp.float64(.5)*dt)
    expected=pair_add(wp.vec2d(u0[j],u0l[j]),average)
    error=pair_add(wp.vec2d(u[j],ul[j]),wp.vec2d(-expected[0],-expected[1]))
    if not wp.isfinite(v[j]) or not wp.isfinite(u[j]) or wp.abs(error[0]+error[1])>wp.float64(2e-14):wp.atomic_max(failure,0,8)

@wp.kernel
def initial_energy(elastic:wp.array(dtype=wp.float64),kinetic:wp.array(dtype=wp.float64),energy:wp.array(dtype=wp.float64)):
    energy[0]=elastic[0]+wp.float64(.5)*kinetic[0]

@wp.kernel
def energy_balance(u0:wp.array(dtype=wp.float64),u0l:wp.array(dtype=wp.float64),u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),
                   force:wp.array(dtype=wp.float64),elastic:wp.array(dtype=wp.float64),kinetic:wp.array(dtype=wp.float64),energy:wp.array(dtype=wp.float64),failure:wp.array(dtype=wp.int32)):
    work=wp.float64(0.)
    for i in range(u.shape[0]):
        difference=pair_add(wp.vec2d(u[i],ul[i]),wp.vec2d(-u0[i],-u0l[i]))
        work+=force[i]*(difference[0]+difference[1])
    energy[2]=work;energy[1]=(elastic[0]+wp.float64(.5)*kinetic[0])-energy[0]-work
    if not wp.isfinite(energy[1]):wp.atomic_max(failure,0,9)

@wp.kernel
def candidate_ready(failure:wp.array(dtype=wp.int32),accept:wp.array(dtype=wp.int32)):
    accept[0]=0
    if failure[0]==0:accept[0]=1
