"""GPU 생성·검산 사이 전달과 프레임 경계의 실패 전파."""
import warp as wp

@wp.kernel
def store_state(u:wp.array(dtype=wp.float64),ul:wp.array(dtype=wp.float64),v:wp.array(dtype=wp.float64),vl:wp.array(dtype=wp.float64),
                target:wp.array3d(dtype=wp.float64),slot:int):
    i=wp.tid();target[slot,0,i]=u[i];target[slot,1,i]=ul[i];target[slot,2,i]=v[i];target[slot,3,i]=vl[i]

@wp.kernel
def store_balance(energy:wp.array(dtype=wp.float64),target:wp.array(dtype=wp.float64),slot:int):
    target[slot]=energy[1]

@wp.kernel
def ready(failure:wp.array(dtype=wp.int32),enabled:wp.array(dtype=wp.int32)):
    enabled[0]=int(failure[0]==0)

@wp.kernel
def stop_on_audit(flags:wp.array(dtype=wp.int32),start:int,count:int,first_bad:wp.array(dtype=wp.int32),
                  failure:wp.array(dtype=wp.int32),enabled:wp.array(dtype=wp.int32)):
    i=wp.tid()
    if enabled[0]!=0 and flags[start+i]!=0:
        wp.atomic_min(first_bad,0,start+i)
        wp.atomic_max(failure,0,99)

@wp.kernel
def stop_on_audit_masked(flags:wp.array(dtype=wp.int32),start:int,count:int,first_bad:wp.array(dtype=wp.int32),
                         failure:wp.array(dtype=wp.int32),enabled:wp.array(dtype=wp.int32),mask:int):
    i=wp.tid()
    if enabled[0]!=0 and (flags[start+i]&mask)!=0:
        wp.atomic_min(first_bad,0,start+i)
        wp.atomic_max(failure,0,99)
