"""독립 FP64 W1 진단: conditional Graph 안의 globaltimer 계측. 생산 경로에서 사용하지 않는다.
Nsight CUDA tracing 없이 연산군 시간을 기록한다. 포함 관계가 있어 행들을 합산하지 않는다.
"""
import argparse
import csv
import json
from pathlib import Path
import runpy
import sys
import warp as wp

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.func_native(snippet='''
#if defined(__CUDA_ARCH__)
unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(v)); return v;
#else
return 0;
#endif
''')
def now_ns()->wp.uint64: ...

@wp.kernel
def stamp(begin:wp.array(dtype=wp.uint64),role:int):
    begin[role]=now_ns()

@wp.kernel
def finish(begin:wp.array(dtype=wp.uint64),total:wp.array(dtype=wp.uint64),count:wp.array(dtype=wp.int64),role:int):
    total[role]+=now_ns()-begin[role];count[role]+=wp.int64(1)

ROLES=['hvp_mass','preconditioner_apply','dot_norm','orthogonalization_including_dot',
       'P_assembly_including_hvp','P_factor','force_energy_evaluate','line_search_including_force',
       'true_residual_action_including_hvp','empty_marker']

class Timer:
    def __init__(self,s):
        self.device=s.device;self.begin=wp.zeros(len(ROLES),dtype=wp.uint64,device=self.device)
        self.total=wp.zeros_like(self.begin);self.count=wp.zeros(len(ROLES),dtype=wp.int64,device=self.device)
        wp.load_module(module=sys.modules[__name__],device=self.device)
        for obj,name,role in [(s.preconditioner,'_matvec',1),(s.gmres,'inner_product',2),(s.gmres,'_orth',3),
            (s.coloring,'assemble',4),(s.current,'factor',5),(s.ops,'evaluate',6),(s,'_line_search',7)]:
            old=getattr(obj,name)
            setattr(obj,name,self.wrap(old,role))
        old=s.operator.matvec
        def action(*a,**kw):
            if kw.get('alpha')==-1. and kw.get('beta')==1.:
                return self.wrap(self.wrap(old,0),8)(*a,**kw)
            return self.wrap(old,0)(*a,**kw)
        s.operator._matvec=action
    def wrap(self,fn,role):
        def call(*a,**kw):
            wp.launch(stamp,1,inputs=[self.begin,role],device=self.device)
            value=fn(*a,**kw)
            wp.launch(finish,1,inputs=[self.begin,self.total,self.count,role],device=self.device)
            return value
        return call
    def reset(self):self.total.zero_();self.count.zero_()
    def rows(self,frame):
        total=self.total.numpy();count=self.count.numpy()
        return [dict(frame=frame,role=r,calls=int(count[i]),gpu_interval_s=float(total[i])*1e-9,
                     inclusive=True,instrumented=True) for i,r in enumerate(ROLES)]


def main():
    # launcher만 현재 소스이고 수치 solver와 입력은 caller의 frozen PYTHONPATH를 따른다.
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--blocks',required=True);p.add_argument('--frames',type=int,default=3);a=p.parse_args()
    from wind3dgs.teacher.resident_linear_trace_v3 import LinearTrace
    from wind3dgs.teacher.resident_newmark_gauss_retry import NewmarkGaussRetrySequence
    init=LinearTrace.__init__;run=NewmarkGaussRetrySequence.run_frame
    timers=[];rows=[];frame=[0]
    def traced_init(self,s,*args,**kwargs):
        init(self,s,*args,**kwargs);self.phase_timer=Timer(s);timers.append(self.phase_timer)
    def traced_run(self,*args,**kwargs):
        timer=self.solvers['base'].linear_v3.phase_timer;timer.reset()
        result=run(self,*args,**kwargs)
        rows.extend(timer.rows(frame[0]));frame[0]+=1
        with (a.out/'device_phase_times.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        (a.out/'device_phase_scope.json').write_text(json.dumps(dict(
            timing='GPU globaltimer nanoseconds around original operations in conditional Graph',
            numerical_changes=False,independent_audit='original worker; wall timing in result.json',
            restrictions=['instrumented run only; not ordinary speed','nested rows not additive','Gauss retry internals not instrumented','interval includes two marker launch gaps'],
            source_roles={r:i for i,r in enumerate(ROLES)}),indent=2))
        return result
    LinearTrace.__init__=traced_init;NewmarkGaussRetrySequence.run_frame=traced_run
    sys.argv=['teacher_precision_v3_worker','--root',str(a.root),'--out',str(a.out),'--stage','sequence','--blocks',a.blocks,'--frames',str(a.frames)]
    runpy.run_module('wind3dgs.evaluation.teacher_precision_v3_worker',run_name='__main__')

if __name__=='__main__':main()
