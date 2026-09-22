"""Newton 선형계 device 로그와 지정 solve 한 개의 원본 snapshot. CPU read는 저장 경계만."""
import numpy as np
import warp as wp
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def trace_begin(index:wp.array(dtype=wp.int32),step:wp.array(dtype=wp.int32),tol:wp.array(dtype=wp.float64),norm:wp.array(dtype=wp.float64),rows:wp.array2d(dtype=wp.float64),target:int,take:wp.array(dtype=wp.int32),frame:wp.array(dtype=wp.int32),rebuild_every:int,origin:wp.array(dtype=wp.int32),generation:wp.array(dtype=wp.int32)):
    i=index[0];take[0]=0
    if i==target:take[0]=1
    if i<rows.shape[0]:
        rows[i,0]=wp.float64(i);rows[i,1]=wp.float64(frame[0]);rows[i,2]=wp.float64(step[14]+origin[0]);rows[i,3]=wp.float64(step[0])
        rows[i,4]=tol[0];rows[i,5]=wp.sqrt(norm[0]);rows[i,6]=tol[0]*wp.sqrt(norm[0]);rows[i,7]=wp.float64(step[1]);rows[i,8]=wp.float64(generation[0]);rows[i,9]=wp.float64(-1.)
        if step[1]!=0:rows[i,9]=wp.float64((step[2]-1)%rebuild_every)

@wp.kernel
def trace_end(index:wp.array(dtype=wp.int32),gc:wp.array(dtype=wp.int32),gs:wp.array(dtype=wp.float64),rows:wp.array2d(dtype=wp.float64),counts:wp.array(dtype=wp.int32)):
    i=index[0]
    if i<rows.shape[0]:
        rows[i,5]=gs[0];rows[i,6]=gs[1];rows[i,10]=wp.float64(gc[7]);rows[i,11]=wp.float64(gc[2]);rows[i,12]=gs[5];rows[i,13]=wp.float64(gc[8]);rows[i,14]=wp.float64(counts[4]);rows[i,15]=wp.float64(counts[1]);rows[i,16]=wp.float64(counts[2]);rows[i,17]=wp.float64(counts[5])
    index[0]+=1

@wp.kernel
def next_generation(generation:wp.array(dtype=wp.int32)):
    generation[0]+=1

@wp.kernel
def record_newton(index:wp.array(dtype=wp.int32),step:wp.array(dtype=wp.int32),rows:wp.array2d(dtype=wp.float64)):
    i=index[0]-1
    if i>=0 and i<rows.shape[0]:
        rows[i,18]=wp.float64(0.)
        if rows[i,21]!=wp.float64(0.):rows[i,18]=wp.float64(step[6])
        rows[i,19]=wp.float64(step[7]);rows[i,20]=wp.float64(step[3])

@wp.kernel
def record_line_start(index:wp.array(dtype=wp.int32),step:wp.array(dtype=wp.int32),rows:wp.array2d(dtype=wp.float64)):
    i=index[0]-1
    if i>=0 and i<rows.shape[0]:rows[i,21]=wp.float64(step[5])

@wp.kernel
def store_corrections(index:wp.array(dtype=wp.int32),hist:wp.array2d(dtype=wp.float64),out:wp.array3d(dtype=wp.float64)):
    j,k=wp.tid();i=index[0]
    if i<out.shape[0]:out[i,j,k]=hist[j,k]


FIELDS=('solve_id','frame','substep','newton','eta','rhs_l2','target','current_P','generation','age','iterations','cycles','true_residual','linear_failure','fallback_cumulative','correction_cumulative','inner_iter_cumulative','fallback_iter_cumulative','backtracks','newton_failure','P_rebuilt','line_search_attempted')

class LinearTrace:
    def __init__(self,s,mode='R64',target=-1,capacity=32768):
        self.s=s;self.g=s.gmres;self.mode=mode;self.target=target;self.device=s.device
        self.strategy=None
        if mode in ('M1','M2','M2_P64'):
            from .resident_precision_v3 import MixedLinear
            self.strategy=MixedLinear(s,mode)
        self.index=wp.zeros(1,dtype=wp.int32,device=s.device);self.frame=wp.zeros_like(self.index);self.take=wp.zeros_like(self.index)
        self.origin=wp.zeros_like(self.index);self.generation=wp.zeros_like(self.index)
        self.ir_history=wp.zeros((capacity,6,6),dtype=wp.float64,device=s.device) if self.strategy else None
        self.rows=wp.zeros((capacity,len(FIELDS)),dtype=wp.float64,device=s.device)
        self.counts=self.strategy.counts if self.strategy else wp.zeros(6,dtype=wp.int32,device=s.device)
        self.saved={}
        if target>=0:
            arrays=dict(uh=s.uh,lo=s.lo,a=s.a,b64=s.rhs,held=s.held,eta=s.tolerance,P64=s.current.matrix.values,Prest64=s.rest.matrix.values,x64=s.delta,
                u_hi=s.u,u_lo=s.ul,v_hi=s.v,v_lo=s.vl,basis=s.gmres.tmp)
            self.saved={k:wp.empty_like(v) for k,v in arrays.items()};self.source=arrays
        wp.load_module(module=__name__,device=s.device)
    def launch(self,k,args):wp.launch(k,dim=1,inputs=args,device=self.device)
    def factor(self):
        if self.strategy:self.strategy.factor()
        else:self.s.current.factor()
        self.launch(next_generation,[self.generation])
    def before(self):
        for k,v in self.source.items():
            if k not in ('x64','basis'):wp.copy(self.saved[k],v)
    def after(self):
        for k in ('x64','basis'):wp.copy(self.saved[k],self.source[k])
    def __call__(self):
        self.launch(trace_begin,[self.index,self.s.c,self.s.tolerance,self.g.bn,self.rows,self.target,self.take,self.frame,self.s.rebuild_every,self.origin,self.generation])
        if self.target>=0:wp.capture_if(self.take,self.before)
        if self.strategy:self.strategy()
        else:self.g()
        if self.target>=0:wp.capture_if(self.take,self.after)
        if self.strategy:wp.launch(store_corrections,dim=(6,6),inputs=[self.index,self.strategy.hist,self.ir_history],device=self.device)
        self.launch(trace_end,[self.index,self.g.c,self.g.s,self.rows,self.counts])
    def before_line(self):self.launch(record_line_start,[self.index,self.s.c,self.rows])
    def after_newton(self):self.launch(record_newton,[self.index,self.s.c,self.rows])
    def report(self):
        count=int(self.index.numpy()[0]);data=self.rows.numpy()[:min(count,len(self.rows))]
        return dict(count=count,overflow=count>len(self.rows),rows=[dict(zip(FIELDS,map(float,r))) for r in data],
            counts=self.counts.numpy().tolist())
    def close(self):
        if self.strategy:self.strategy.close()


def specialize_stepper(source,mode,target=-1):
    from ..evaluation.teacher_precision_compare import once
    source=once(source,'        wp.load_module(module=k,device=self.device);',
        '        from .resident_linear_trace_v3 import LinearTrace\n'+f'        self.linear_v3=LinearTrace(self,mode={mode!r},target={target})\n'+'        wp.load_module(module=k,device=self.device);')
    source=once(source,'self.coloring.assemble();self.current.factor();','self.coloring.assemble();self.linear_v3.factor();')
    source=once(source,'        self.gmres()\n','        self.linear_v3()\n')
    source=once(source,'        wp.capture_while(self.c[5:6],self._line_search)\n','        self.linear_v3.before_line()\n        wp.capture_while(self.c[5:6],self._line_search)\n        self.linear_v3.after_newton()\n')
    source=once(source,'    def close(self):\n','    def close(self):\n        self.linear_v3.close()\n')
    return source
