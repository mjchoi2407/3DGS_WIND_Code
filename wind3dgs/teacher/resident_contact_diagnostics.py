"""GPU 접촉 프레임의 최초 오류 보존과 제한 복구 자격 검사."""
import warp as wp

wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.kernel
def record_first_failure(failure:wp.array(dtype=wp.int32),loop:wp.array(dtype=wp.int32),
                         contact:wp.array(dtype=wp.int32),path:wp.array(dtype=wp.int32),
                         c:wp.array(dtype=wp.int32),s:wp.array(dtype=wp.float64),
                         lc:wp.array(dtype=wp.int32),ls:wp.array(dtype=wp.float64),
                         info:wp.array(dtype=wp.int32),controls:wp.array(dtype=wp.int32),
                         stats:wp.array(dtype=wp.float64)):
    if failure[0] != 0 and info[0] == 0:
        info[0]=failure[0];info[1]=loop[0];info[2]=contact[0];info[3]=path[0]
        finite=True
        for j in range(17):controls[j]=c[j]
        for j in range(9):
            controls[17+j]=lc[j];stats[j]=s[j]
            finite=finite and wp.isfinite(s[j])
        for j in range(10):
            stats[9+j]=ls[j];finite=finite and wp.isfinite(ls[j])
        info[4]=int(finite)


@wp.kernel
def allow_half_retry(failure:wp.array(dtype=wp.int32),info:wp.array(dtype=wp.int32),
                     flags:wp.array(dtype=wp.int32),time_status:wp.array(dtype=wp.int32),
                     allowed:wp.array(dtype=wp.int32)):
    ok=(failure[0]==2 and info[0]==2 and info[2]==0 and info[3]==0 and info[4]==1
        and time_status[0]==0 and info[1]>=0 and info[1]<flags.shape[0])
    # 미완료 tail의 flag14와 실제 승인 prefix의 검산 오류를 구분한다.
    for j in range(flags.shape[0]):
        if j<info[1] and flags[j]!=0:ok=False
    allowed[0]=int(ok)


class FirstFailure:
    def __init__(self,device='cuda:0'):
        self.device=device
        self.info=wp.zeros(5,dtype=wp.int32,device=device)
        self.controls=wp.zeros(26,dtype=wp.int32,device=device)
        self.stats=wp.zeros(19,dtype=wp.float64,device=device)
        wp.load_module(module=__name__,device=device)

    def reset(self):
        self.info.zero_();self.controls.zero_();self.stats.zero_()

    def record(self,solver,loop):
        wp.launch(record_first_failure,dim=1,inputs=[solver.failure,loop,solver.contact.status,
            solver.contact.path_status,solver.c,solver.s,solver.gmres.c,solver.gmres.s,
            self.info,self.controls,self.stats],device=self.device)

    def result(self):
        info=self.info.numpy()
        if not info[0]:return None
        c=self.controls.numpy();s=self.stats.numpy()
        return dict(failure=int(info[0]),substep=int(info[1]),contact_status=int(info[2]),
            contact_path_status=int(info[3]),finite=bool(info[4]),
            newton_iteration=int(c[0]),linear_iterations=int(c[17+7]),linear_status=int(c[17+8]),
            linear_residual_n=float(s[9+5]),linear_target_n=float(s[9+1]),
            controls=c.tolist(),stats=s.tolist())
