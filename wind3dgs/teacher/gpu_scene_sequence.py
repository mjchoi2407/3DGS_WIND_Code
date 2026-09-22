"""프레임 경계에서만 기존 solver를 선택하고 전환 비용을 별도 기록한다."""
import time
import numpy as np
from .gpu_scene_policy import environment, select, linear_mode
from .force_launch_profile import force_launches, ROLES
from .resident_capture_audit import track_conditional_bodies
from .resident_audit import device_graph_inventory

class GPUSceneSequence:
    def __init__(self, sequence_type, model, initial, policy, *, scene_config, phase, **kwargs):
        self.sequence_type=sequence_type; self.model=model; self.initial=initial; self.policy=policy
        self.cfg=scene_config; self.phase=phase; self.kwargs=kwargs; self.env=environment()
        self.sequence=None; self.frame=0; self.launch_records=[]; self.segments=[]
        self._ensure(initial)
    def _choice(self):
        return select(self.env['gpu'],self.env['driver_api'],self.phase,self.cfg['shape'],self.frame,self.cfg['gpu_policy'])
    def _ensure(self, initial):
        choice=self._choice()
        if self.sequence is not None and choice['method']==self.choice['method']: return 0.
        start=time.perf_counter()
        if self.sequence is not None: self.sequence.close(); self.sequence=None
        kwargs=dict(self.kwargs)
        if self.cfg.get('solver_backend')=='newmark_adaptive_gauss':
            kwargs.update(gauss_mode=choice['direct_gauss'],
                          branch_probe_steps=self.cfg['branch_probe_steps'],
                          branch_failure_count=self.cfg['branch_failure_count'],
                          branch_cost_margin=self.cfg['branch_cost_margin'])
        with track_conditional_bodies() as bodies, linear_mode(choice['method']), force_launches(dict(zip(ROLES,choice['blocks'])),self.launch_records):
            self.sequence=self.sequence_type(self.model,initial,self.policy,**kwargs)
        graphs={name:device_graph_inventory(s.step_graph,conditional_bodies=bodies) for name,s in self.sequence.solvers.items()}
        self.choice=choice; elapsed=time.perf_counter()-start
        self.segments.append(dict(start_frame=self.frame,setup_s=elapsed,graph_inventory=graphs,**choice))
        return elapsed
    def __getattr__(self,name): return getattr(self.sequence,name)
    def run_frame(self, wind, gravity):
        switch=0.
        if self._choice()['method'] != self.choice['method']:
            transition_start=time.perf_counter()
            initial=[a.numpy().reshape(-1,3) for a in self.sequence.state]
            self._ensure(initial)
            switch=time.perf_counter()-transition_start
        strategy=self.sequence.solvers['base'].scene_linear
        log_start=time.perf_counter()
        before=np.zeros(6,dtype=np.int64) if strategy is None else strategy.counts.numpy()
        log_s=time.perf_counter()-log_start
        with force_launches(dict(zip(ROLES,self.choice['blocks'])),self.launch_records):
            result=self.sequence.run_frame(wind,gravity)
        if 'audit_graph_inventory' not in self.segments[-1]:
            self.segments[-1]['audit_graph_inventory']={name:device_graph_inventory(a.graph) for name,a in self.sequence.audit.items() if getattr(a,'graph',None) is not None}
        log_start=time.perf_counter()
        after=before if strategy is None else strategy.counts.numpy()
        log_s+=time.perf_counter()-log_start
        result['gpu_selection']=dict(self.choice,transition_setup_s=switch,summary_transfer_s=log_s,mixed_counts=(after-before).tolist())
        self.frame+=1
        return result
    def close(self):
        if self.sequence is not None: self.sequence.close(); self.sequence=None
