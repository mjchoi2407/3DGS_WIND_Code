"""고정 reference 질량으로 적분한 중력을 프레임 외력에 더한다. +Z가 위쪽이다."""
import numpy as np
import warp as wp
from .p3_shell_resident_stepper import ResidentShellStepper

@wp.kernel
def add_gravity(weights: wp.array(dtype=wp.float64), gravity: wp.array(dtype=wp.vec3d),
                control: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.float64)):
    i=wp.tid()
    for c in range(3):
        held[3*i+c] += weights[i]*gravity[control[16]][c]


def gravity_load(model, acceleration):
    """일정 체적 가속도의 consistent FE 외력. 질량 역행렬은 바꾸지 않는다."""
    weights=np.asarray(model.mass@np.ones(len(model.rest_positions)))
    return weights[:,None]*np.asarray(acceleration,dtype=np.float64)


class GravityShellStepper(ResidentShellStepper):
    def __init__(self,model,displacement,velocity,wind,*,gravity,**kwargs):
        g=np.asarray(gravity,dtype=np.float64)
        if g.shape!=np.shape(wind) or not np.isfinite(g).all():
            raise ValueError('중력은 바람과 같은 프레임 수의 유한한3벡터 배열이어야 합니다')
        device=kwargs.get('device','cuda:0')
        self.gravity=wp.array(g,dtype=wp.vec3d,device=device)
        self.gravity_weights=wp.array(np.asarray(model.mass@np.ones(len(model.rest_positions))),dtype=wp.float64,device=device)
        wp.load_module(module=__name__,device=device)
        super().__init__(model,displacement,velocity,wind,**kwargs)

    def _aero(self):
        super()._aero()
        wp.launch(add_gravity,dim=self.nodes,inputs=[self.gravity_weights,self.gravity,self.c,self.held],device=self.device)
