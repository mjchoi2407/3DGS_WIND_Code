"""매 프레임 첫 선형 풀이부터 current 보조 행렬을 사용하는 개발 후보."""
from contextlib import contextmanager
import warp as wp
from . import resident_step_kernels as k

wp.set_module_options({'enable_backward':False})

@wp.kernel
def begin_frame_current(c:wp.array(dtype=wp.int32)):
    c[1]=1;c[2]=0;c[14]=0

@contextmanager
def current_first():
    original=wp.launch
    def launch(kernel,*args,**kwargs):
        return original(begin_frame_current if kernel is k.begin_frame else kernel,*args,**kwargs)
    wp.load_module(module=__name__,device='cuda:0')
    wp.launch=launch
    try:yield
    finally:wp.launch=original
