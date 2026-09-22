"""CUDA 12.x의 조건부 graph를 생성 때 추적해 body까지 읽기 전용 검사한다.

기존 driver에는 조건부 body를 사후 조회하는 API가 없다. Warp 1.17의 생성 직후
body 검사 hook에서 주소를 보관한다. 별도 worker의 초기 capture 동안만 사용한다.
"""
from contextlib import contextmanager
import ctypes
import warp as wp


@contextmanager
def track_conditional_bodies():
    wp.init()
    from warp._src.context import runtime
    original=runtime.core.wp_cuda_graph_check_conditional_body
    bodies=set()
    def check(graph):
        result=original(graph)
        if result:
            bodies.add(graph.value if isinstance(graph,ctypes.c_void_p) else graph)
        return result
    runtime.core.wp_cuda_graph_check_conditional_body=check
    try:
        yield bodies
    finally:
        runtime.core.wp_cuda_graph_check_conditional_body=original
