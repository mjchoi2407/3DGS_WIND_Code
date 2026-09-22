"""접촉 합산: 블록 내 FP64 축약으로 전역 atomic 경합과 작은 연속 launch를 줄인다."""
import warp as wp

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
BLOCK = wp.constant(256)


@wp.kernel
def proxy_terms(hi: wp.array(dtype=wp.vec3d), lo: wp.array(dtype=wp.vec3d), dofs: wp.array2d(dtype=wp.int32),
                maps: wp.array3d(dtype=wp.float64), terms: wp.array(dtype=wp.float64),
                budget: wp.float64, status: wp.array(dtype=wp.int32)):
    element, sub, coefficient = wp.tid(); value = wp.vec3d(wp.float64(0.))
    origin = dofs[element,0]
    for j in range(10):
        idx = dofs[element,j]
        value += maps[sub,coefficient,j]*((hi[idx]-hi[origin])+(lo[idx]-lo[origin]))
    size = wp.length(value)
    terms[(element*maps.shape[0]+sub)*10+coefficient] = size
    if not wp.isfinite(size) or (budget >= wp.float64(0.) and size > budget+wp.float64(1e-14)):
        wp.atomic_max(status,0,6)


@wp.kernel
def reduce_sum(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64)):
    block = wp.tid()
    values = wp.tile_load(src,shape=BLOCK,offset=block*BLOCK)
    wp.tile_store(dst,wp.tile_sum(values),offset=block)


@wp.kernel
def reduce_max(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64)):
    block = wp.tid()
    # 입력은 길이/오차의 비음수 값이다. 마지막 블록의 범위 밖 값은0이다.
    values = wp.tile_load(src,shape=BLOCK,offset=block*BLOCK)
    wp.tile_store(dst,wp.tile_max(values),offset=block)


@wp.kernel
def accumulate_max(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64)):
    dst[0] = wp.max(dst[0],src[0])


class ContactReduction:
    def __init__(self,n,device,*,maximum=False):
        self.device = device
        self.kernel = reduce_max if maximum else reduce_sum
        self.levels = []
        while n > 1:
            n = (n+255)//256
            self.levels.append(wp.zeros(n,dtype=wp.float64,device=device))
        # block_dim=256 변형도 CUDA graph 시작 전에 컴파일한다.
        if self.levels:
            wp.launch_tiled(self.kernel,dim=1,inputs=[self.levels[-1],self.levels[-1]],
                            block_dim=256,device=device)

    def __call__(self,src):
        for dst in self.levels:
            wp.launch_tiled(self.kernel,dim=len(dst),inputs=[src,dst],block_dim=256,device=self.device)
            src = dst
        return src
