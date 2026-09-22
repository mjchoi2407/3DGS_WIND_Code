"""v3 inner: 큰 벡터 FP32, 내적·norm·Hessenberg·Givens FP64 GMRES.

SciPy 및 기존 gmres_early의 잔차/반복 한도 의미를 유지한다. CPU 결과 조회 없음.

SciPy license:
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import warp as wp
from warp._src.optim.linear import TiledDot
wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})

@wp.kernel
def initialize(c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64), bn: wp.array(dtype=wp.float64), mn: wp.array(dtype=wp.float64), tol: wp.array(dtype=wp.float64)):
    for k in range(9):
        c[k] = 0
    for k in range(10):
        s[k] = wp.float64(0.0)
    s[0] = wp.sqrt(bn[0])
    s[1] = s[0] * tol[0]
    s[2] = wp.sqrt(mn[0])
    s[3] = wp.float64(1.0)
    s[4] = s[2] * wp.min(s[3], tol[0])
    s[5] = s[0]
    if s[0] > wp.float64(0.0):
        c[3] = 1
    if not wp.isfinite(s[0]) or not wp.isfinite(s[2]):
        c[3] = 0
        c[8] = 1

@wp.kernel
def cycle_start(c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64), norm: wp.array(dtype=wp.float64), H: wp.array2d(dtype=wp.float64), givens: wp.array2d(dtype=wp.float64), rhs: wp.array(dtype=wp.float64)):
    c[0] = 0
    c[6] = 0
    c[4] = 1
    s[9] = wp.sqrt(norm[0])
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            H[i, j] = wp.float64(0.0)
        givens[i, 0] = wp.float64(0.0)
        givens[i, 1] = wp.float64(0.0)
    for i in range(rhs.shape[0]):
        rhs[i] = wp.float64(0.0)
    rhs[0] = s[9]
    if not wp.isfinite(s[9]) or s[9] == wp.float64(0.0):
        c[4] = 0
        c[8] = 1

@wp.kernel
def first_basis(w: wp.array(dtype=wp.float32), V: wp.array2d(dtype=wp.float32), s: wp.array(dtype=wp.float64)):
    i = wp.tid()
    V[0, i] = wp.float32(0.0)
    if s[9] > wp.float32(0.0):
        V[0, i] = w[i] / wp.float32(s[9])

@wp.kernel
def select_basis(V: wp.array2d(dtype=wp.float32), c: wp.array(dtype=wp.int32), which: int, x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x[i] = V[c[which], i]

@wp.kernel
def orth_start(c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64), norm: wp.array(dtype=wp.float64)):
    c[1] = 0
    c[5] = 1
    s[7] = wp.sqrt(norm[0])

@wp.kernel
def subtract_basis(w: wp.array(dtype=wp.float32), basis: wp.array(dtype=wp.float32), dot: wp.array(dtype=wp.float64)):
    i = wp.tid()
    w[i] -= wp.float32(dot[0]) * basis[i]

@wp.kernel
def orth_next(c: wp.array(dtype=wp.int32), H: wp.array2d(dtype=wp.float64), dot: wp.array(dtype=wp.float64)):
    H[c[0], c[1]] = dot[0]
    c[1] += 1
    if c[1] > c[0]:
        c[5] = 0

@wp.kernel
def inner_end(c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64), norm: wp.array(dtype=wp.float64), H: wp.array2d(dtype=wp.float64), g: wp.array2d(dtype=wp.float64), rhs: wp.array(dtype=wp.float64)):
    j = c[0]
    h1 = wp.sqrt(norm[0])
    s[8] = h1
    H[j, j + 1] = h1
    if h1 <= wp.float64(2.220446049250313e-16) * s[7]:
        H[j, j + 1] = wp.float64(0.0)
        c[6] = 1
    for k in range(j):
        a = H[j, k]
        b = H[j, k + 1]
        H[j, k] = g[k, 0] * a + g[k, 1] * b
        H[j, k + 1] = -g[k, 1] * a + g[k, 0] * b
    a = H[j, j]
    b = H[j, j + 1]
    scale = wp.max(wp.abs(a), wp.abs(b))
    cosine = wp.float64(1.0)
    sine = wp.float64(0.0)
    r = wp.float64(0.0)
    if scale > wp.float64(0.0):
        r = scale * wp.sqrt(a / scale * (a / scale) + b / scale * (b / scale))
        if a < wp.float64(0.0):
            r = -r
        cosine = a / r
        sine = b / r
    g[j, 0] = cosine
    g[j, 1] = sine
    H[j, j] = r
    H[j, j + 1] = wp.float64(0.0)
    tail = -sine * rhs[j]
    rhs[j] = cosine * rhs[j]
    rhs[j + 1] = tail
    s[6] = wp.abs(tail)
    c[7] += 1
    if s[6] <= s[4] or c[6] != 0 or j + 1 >= H.shape[0]:
        c[4] = 0
    if not wp.isfinite(s[6]) or not wp.isfinite(h1):
        c[4] = 0
        c[8] = 1

@wp.kernel
def next_basis(w: wp.array(dtype=wp.float32), V: wp.array2d(dtype=wp.float32), c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64)):
    i = wp.tid()
    V[c[0] + 1, i] = wp.float32(0.0)
    if c[6] == 0 and s[8] > wp.float64(0.0):
        V[c[0] + 1, i] = w[i] / wp.float32(s[8])

@wp.kernel
def next_inner(c: wp.array(dtype=wp.int32)):
    if c[4] != 0:
        c[0] += 1

@wp.kernel
def backsolve(c: wp.array(dtype=wp.int32), H: wp.array2d(dtype=wp.float64), rhs: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
    j = c[0]
    for k in range(j + 1):
        y[k] = rhs[k]
    if H[j, j] == wp.float64(0.0):
        y[j] = wp.float64(0.0)
    for reverse in range(j + 1):
        k = j - reverse
        if y[k] != wp.float64(0.0):
            y[k] = y[k] / H[k, k]
            for i in range(k):
                y[i] -= y[k] * H[k, i]

@wp.kernel
def add_solution(x: wp.array(dtype=wp.float32), V: wp.array2d(dtype=wp.float32), y: wp.array(dtype=wp.float64), c: wp.array(dtype=wp.int32)):
    i = wp.tid()
    value = wp.float32(0.0)
    for k in range(c[0] + 1):
        value += wp.float32(y[k]) * V[k, i]
    x[i] += value

@wp.kernel
def cycle_end(c: wp.array(dtype=wp.int32), s: wp.array(dtype=wp.float64), norm: wp.array(dtype=wp.float64), max_cycles: int):
    s[5] = wp.sqrt(norm[0])
    c[2] += 1
    if s[5] <= s[1]:
        c[3] = 0
        c[8] = 0
    elif c[8] != 0 or c[6] != 0 or c[2] >= max_cycles or (not wp.isfinite(s[5])):
        c[3] = 0
        c[8] = 1
    else:
        if s[6] <= s[4]:
            s[3] = wp.max(wp.float64(2.220446049250313e-16), wp.float64(0.25) * s[3])
        else:
            s[3] = wp.min(wp.float64(1.0), wp.float64(1.5) * s[3])
        s[4] = s[6] * wp.min(s[3], s[1] / s[5])

class InnerGMRES32:

    def __init__(self, A, M, b, x, tolerance, *, restart=240, cycles=3):
        self.A, self.M, self.b, self.x, self.tol = (A, M, b, x, tolerance)
        self.device = b.device
        self.n = len(b)
        self.restart = min(restart, self.n)
        self.cycles = cycles
        self.c = wp.zeros(9, dtype=wp.int32, device=self.device)
        self.s = wp.zeros(10, dtype=wp.float64, device=self.device)
        self.cycle = self.c[3:4]
        self.inner = self.c[4:5]
        self.orth = self.c[5:6]
        self.r, self.w, self.tmp, self.av = [wp.zeros_like(b) for _ in range(4)]
        self.bn, self.mn, self.dot = [wp.zeros(1, dtype=wp.float64, device=self.device) for _ in range(3)]
        self.V = wp.zeros((self.restart + 1, self.n), dtype=wp.float32, device=self.device)
        self.H = wp.zeros((self.restart, self.restart + 1), dtype=wp.float64, device=self.device)
        self.g = wp.zeros((self.restart, 2), dtype=wp.float64, device=self.device)
        self.rhs = wp.zeros(self.restart + 1, dtype=wp.float64, device=self.device)
        self.y = wp.zeros_like(self.rhs)
        self.dotter = TiledDot(max_length=self.n, device=self.device, scalar_type=wp.float64)
        wp.load_module(module=__name__, device=self.device)
        self.dot_a64 = wp.zeros(self.n, dtype=wp.float64, device=self.device)
        self.dot_b64 = wp.zeros_like(self.dot_a64)

    def launch(self, kernel, args, dim=1):
        wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def inner_product(self, a, b, out):
        from .resident_precision_v3 import copy_vectors
        self.launch(copy_vectors, [a, b, self.dot_a64, self.dot_b64], self.n)
        self.dotter.compute(self.dot_a64, self.dot_b64)
        wp.copy(out, self.dotter.col(0))

    def __call__(self):
        self.x.zero_()
        wp.copy(self.r, self.b)
        self.M.matvec(self.b, self.w, self.w, alpha=1.0, beta=0.0)
        self.inner_product(self.b, self.b, out=self.bn)
        self.inner_product(self.w, self.w, out=self.mn)
        self.launch(initialize, [self.c, self.s, self.bn, self.mn, self.tol])
        wp.capture_while(self.cycle, self._cycle)
        return (self.c, self.s)

    def _cycle(self):
        def restarted():
            self.M.matvec(self.r,self.w,self.w,alpha=1.,beta=0.)
            self.inner_product(self.w,self.w,out=self.dot)
        wp.capture_if(self.c[2:3],restarted,lambda:wp.copy(self.dot,self.mn))
        self.launch(cycle_start, [self.c, self.s, self.dot, self.H, self.g, self.rhs])
        self.launch(first_basis, [self.w, self.V, self.s], self.n)
        wp.capture_while(self.inner, self._inner)
        self.launch(backsolve, [self.c, self.H, self.rhs, self.y])
        self.launch(add_solution, [self.x, self.V, self.y, self.c], self.n)
        self.A.matvec(self.x, self.b, self.r, alpha=-1.0, beta=1.0)
        self.inner_product(self.r, self.r, out=self.dot)
        self.launch(cycle_end, [self.c, self.s, self.dot, self.cycles])

    def _inner(self):
        self.launch(select_basis, [self.V, self.c, 0, self.tmp], self.n)
        self.A.matvec(self.tmp, self.av, self.av, alpha=1.0, beta=0.0)
        self.M.matvec(self.av, self.w, self.w, alpha=1.0, beta=0.0)
        self.inner_product(self.w, self.w, out=self.dot)
        self.launch(orth_start, [self.c, self.s, self.dot])
        wp.capture_while(self.orth, self._orth)
        self.inner_product(self.w, self.w, out=self.dot)
        self.launch(inner_end, [self.c, self.s, self.dot, self.H, self.g, self.rhs])
        self.launch(next_basis, [self.w, self.V, self.c, self.s], self.n)
        self.launch(next_inner, [self.c])

    def _orth(self):
        self.launch(select_basis, [self.V, self.c, 1, self.tmp], self.n)
        self.inner_product(self.tmp, self.w, out=self.dot)
        self.launch(subtract_basis, [self.w, self.tmp, self.dot], self.n)
        self.launch(orth_next, [self.c, self.H, self.dot])
