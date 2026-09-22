"""CuPy device 벡터를 유지하는 GMRES와 분석 계획을 재사용하는 희소 LU 풀이.

GMRES의 left preconditioning/MGS/종료 제어는 SciPy iterative.gmres를 기반으로 한다.
고정 restart 60, cycle 8 및 원래 true residual 기준은 호출자가 지정한다.
큰 벡터·희소 풀이·내적은 GPU, 작은 Hessenberg/Givens 제어는 CPU에 둔다.
SpSM descriptor 구성은 CuPy 13.6 cupyx.cusparse.spsm을 기반으로 한다.
CuPy 13.x 내부 API를 사용하므로 optional dependency major를 제한한다.

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


CuPy license:
Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.cusparse import SpMatDescriptor, DnMatDescriptor
from cupy_backends.cuda.libs import cusparse as cs
from cupy.cuda import device
from cupy._core import _dtype
from scipy.sparse.linalg import splu as cpu_splu
from scipy.linalg.lapack import dlartg

class _TriangularPlan:
    def __init__(self,matrix,columns,lower):
        self.matrix=matrix
        self.b=cp.zeros((matrix.shape[0],columns),dtype=cp.float64,order='F')
        self.x=cp.zeros_like(self.b,order='F')
        self.handle=device.get_cusparse_handle()
        self.a_desc=SpMatDescriptor.create(matrix);self.b_desc=DnMatDescriptor.create(self.b);self.x_desc=DnMatDescriptor.create(self.x)
        self.desc=cs.spSM_createDescr();self.alpha=np.array(1.,dtype=np.float64)
        self.a_desc.set_attribute(cs.CUSPARSE_SPMAT_FILL_MODE,cs.CUSPARSE_FILL_MODE_LOWER if lower else cs.CUSPARSE_FILL_MODE_UPPER)
        self.a_desc.set_attribute(cs.CUSPARSE_SPMAT_DIAG_TYPE,cs.CUSPARSE_DIAG_TYPE_NON_UNIT)
        self.args=(self.handle,cs.CUSPARSE_OPERATION_NON_TRANSPOSE,cs.CUSPARSE_OPERATION_NON_TRANSPOSE,
            self.alpha.ctypes.data,self.a_desc.desc,self.b_desc.desc,self.x_desc.desc,
            _dtype.to_cuda_dtype(self.b.dtype),cs.CUSPARSE_SPSM_ALG_DEFAULT,self.desc)
        self.workspace=cp.empty(cs.spSM_bufferSize(*self.args),dtype=cp.int8)
        cs.spSM_analysis(*self.args,self.workspace.data.ptr)

    def solve(self,rhs):
        cp.copyto(self.b,rhs.reshape(self.b.shape));self.x.fill(0)
        cs.spSM_solve(*self.args,self.workspace.data.ptr)
        return self.x.reshape(rhs.shape)

    def __del__(self):
        if getattr(self,'desc',None) is not None:
            try:cs.spSM_destroyDescr(self.desc)
            except Exception:pass
            self.desc=None

class CachedGPULU:
    """한 번의 CPU LU 분해 후 GPU 삼각 풀이/analysis 재사용. 동시 호출 불가."""
    def __init__(self,matrix):
        cpu=matrix.get() if hasattr(matrix,'get') else matrix
        lu=cpu_splu(cpu.tocsc())
        self.L=csr_matrix(lu.L.tocsr());self.U=csr_matrix(lu.U.tocsr())
        self.r=cp.argsort(cp.asarray(lu.perm_r));self.c=cp.asarray(lu.perm_c);self.plans={}
    def solve(self,rhs):
        columns=1 if rhs.ndim==1 else rhs.shape[1]
        if columns not in self.plans:
            self.plans[columns]=(_TriangularPlan(self.L,columns,True),_TriangularPlan(self.U,columns,False))
        lower,upper=self.plans[columns]
        return upper.solve(lower.solve(rhs[self.r]))[self.c]

def gmres_early(A,b,*,M,restart,maxiter,atol=0.,tol=1e-10,callback=None,callback_type='pr_norm'):
    """Float64 real, zero initial guess, left-preconditioned restarted GMRES.

    maxiter는 SciPy와 같은 restart cycle 수. callback은 inner iteration마다 호출.
    """
    if b.dtype!=cp.float64 or b.ndim!=1:raise ValueError('float64 vector 필요')
    if callback_type!='pr_norm':raise ValueError('pr_norm callback만 지원')
    n=len(b);x=cp.zeros_like(b);bnorm=float(cp.linalg.norm(b));target=max(atol,tol*bnorm)
    if bnorm==0:return x,0
    restart=min(restart,n);eps=np.finfo(float).eps
    mbnorm=float(cp.linalg.norm(M@b));factor=1.;ptol=mbnorm*min(factor,target/bnorm)
    V=cp.empty((restart+1,n),dtype=cp.float64);H=np.zeros((restart,restart+1));givens=np.zeros((restart,2))
    r=b.copy();rnorm=bnorm
    for cycle in range(maxiter):
        V[0]=M@r;vnorm=float(cp.linalg.norm(V[0]))
        if not np.isfinite(vnorm) or vnorm==0:return x,maxiter
        V[0]/=vnorm;S=np.zeros(restart+1);S[0]=vnorm;breakdown=False
        for j in range(restart):
            w=(M@(A@V[j])).copy();h0=float(cp.linalg.norm(w))
            for k in range(j+1):
                coefficient=cp.dot(V[k],w);H[j,k]=float(coefficient);w-=coefficient*V[k]
            h1=float(cp.linalg.norm(w));H[j,j+1]=h1;V[j+1]=w
            if h1<=eps*h0:H[j,j+1]=0.;breakdown=True
            else:V[j+1]/=h1
            for k in range(j):
                c,s=givens[k];h,k1=H[j,k:k+2];H[j,k:k+2]=(c*h+s*k1,-s*h+c*k1)
            c,s,mag=dlartg(H[j,j],H[j,j+1]);givens[j]=(c,s);H[j,j:j+2]=(mag,0.)
            tail=-s*S[j];S[j:j+2]=(c*S[j],tail);presid=abs(tail)
            if callback is not None:callback(presid/bnorm)
            if presid<=ptol or breakdown:break
        if H[j,j]==0:S[j]=0.
        y=S[:j+1].copy()
        for k in range(j,0,-1):
            if y[k]!=0:y[k]/=H[k,k];y[:k]-=y[k]*H[k,:k]
        if y[0]!=0:y[0]/=H[0,0]
        x+=cp.asarray(y)@V[:j+1];r=b-A@x;rnorm=float(cp.linalg.norm(r))
        if rnorm<=target:return x,0
        if breakdown or not np.isfinite(rnorm):break
        if presid<=ptol:factor=max(eps,.25*factor)
        else:factor=min(1.,1.5*factor)
        ptol=presid*min(factor,target/rnorm)
    return x,maxiter
