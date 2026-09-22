"""FP32 쌍으로 법선과 법선 차이를 계산하는 선택적 보정. 모든 연산은 FP32다.

Rest/shape 계수의 FP32 양자화는 복원하지 않는다. 비퇴화 요소에서만 호출한다.
Two-Sum/분할 곱의 평가 순서를 유지한다.
"""
import warp as wp
wp.set_module_options({'enable_backward':False, 'fast_math':False, 'fuse_fp':False})


@wp.func
def nsum(a:wp.float32,b:wp.float32):
    s=a+b;bb=s-a
    return wp.vec2f(s,(a-(s-bb))+(b-bb))


@wp.func
def nproduct(a:wp.float32,b:wp.float32):
    p=a*b;ca=wp.float32(4097.)*a;cb=wp.float32(4097.)*b
    ah=ca-(ca-a);bh=cb-(cb-b);al=a-ah;bl=b-bh
    return wp.vec2f(p,((ah*bh-p)+ah*bl+al*bh)+al*bl)


@wp.func
def nadd(a:wp.vec2f,b:wp.vec2f):
    s=nsum(a[0],b[0]);t=nsum(a[1],b[1]);r=nsum(s[0],s[1]+t[0])
    return nsum(r[0],r[1]+t[1])


@wp.func
def nmul(a:wp.vec2f,b:wp.vec2f):
    return nadd(nproduct(a[0],b[0]),wp.vec2f(a[0]*b[1]+a[1]*b[0],a[1]*b[1]))


@wp.func
def ndiv(a:wp.vec2f,b:wp.vec2f):
    q=a[0]/b[0];r=nadd(a,-nmul(b,wp.vec2f(q,wp.float32(0.))))
    return nsum(q,(r[0]+r[1])/b[0])


@wp.struct
class NormalPair:
    hi:wp.vec3f
    lo:wp.vec3f
    area:wp.float32


@wp.func
def normal_pair(ah:wp.vec3f,al:wp.vec3f,bh:wp.vec3f,bl:wp.vec3f):
    ch=wp.vec3f(wp.float32(0.));cl=ch;square=wp.vec2f(wp.float32(0.))
    for i in range(3):
        j=(i+1)%3;k=(i+2)%3
        c=nadd(nmul(wp.vec2f(ah[j],al[j]),wp.vec2f(bh[k],bl[k])),
               -nmul(wp.vec2f(ah[k],al[k]),wp.vec2f(bh[j],bl[j])))
        ch[i]=c[0];cl[i]=c[1];square=nadd(square,nmul(c,c))
    root=wp.sqrt(square[0]);remainder=nadd(square,-nproduct(root,root))
    length=nsum(root,(remainder[0]+remainder[1])/(wp.float32(2.)*root))
    result=NormalPair();result.area=length[0]+length[1]
    for i in range(3):
        value=ndiv(wp.vec2f(ch[i],cl[i]),length)
        result.hi[i]=value[0];result.lo[i]=value[1]
    return result


@wp.func
def normal_difference(ah:wp.vec3f,al:wp.vec3f,bh:wp.vec3f,bl:wp.vec3f):
    result=wp.vec3f(wp.float32(0.))
    for i in range(3):
        value=nadd(wp.vec2f(ah[i],al[i]),-wp.vec2f(bh[i],bl[i]))
        result[i]=value[0]+value[1]
    return result


@wp.func
def ndot(ah:wp.vec3f,al:wp.vec3f,bh:wp.vec3f,bl:wp.vec3f):
    value=wp.vec2f(wp.float32(0.))
    for i in range(3):value=nadd(value,nmul(wp.vec2f(ah[i],al[i]),wp.vec2f(bh[i],bl[i])))
    return value


@wp.func
def metric_pair(t0:wp.vec3f,t1:wp.vec3f,d0:wp.vec3f,l0:wp.vec3f,d1:wp.vec3f,l1:wp.vec3f):
    zero=wp.vec3f(wp.float32(0.));half=wp.vec2f(wp.float32(.5),wp.float32(0.))
    a=nadd(ndot(t0,zero,d0,l0),nmul(ndot(d0,l0,d0,l0),half))
    b=nadd(ndot(t1,zero,d1,l1),nmul(ndot(d1,l1,d1,l1),half))
    c=nadd(nadd(ndot(t0,zero,d1,l1),ndot(t1,zero,d0,l0)),ndot(d0,l0,d1,l1))
    return wp.vec3f(a[0]+a[1],b[0]+b[1],c[0]+c[1])
