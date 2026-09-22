/* cuDSS가 만드는 cuBLAS handle에 초기 GPU workspace를 제공한다.
 * 새 비교 subprocess에만 LD_PRELOAD한다. CUDA graph 내부 malloc을 방지한다.
 * CUDA/cuBLAS 공개 C API를 사용하며 수치 알고리즘이나 결과는 변경하지 않는다.
 */
#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <pthread.h>

typedef void *handle_t;
typedef int (*create_fn)(handle_t *);
typedef int (*destroy_fn)(handle_t);
typedef int (*stream_fn)(handle_t,void *);
typedef int (*workspace_fn)(handle_t,void *,size_t);
typedef int (*alloc_fn)(uint64_t *,size_t);
typedef int (*free_fn)(uint64_t);
static create_fn real_create;
static destroy_fn real_destroy;
static stream_fn real_stream;
static workspace_fn real_workspace;
static alloc_fn device_alloc;
static free_fn device_free;
static pthread_once_t init_once=PTHREAD_ONCE_INIT;
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static struct {handle_t handle;uint64_t workspace;} entries[256];
static const size_t workspace_bytes=32u*1024u*1024u;
static int initialized;

static void initialize(void) {
    void *blas=dlopen("libcublas.so.12",RTLD_NOW|RTLD_LOCAL);
    void *driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if (!blas || !driver) return;
    real_create=(create_fn)dlsym(blas,"cublasCreate_v2");
    real_destroy=(destroy_fn)dlsym(blas,"cublasDestroy_v2");
    real_stream=(stream_fn)dlsym(blas,"cublasSetStream_v2");
    real_workspace=(workspace_fn)dlsym(blas,"cublasSetWorkspace_v2");
    device_alloc=(alloc_fn)dlsym(driver,"cuMemAlloc_v2");
    device_free=(free_fn)dlsym(driver,"cuMemFree_v2");
    initialized=real_create&&real_destroy&&real_stream&&real_workspace&&device_alloc&&device_free;
}
static uint64_t lookup(handle_t handle) {
    uint64_t result=0;
    pthread_mutex_lock(&lock);
    for(int i=0;i<256;i++)if(entries[i].handle==handle){result=entries[i].workspace;break;}
    pthread_mutex_unlock(&lock);
    return result;
}
int cublasCreate_v2(handle_t *handle) {
    pthread_once(&init_once,initialize);
    if(!initialized)return 14;
    int result=real_create(handle);if(result)return result;
    uint64_t workspace=0;
    if(device_alloc(&workspace,workspace_bytes)){real_destroy(*handle);*handle=NULL;return 3;}
    int slot=-1;
    pthread_mutex_lock(&lock);
    for(int i=0;i<256;i++)if(!entries[i].handle){slot=i;entries[i].handle=*handle;entries[i].workspace=workspace;break;}
    pthread_mutex_unlock(&lock);
    if(slot<0){device_free(workspace);real_destroy(*handle);*handle=NULL;return 3;}
    result=real_workspace(*handle,(void *)(uintptr_t)workspace,workspace_bytes);
    if(result){
        pthread_mutex_lock(&lock);entries[slot].handle=NULL;entries[slot].workspace=0;pthread_mutex_unlock(&lock);
        real_destroy(*handle);device_free(workspace);*handle=NULL;
    }
    return result;
}
int cublasSetStream_v2(handle_t handle,void *stream) {
    pthread_once(&init_once,initialize);if(!initialized)return 14;
    int result=real_stream(handle,stream);if(result)return result;
    uint64_t workspace=lookup(handle);
    return workspace?real_workspace(handle,(void *)(uintptr_t)workspace,workspace_bytes):0;
}
int cublasSetWorkspace_v2(handle_t handle,void *workspace,size_t bytes) {
    pthread_once(&init_once,initialize);if(!initialized)return 14;
    /* 명시적으로 준 workspace는 존중한다. null reset은 초기 workspace를 유지한다. */
    uint64_t initial=lookup(handle);
    if(!workspace && initial)return real_workspace(handle,(void *)(uintptr_t)initial,workspace_bytes);
    return real_workspace(handle,workspace,bytes);
}
int cublasDestroy_v2(handle_t handle) {
    pthread_once(&init_once,initialize);if(!initialized)return 14;
    uint64_t workspace=lookup(handle);
    int result=real_destroy(handle);if(result)return result;
    pthread_mutex_lock(&lock);
    for(int i=0;i<256;i++)if(entries[i].handle==handle){entries[i].handle=NULL;entries[i].workspace=0;break;}
    pthread_mutex_unlock(&lock);
    if(workspace)device_free(workspace);
    return 0;
}
