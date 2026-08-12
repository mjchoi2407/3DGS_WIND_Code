# 2026-08-05 01 gsplat와 GTX 1080 Ti 호환 경로 확인

## 배경

수정된 연구 계획을 검토하는 과정에서 root `.venv`의 `gsplat 1.5.3`을 GTX 1080 Ti에서 사용할 수 없다고 판단했으나, 사용자가 기존 1080 Ti용 커스텀 설정의 존재 여부를 다시 확인해 달라고 요청했다. 패키지 `gsplat`과 Graphdeco 계열 Gaussian splatting rasterizer를 구분해 현재 설치 상태, 빌드 산출물, 실제 CUDA 실행을 점검했다.

## 결론

- 1080 Ti용 커스텀 CUDA rasterizer는 존재하며 현재도 동작한다.
- 다만 이것은 Python 패키지 `gsplat==1.5.3`이 아니라 GOF/SuGaR의 `diff_gaussian_rasterization`과 Graphdeco SIBR의 `CudaRasterizer`다.
- root `.venv`의 `gsplat 1.5.3`은 stock pip 설치이고, 성공한 `gsplat_cuda.so`가 없다.
- 현재 root `.venv`의 PyTorch `2.12.1+cu130`도 `sm_61`을 포함하지 않으므로 GTX 1080 Ti에서 최소 CUDA tensor 연산부터 실패한다.

## 설치 및 소스 확인

### root `.venv`의 gsplat

- Python: `3.12.3`
- PyTorch: `2.12.1+cu130`
- `torch.version.cuda`: `13.0`
- gsplat: `1.5.3`
- 설치 위치: `.venv/lib/python3.12/site-packages/gsplat`
- `direct_url.json`이 없고 `INSTALLER`는 `pip`다.
- `gsplat-1.5.3.dist-info/RECORD`에 기록된 패키지 파일 1,729개의 hash와 현재 파일을 비교했다.
  - mismatch: `0`
  - missing: `0`
- 따라서 설치 뒤 `site-packages/gsplat`을 직접 패치한 흔적은 없다.
- stock source의 projection kernel에는 `cooperative_groups::labeled_partition` 호출이 그대로 남아 있다.

### gsplat JIT cache

기존 cache는 다음 위치에 일부 object file만 남아 있다.

```text
/home/choi/.cache/torch_extensions/py312_cu126/gsplat_cuda
```

- `build.ninja`는 `compute_61/sm_61` target을 포함한다.
- source/include 경로는 이전 `/mnt/h/...` workspace를 가리킨다.
- 여러 `.cuda.o`는 있으나 최종 `gsplat_cuda.so`는 없다.
- 이는 1080 Ti target으로 빌드를 시도한 흔적이지 성공한 커스텀 gsplat 설치가 아니다.

### 프로젝트 방어 코드

`wind3dgs.m01_static_3dgs_io.render_static_baseline.ensure_gsplat_cuda_ready()`는 CC 7.0 미만이면 gsplat JIT를 시작하지 않고 `cpu_debug` 사용을 안내한다. `check_gsplat_env.py`도 `TORCH_CUDA_ARCH_LIST`를 설정하지 않고 현재 값만 출력한다.

## 현재 GPU 실행 확인

GPU와 driver:

```text
NVIDIA GeForce GTX 1080 Ti
Compute Capability 6.1
VRAM 11264 MiB
Driver 582.28
```

root `.venv`에서 실제 CUDA tensor 연산 결과:

```text
torch 2.12.1+cu130 cuda 13.0
device NVIDIA GeForce GTX 1080 Ti capability (6, 1)
cuda_tensor_failed AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

현재 wheel이 보고한 지원 target은 `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`이며 `sm_61`이 없다. `torch.cuda.is_available() == True`만으로 실행 가능하다고 판단하면 안 된다.

## 실제 동작하는 1080 Ti 커스텀 경로

### GOF

- 환경: `/home/choi/conda-envs/wind3dgs/gof`
- PyTorch: `1.12.1+cu113`
- extension: `diff_gaussian_rasterization`
- 기존 설치 시 `TORCH_CUDA_ARCH_LIST=6.1`로 빌드했다.
- 현재 GTX 1080 Ti에서 1-Gaussian, 32 x 32 forward smoke를 다시 실행했다.

```text
torch 1.12.1+cu113 cuda 11.3
device NVIDIA GeForce GTX 1080 Ti capability (6, 1)
gof_raster_ok (9, 32, 32) (1,) 111.6875 [6] True
```

즉 현재 시점에도 실제 CUDA image tensor를 유한값으로 생성한다.

### SuGaR

SuGaR 환경에도 `sm_61`로 빌드된 `diff_gaussian_rasterization`과 `simple_knn` extension이 존재한다. 기존 기록상 최소 raster smoke가 성공했다. 이번 점검에서는 주 renderer 후보인 GOF를 실제 재실행했다.

### Graphdeco SIBR

SIBR의 로컬 `CudaRasterizer/CMakeLists.txt`에는 다음 target이 있다.

```cmake
set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "61;70;75;86")
```

`GaussianView.cpp`도 최소 guard를 CC 6.0으로 낮추고 CC 6.x에서는 fast culling을 비활성화하도록 패치돼 있다. 이 경로 역시 `gsplat` 패키지와 별개의 C++/CUDA viewer rasterizer다.

## 계획 반영

- GTX 1080 Ti 로컬 V0/V1의 주 renderer는 현재 실동작이 재확인된 GOF `diff_gaussian_rasterization`으로 둔다.
- Graphdeco SIBR는 static interactive inspection에 사용한다.
- `gsplat==1.5.3`은 generic Gaussian splatting과 이름을 구분하며, 현재 root `.venv`의 실행 backend로 간주하지 않는다.
- gsplat 자체가 반드시 필요하면 별도 포팅 작업으로 분리한다.
  1. `sm_61`을 지원하는 PyTorch/CUDA 환경 고정
  2. `labeled_partition`의 CC 6.x fallback 구현
  3. 별도 `TORCH_EXTENSIONS_DIR`에서 격리 빌드
  4. 1-Gaussian forward/backward와 실제 asset render 검증
- 새 dynamics 경로에서는 GOF rasterizer가 per-frame mean/covariance 갱신을 실제 반영하는 1-frame/2-frame smoke를 다음 gate로 둔다.

## 변경 사항

- 코드, 패키지, conda 환경은 변경하지 않았다.
- 이 진단 기록만 추가했다.
