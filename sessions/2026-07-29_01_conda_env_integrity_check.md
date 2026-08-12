# 2026-07-29 Conda 환경 이전 무결성 점검

## 배경

이전에 `/mnt/h/.../external/miniforge3/envs/`에서 `conda-pack`으로 옮긴 다음 환경들이 현재 `/home/choi/conda-envs/wind3dgs/`에서 손상 없이 동작하는지 확인했다.

- `gof`
- `sibr`
- `sugar`

환경이나 패키지는 변경하지 않고 읽기·실행 검증만 수행했다.

## 결과 요약

| 환경 | Python/prefix | `pip check` | 핵심 runtime | 판정 |
| --- | --- | --- | --- | --- |
| `gof` | 정상 | 통과 | CUDA custom extension 실제 연산 통과 | 정상 |
| `sibr` | 정상 | 통과 | OpenCV import, CMake/pkg-config, viewer linker 통과 | 정상 |
| `sugar` | 정상 | `triton`의 `cmake`, `lit` 경고 | CUDA custom extension 실제 연산 및 `nvdiffrast` import 통과 | 이전 경로는 정상, 경미한 기존 의존성 불일치 |

## Prefix와 Python

- `gof`
  - Python `3.8.20`
  - prefix `/home/choi/conda-envs/wind3dgs/gof`
- `sibr`
  - Python `3.13.14`
  - prefix `/home/choi/conda-envs/wind3dgs/sibr`
- `sugar`
  - Python `3.9.18`
  - prefix `/home/choi/conda-envs/wind3dgs/sugar`

세 환경 모두 `sys.executable`, `sys.prefix`, `sys.base_prefix`가 현재 `/home/choi/conda-envs/wind3dgs/<env>`를 정확히 가리켰다.

## 이전 경로 검사

- `bin/`의 활성 스크립트 shebang에서 `/mnt/h`를 찾지 못했다.
- Python `.pth`와 `.egg-link`에서 이전 `/mnt/h` prefix를 찾지 못했다.
- 이전 prefix는 각 환경의 `bin/conda-unpack` 내부 replacement history에만 남아 있었다. 이는 relocation 당시 사용된 치환 기록이며 활성 runtime 경로가 아니다.
- `sugar`의 editable CUDA module은 현재 workspace 아래를 가리킨다.
  - `/home/choi/projects/2026_paper_work/Wind_Deformable_3DGS/external/SuGaR/...`
- GOF와 SuGaR의 custom `.so`는 환경 및 Torch library 경로를 적용했을 때 linker dependency 누락이 없었다.

## 패키지와 CUDA 검증

### GOF

- `pip check`: `No broken requirements found.`
- `numpy 1.24.4`
- `torch 1.12.1+cu113`
- `diff_gaussian_rasterization`: import 통과
- `simple_knn._C`: import 통과
- GTX 1080 Ti에서 `simple_knn.distCUDA2` 실제 CUDA 연산과 synchronize 통과

### SIBR

- `pip check`: `No broken requirements found.`
- `numpy 2.5.0`
- `opencv 4.12.0`
- `cmake 4.3.4`
- `pkg-config 0.29.2`
- 설치된 `SIBR_gaussianViewer_app`, `SIBR_remoteGaussian_app`에 SIBR install library와 Conda library 경로를 적용한 `ldd` 검사에서 누락 없음

빌드 트리 안의 실행 파일만 Conda library 경로로 검사하면 프로젝트 자체 `libsibr_*.so`가 `not found`로 표시되지만, 실제 설치본의 `install/bin`과 `install/lib`를 포함한 정상 실행 경로에서는 모두 해소됐다. 이는 Conda 이전 손상이 아니다.

### SuGaR

- `numpy 1.26.2`
- `torch 2.0.1`, CUDA build `11.8`
- `diff_gaussian_rasterization`: import 통과
- `simple_knn._C`: import 통과
- `nvdiffrast.torch`: import 통과
- GTX 1080 Ti에서 `simple_knn.distCUDA2` 실제 CUDA 연산과 synchronize 통과

`pip check`는 다음을 보고했다.

```text
triton 2.0.0 requires cmake, which is not installed.
triton 2.0.0 requires lit, which is not installed.
```

확인 결과:

- Conda 패키지의 `cmake 3.29.4` 실행 파일은 정상적으로 존재한다.
- pip Python distribution으로서의 `cmake`가 없기 때문에 `pip check`가 이를 충족된 것으로 인식하지 못한다.
- `lit` 실행 파일과 Python distribution은 없다.
- `triton 2.0.0` 자체 import는 통과했다.
- 프로젝트가 사용하는 CUDA custom extension과 `nvdiffrast` runtime은 정상이다.

따라서 이는 이전 prefix 파손이 아니라 Conda와 pip의 혼합 의존성 메타데이터 불일치 및 `lit` 누락이다. 현재 확인한 GOF/SuGaR 작업에는 장애가 없지만, 향후 Triton kernel 개발·빌드 경로를 직접 사용한다면 `lit` 설치 또는 환경 재검증이 필요하다.

## 참고

`pip check` 중 `/home/choi/.cache/pip`가 쓰기 불가능하여 cache를 비활성화했다는 warning이 출력됐다. 이번 검증 실행 환경의 쓰기 제한에 따른 것으로 패키지 무결성 문제는 아니다.

## `conda env list` 실패 원인

현재 일반 셸과 interactive Bash 모두에서 `conda` 명령을 찾지 못했다.

```text
conda: command not found
```

확인 결과:

- `/home/choi/miniforge3`, `/home/choi/mambaforge`, `/home/choi/anaconda3`에 Conda 본체가 없다.
- 복원된 `gof`, `sibr`, `sugar` 환경의 `bin/`에도 `conda` 실행 파일은 없다.
- `.bashrc`, `.profile` 등에 새 Conda base를 초기화한 설정이 없다.
- `~/.conda/environments.txt`에는 이전 H 경로의 `sibr` 한 줄만 남아 있다.

이전 작업에서는 대형 custom 환경 세 개만 `conda-pack`으로 복원하고, 기존 `/mnt/h/.../external/miniforge3` base 설치는 새 workspace에 복사하지 않았다. 따라서 각 환경의 Python과 package runtime은 정상이지만, `conda env list` 명령을 제공하고 환경 목록을 관리할 Conda base가 없는 상태다.

이는 환경 자체의 손상이 아니다. 현재처럼 절대 경로의 Python을 직접 실행할 수 있다.

```bash
/home/choi/conda-envs/wind3dgs/gof/bin/python
/home/choi/conda-envs/wind3dgs/sibr/bin/python
/home/choi/conda-envs/wind3dgs/sugar/bin/python
```

향후 `conda env list`와 `conda activate`를 다시 사용하려면 건강한 최종 디스크로 WSL을 옮긴 뒤 Miniforge base를 별도로 설치하고, shell init과 `envs_dirs=/home/choi/conda-envs/wind3dgs`를 설정하는 것이 적합하다. 디스크 이전 직전인 현재는 정상 동작 중인 환경을 불필요하게 변경하지 않는다.

## 변경 사항

- Conda 환경, 패키지, 프로젝트 코드는 변경하지 않았다.
- 이 검증 기록만 추가했다.
