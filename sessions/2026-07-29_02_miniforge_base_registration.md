# 2026-07-29 Miniforge base 설치 및 기존 환경 등록

## 배경

이전에 `conda-pack`으로 복원한 `gof`, `sibr`, `sugar` runtime은 정상이었지만 Conda base가 없어 `conda env list`와 이름 기반 `conda activate`를 사용할 수 없었다. 사용자가 최종 디스크 이전을 완료한 뒤 Conda 관리 기능 복구를 요청했다.

## 설치

- 공식 배포처: `conda-forge/miniforge`
- 버전: `Miniforge3 26.3.2-2`
- 설치 파일: `Miniforge3-26.3.2-2-Linux-x86_64.sh`
- 설치 위치: `/home/choi/miniforge3`
- 설치 크기: 약 `1003M`
- Conda: `26.3.2`
- base Python: `3.13.13`

공식 `.sha256` 파일로 설치 파일을 검증했다.

```text
42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94
Miniforge3-26.3.2-2-Linux-x86_64.sh: OK
```

## 설정

`conda init bash`로 `/home/choi/.bashrc`에 Miniforge shell hook을 추가했다.

`/home/choi/.condarc`의 주요 설정:

```yaml
auto_activate: false
envs_dirs:
  - /home/choi/conda-envs/wind3dgs
```

- 새 셸에서 `conda` 명령을 사용할 수 있다.
- base는 자동 활성화하지 않는다.
- 기존 packed environment directory를 이름으로 탐색한다.
- 기존 environment 자체의 package와 파일은 변경하지 않았다.

## `conda env list`

다음 네 환경이 정상 표시된다.

```text
gof      /home/choi/conda-envs/wind3dgs/gof
sibr     /home/choi/conda-envs/wind3dgs/sibr
sugar    /home/choi/conda-envs/wind3dgs/sugar
base     /home/choi/miniforge3
```

## 활성화 및 runtime 검증

새 interactive Bash에서 이름 기반 활성화를 확인했다.

### `gof`

- `conda activate gof`: 통과
- `CONDA_PREFIX=/home/choi/conda-envs/wind3dgs/gof`
- GPU: `NVIDIA GeForce GTX 1080 Ti`
- `simple_knn.distCUDA2` 실제 CUDA 연산: 통과

### `sibr`

- `conda activate sibr`: 통과
- `CONDA_PREFIX=/home/choi/conda-envs/wind3dgs/sibr`
- `opencv 4.12.0`: import 통과
- `numpy 2.5.0`: import 통과

### `sugar`

- `conda activate sugar`: 통과
- `CONDA_PREFIX=/home/choi/conda-envs/wind3dgs/sugar`
- GPU: `NVIDIA GeForce GTX 1080 Ti`
- `simple_knn.distCUDA2` 실제 CUDA 연산: 통과
- `nvdiffrast.torch`: import 통과

## 사용

현재 열려 있는 기존 terminal에서는 다음 중 하나를 수행해야 한다.

```bash
source ~/.bashrc
```

또는 terminal을 닫고 새로 연다.

이후 다음 명령을 사용할 수 있다.

```bash
conda env list
conda activate gof
conda activate sibr
conda activate sugar
```

## 변경 사항

- 생성: `/home/choi/miniforge3`
- 생성: `/home/choi/.condarc`
- 수정: `/home/choi/.bashrc`
- 추가: `code/sessions/2026-07-29_02_miniforge_base_registration.md`
- 기존 `gof`, `sibr`, `sugar` 환경과 프로젝트 코드는 변경하지 않았다.
