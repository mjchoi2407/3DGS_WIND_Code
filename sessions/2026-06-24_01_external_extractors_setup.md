# 2026-06-24 external extractors setup

## Context

The user decided not to reimplement mesh extraction because Mesh Extraction is not the paper contribution.
The plan is to use official GOF and SuGaR code as external extractors and build Wind3DGS adapters around their outputs.

## Decisions

- Keep official extractor repositories outside the tracked project code under `external/`.
- Add `external/` to `.gitignore` so third-party code and generated external outputs do not enter the Wind3DGS repository.
- Treat GOF as the main raw proxy extraction path.
- Treat SuGaR as a comparison baseline.
- Defer environment installation because the current shell does not expose `conda`/`mamba`, and official GOF/SuGaR environments require different Python/PyTorch/CUDA stacks from the project `.venv`.

## Local Checkouts

- `external/gaussian-opacity-fields`
  - source: `https://github.com/autonomousvision/gaussian-opacity-fields.git`
  - local commit: `5245b20e5d11acd6d1ff5af4b890dc2bedd99693`
  - relevant scripts: `extract_mesh.py`, `extract_mesh_tsdf.py`, `train.py`
  - official environment target: Python 3.8, PyTorch 1.12.1, CUDA 11.3-era setup plus `diff-gaussian-rasterization`, `simple-knn`, and tetra triangulation dependencies

- `external/SuGaR`
  - source: `https://github.com/Anttwo/SuGaR.git`
  - local commit: `7c10c4ae4a267dece512f5c7f40ed212a0a2ab44`
  - relevant scripts: `train_full_pipeline.py`, `extract_mesh.py`, `train.py`
  - official environment target: Python 3.9, PyTorch 2.0.1, CUDA 11.8-era setup plus 3DGS rasterizer dependencies and optional Nvdiffrast

## Environment Notes

- In the current shell, `conda` and `mamba` were not found.
- CUDA toolkit exists at `/usr/local/cuda-12.6`, but `nvcc` is not on PATH unless `CUDA_HOME` and PATH are exported.
- The project `.venv` currently uses PyTorch `2.11.0+cu126`; this should not be mixed with GOF/SuGaR official dependency stacks.

## Verification

- `git -C external/gaussian-opacity-fields status --short` returned clean.
- `git -C external/SuGaR status --short` returned clean.
- `git check-ignore -v external/gaussian-opacity-fields external/SuGaR` confirmed both paths are ignored by `.gitignore`.
- `/usr/local/cuda-12.6/bin/nvcc --version` reports CUDA 12.6.

## Next

- Decide whether to install Miniconda/Mambaforge or use an existing conda setup outside this shell.
- After conda is available, create separate environments for GOF and SuGaR rather than installing their dependencies into the Wind3DGS `.venv`.
- Implement `code/wind3dgs/extraction/` adapters that call external extractor commands and ingest their produced mesh files into a shared `RawProxyMesh` format.
