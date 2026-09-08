# Teacher 개발용 샘플 추출·검증

2026-09-08, Wind3DGS code-side. 사용자의 “샘플 학습데이터를 뽑고 검증하는 것 까지 연속적으로” 진행 지시를 따른다.
[선형 공간 검사](2026-09-08_05_teacher_shell_linear_spatial_design.md)의 계산은 완료했지만 물리 기준은 실패했다.

용도를 비동기 질문했고, 별도 답변 없이 계속 진행하는 동안 개발용 소량 샘플을 먼저 만든다는 가정을 알렸다.
본 학습용 데이터로 채택하거나 물리 기준을 완화하지 않는다. `training_eligible=false`, development 전용을
loader에서도 강제한다. R1/R2 완결이나 연구 방법 전환을 확정하지 않는다.

## 기능·입출력·검증

기존 Newton/Registry/trajectory/probe를 재사용해 pulse, step-on/off, aero-off displaced free decay를 각 1초 실행한다.
CPU, n=8, 60fps, substeps=16, iterations=10, native 물성·M_ref=0.1kg, 25 probe다.
각 사례 60 interval/61 state, 12 interval/13 state window 5개씩 총 15개를 만든다.

- `teacher/sample_dataset.py`: 완료 raw/probe 쌍 → NPZ window·static measure·registry·manifest, 원본 대조와 NumPy batch loader.
- `evaluation/teacher_sample_dataset_check.py`: 세 원본 생성·probe 추출·동일 CPU 재생·건전성·dataset·batch 검증.
- `scripts/generate_teacher_sample_dataset.sh`: 새 output/dataset 경로, 프로젝트 내부 Warp cache와 한국어 로그.
- `tests/test_teacher_sample_dataset.py`: 실제 작은 CPU 원본, 시간/단위/분모·source group·hash 손상·미완료·무단 학습 채택 거부.

선언한 window를 누락하거나 case 경계를 넘어 섞지 않는다. 모든 source-object 파생물은 development 한 group이다.
원본 전체 상태·force/work는 기존 raw artifact에 보존하고 sample은 common-probe response/input/work만 제공한다.
Writer 완료 발행 전 원본 값 대조, batch [4,4,4,3]와 NumPy-only 재로드를 요구한다.
Dependency 설치·기존 runtime source 변경은 없다. 새 샘플에는 independent GS 입력이 없으며 R2 학습은 실행하지 않는다.

실제 테스트·결과·경로와 Git 상태는 아래 완료 기록을 따른다.

## 완료 결과

[개발용 샘플 결과](../../experiments/R1_teacher_sample_dataset/README.md): 3개 시계열·183 state·15 window를 생성했다.
48.370초, CPU replay 세 개 모두 x/v/force/work 최대 차이 0, pin drift=0·guard=0이며
free-decay 외력 work=0이다. Batch 크기 [4,4,4,3], 첫 정답 shape [4,13,25,3]을 검증했다.

Dataset은 `experiments/artifacts/datasets/teacher_samples/20260908_sample_v1/`의 20개 파일·125,293byte다.
원본은 `experiments/artifacts/runs/teacher_sample_dataset/20260908_sample_v1/`의 93개 inventory·1,150,138byte다.
전체 byte/hash·producer source 24개·모든 sample 값을 추가 독립 프로세스에서 검산했다.
Manifest SHA-256은 `1ab229207fa8ec60c282aa433c49d7a5f8a288deb1575688794b72b2afdae8f5`다.

새 검사 11개(12.642초)가 통과했다. 실제 CPU 원본→추출→원본 대조, incomplete/손상/단위/시각/누락·중복/잘못된
split·재해시된 정답/geometry 변경 검출과 Torch/Newton/Warp 차단 subprocess의 NumPy-only loading을 포함한다.
완료 manifest는 원본 대조까지 통과한 뒤에만 발행하며 해당 순서를 failure injection으로 확인했다.

```bash
PYTHONPATH=code:code/tests WARP_CACHE_PATH=code/outputs/warp-cache \
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest test_teacher_sample_dataset -v
```

관련 공간/회귀 228개와 합쳐 서로 다른 239개 검사를 실행했다. 전체 repository test discovery는 실행하지 않았다.
원본·dataset·compact evidence·한국어 결과 그래프와 재현 명령을 experiments에 보존했다.
기존 Newton native 물성을 사용한 개발용 자료이며 새 shell Teacher 채택이나 본 학습 데이터 인증은 아니다.
Source object 한 개의 모든 파생물을 development에 두고 학습 적격성 false를 유지했다.
Code/experiments 파일만 추가·갱신했고 root/ideas 기존 변경은 유지했다. Stage·commit·push·fetch는 하지 않았다.

## 최종 QA

시작 시 비ignored 파일 629개를 기준으로 기존 파일 5개(code README/index/공간 note, experiments README/index)만
갱신했고 새 파일 33개를 추가했다. 삭제는 없고 root/ideas status는 시작과 동일하다.
문서 링크 117개·전체 저장소 diff check·새 텍스트 공백/개인 경로/secret 패턴 검사와 runtime snapshot 보존을 확인했다.
Raw/dataset Git ignore를 확인했고 packaging/import 4개를 마지막으로 재실행해 통과했다(기존 239개에 포함).
결과 그래프의 한국어 글꼴·축·세 시계열 표시도 직접 확인했다.
