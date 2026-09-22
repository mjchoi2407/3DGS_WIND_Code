# GPU 자동 선택과 기존 세 씬 비교

## 현재 상태

확인 기준: 2026-09-17. 구현·제한 검증 완료, 본 실행은 사용자 실행 대기.

기존 MixedLinear만 frozen runtime에 연결했다. 원본 stepper와 v3 full/summary는 보존한다. 프레임 경계에서 hi/lo 상태를 보존하여 Graph를 재생성하며 전환 비용을 기록한다.

1080Ti 일반 바람 M1,5070 일반 바람 M2,무풍과1080Ti 알려진 HL01은 R64다.
5070 성공 API13000과 다른 환경은 실행을 보류한다. Graph 장애 복구는 미완료다.
1080Ti 세 씬 각3개의1프레임 smoke에서 R64/R64/M1 선택과 기존 독립 검산을 통과했다.
GPU 동시 부하가 있어 이 시간을 가속 근거로 쓰지 않는다. CPU 정책 경계 검사4개 통과.
원래 정확도·Gauss retry·필수 상태 저장을 유지했다. 새 solver/full FP32/fresh 개발은 재개하지 않았다.
학습 적격성·생산 인증은 false이고 장기 궤적 및 전체 가속은 미검증이다.

근거·명령: [실행 안내](../../experiments/R1_teacher_velocity_reset/timestep_search/gpu_auto_scenes/README.md).
원시 경로와 검사값은 같은 폴더 `validation.json`을 따른다. Commit/push하지 않았다.
