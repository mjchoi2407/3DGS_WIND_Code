# 고부하 연산별 정밀도 벤치마크

## 현재 상태

2026-09-20. 스크립트 준비 완료·GPU 사용자 실행 대기.
- 기존 동결 runtime을 보존하고 진단 사본에서 R64 첫 substep의 Newton snapshot을 추출한다.
- P 분해/적용, HVP·질량, subtract_basis를 기존 FP64/FP32 구현으로 동일 작업량 비교한다.
- 부모 CUDA 초기화 없음, graph20호출·6교대쌍, 변환 포함/제외·출력 차이·raw/hash/환경 기록.
- GPU 미검증. CPU unittest2개·구문 검사·실제 입력 준비/hash 검증 통과.
- 원본 솔버/운영 선택/허용오차는 수정하지 않았다. 전체 teacher 가속·정확도 비중 판단은 미완료.
- [실행/측정 계약](../../experiments/R1_teacher_velocity_reset/timestep_search/precision_v3/highload_fixed_work.md).
