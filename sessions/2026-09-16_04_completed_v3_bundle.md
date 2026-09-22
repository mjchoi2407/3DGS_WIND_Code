# v3 완료 결과 취합

## 현재 상태

- 확인일: 2026-09-16. 사용자 요청으로 진행 중 메인 v3 controller/소유 worker 종료 후 완료 시험만 취합했다.
- `teacher_precision_completed_bundle.py` 추가. 종료코드0·passed 결과를 가진 시험만 원시 배열을 복사하고, 미완료 prefix는 제외 목록과 실패 로그로 분리한다.
- 누락된 최종 summary/phase/step/반복 CSV는 보존 원시값에서 작성한다. 실제 GPU 계산은 재실행하지 않는다.
- 동결 cases/runtime/variants, 실제 dtype, A/P·checkpoint·forcing·소스 hash, 로그·telemetry·재현 명령을 보존한다.
- 교차 입력/배열 대응과 실제 차이를 기록하되 회귀/교차 예산은 미정, 생산/teacher 적격성은 승격하지 않는다.
- 결과·ZIP 전체 CRC/SHA 검증 근거는 [실험 인계](../../experiments/sessions/2026-09-16_04_completed_v3_bundle.md)를 따른다.
- 다음: 완료 데이터 분석. HL01 mixed 완주와 서브컴 정체 원인은 미완료다.
- 기존 사용자 변경 보존. commit/push 없음.
- 업로드512MB 제한 대응: `teacher_precision_compact_bundle.py`로 SHA-256 동일 파일을 한 번만 저장하고 큰 JSON만 ZIP LZMA 압축한다. 수치 배열/정밀도는 그대로이며 중복 관계표와 표준 Python 복원기를 포함한다. 원본 큰 ZIP은 보존한다.
