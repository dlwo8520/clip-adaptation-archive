# CLIP 연구 아카이브 안내

종료된 탐색 연구이며, 논문/SOTA 성과로 표현하지 않았습니다. `README.md`는 영어 연구 개요, `docs/EXPERIMENTS.md`는 실험별 코드와 문서 연결, `docs/RESULTS.md`는 실제 결과와 근거 수준입니다.

원래 `CLIP/코드`, `CLIP/실험완료`, `CLIP/그래프` 구조를 유지했습니다. 완료 폴더 이름만으로 실행 완료를 판단하지 않았습니다. 0-byte 결과는 수치 근거로 쓰지 않았고, 독립 그래프 CSV만 있는 결과는 raw/config가 복원되지 않은 기록으로 표시했습니다. 동일 결과 복사본을 추가 seed로 세지 않습니다.

큰 표는 `python tools/restore_tables.py`로 복원합니다. `python tools/rebuild_indexes.py`로 수치 목록을 재생성하고 `python tools/validate_export.py`로 보존 검사를 합니다. GPU 재학습은 이번 정리 작업에서 실행하지 않았습니다. 교사 checkpoint, dataset, 환경은 별도로 준비해야 합니다. 개인 경로 수정은 복사본에만 적용했고 원본은 보존했습니다.

2026-10-13 KST에 [GitHub private 저장소](https://github.com/dlwo8520/clip-adaptation-archive)로 업로드하고 main 커밋 일치를 확인했습니다. 후속 변경은 검토·commit 후 `git push origin main`으로 반영합니다. `docs/UPLOAD.md`에 업로드 기록과 확인 명령이 있습니다.
