# AGENTS.md — xsec_alpha 운영 규칙 (2026-10-04)

Codex, Claude 등 이 저장소를 수정하는 모든 에이전트에 적용한다. 상세 규칙은 `CLAUDE.md`, 결정 기록은 `/home/soccz/22tb/DECISIONS.md` §8에 있다.

## 1. 작업 트리가 곧 운영이다
- systemd 서비스(alpha·measure·retrain·telegram-retry, user: experiment-supervisor·backup)는 이 디렉터리 코드를 그대로 실행한다. **파일을 저장하는 순간 배포된다.**
- 운영 코드를 바꾸기 전에 다음을 확인한다:
  - import 스모크: `python -c 'import scripts.fetch_and_rank, scripts.supervise_experiment, utils.dashboard_publish, utils.forecast_audit, utils.rotation_pilot'`
  - 관련 pytest
- 배포 시각:
  - 신호 런(KST 02/08/14/20시) 직후 +50분 무렵에 한다.
  - 감독 틱(매 10분 :X0:00 UTC)과 겹치지 않게 한다.

## 2. 336회 국면 전환 파일럿 보호 (등록 2026-09-30, 종료 약 2026-12-24, hard review 2026-12-25 21:00 KST)
- 동결 소스 6개는 수정 금지이며 쓰기 권한도 제거돼 있다: `utils/rotation_pilot.py`, `utils/rotation_research.py`, `utils/prospective.py`, `utils/regime_observer.py`, `utils/eval_metrics.py`, `scripts/rotation_pilot.py`.
- 패키지 버전이 고정돼 있다: numpy 1.26.4, pandas 2.1.4, scipy 1.11.4(`~/.local`). `pip install --user`·`pip uninstall`을 금지한다.
- 숨은 결합: `scripts/supervise_experiment.py`가 `scripts.fetch_and_rank`를 try 없이 import한다. 따라서 fetch_and_rank, config, utils.bitget, utils.logger, utils.run_lock, utils.model_release 중 하나라도 import가 깨지면 alpha와 파일럿 캡처가 함께 멈춘다.

## 3. 모델 동결 (2026-10-04 결정)
- 주간 재학습은 `--dry-run`으로만 돌린다(학습·측정만 하고 승격하지 않음). 6h/12h 운영 모델 교체는 사용자 결정으로만 한다.
- 새 모델은 운영 모델을 덮어쓰지 않는다. 섀도 후보(모델 동물원)로 추가하고 `docs/prereg/` 사전등록을 따른다.

## 4. 사전등록 존중
- README §14-bis 블록(450~476행)은 수정하지 않는다.
- 판정·편차·개정은 블록 밖 별도 절과 DECISIONS.md에 기록한다. 결과를 본 뒤 기준을 바꾸지 않는다.
- **기한과 기본값은 반드시 `docs/deadlines.json`에 등록한다.** 문장으로만 적지 않는다. `scripts/check_deadlines.py`가 매일 09:25 KST(`xsec-deadlines.timer`)에 점검한다. 7일·1일 전에 알림을 보내고, 기한이 지나면 기본값을 기록하고 텔레그램으로 알린다. §14-bis 데드맨이 문장으로만 있어서 판정이 33일 늦어진 일이 있었고, 그 재발을 막기 위한 규칙이다.

## 5. 부하·임시 파일·비밀
- 호스트는 부하에 민감하다(바쁜 HDD, inotify 여유 거의 없음). 무거운 분석은 백업 스냅샷(`/home/soccz/22tb/backups/xsec_db/`)에서 `nice -n 19 ionice -c3`로 돌리고, 병렬 작업은 최소화한다.
- 루트 `/tmp`를 쓰지 않는다. `$TMPDIR`(`/home/soccz/22tb/tmp`) 또는 `output/` 아래를 쓴다.
- `.env`, 텔레그램 토큰, sudo 비밀번호를 명령줄·로그·대화에 쓰지 않는다.
