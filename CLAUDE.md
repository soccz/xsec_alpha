# CLAUDE.md — xsec_alpha

gan_t에서 실패한 코인 예측의 2단계. 절대 수익률 예측 → 크로스섹션 상대 랭킹.
글로벌 `~/.claude/CLAUDE.md` + gan_t 규칙은 여기서 반복하지 않는다.

---

## 1. 뭘 만드는가

200+ KRW 마켓을 읽되, **freshness filter 통과 후 최근 24h 거래대금 상위 100개**를 기본 활성 유니버스로 사용한다.
연구는 이 활성 유니버스에서 랭킹을 계산하고, 현재 라이브 프로브는 **Upbit 신호 → Bitget tradable 필터 → SHORT 5 실행 + WATCH LONG 5 참고** 계약을 사용한다.

gan_t와 다른 세 가지:
- 절대 수익률 예측 → **상대 랭킹** (베타 노이즈 상쇄됨)
- 22h 백테스트 루프 → **5분 IC 루프** (신호 확인이 먼저)
- CVAE/GAN 복잡도 → **XGBoost 먼저**, IC 확인 후 복잡도 올림

---

## 2. gan_t 실수 — 여기서 반복 금지

| 실수 | 방지 규칙 |
|------|----------|
| `fillna(0)` → RSI=0 불가능 값 | `ffill().bfill()` 전용. 첫 480행 드롭. 예외 없음. |
| Train ⊃ Test (720일 ⊃ 90일) | 처음부터 temporal holdout 20% 분리. 절대 섞지 않음. |
| 3h AC=0.025 (노이즈) | 호라이즌 설정 전 AC 검증. AC ≥ 0.08 이상에서만 시작. |
| 백테스트로 신호 검증 | IC 먼저. IC > 0.05 없으면 백테스트는 의미 없음. |
| 복잡한 모델 먼저 | 단순 팩터 IC 기준선 → 모델 순서. 무조건. |
| Screener > Model | 모델 도입 전 baseline(rule-based) IC 먼저 측정. |
| 98% Short 편향 | Long:Short 비율 매 run 확인. 한쪽 > 70% 면 임계값 재검토. |

---

## 3. 즉시 확정된 설계 결정

| 항목 | 결정 | 이유 |
|------|------|------|
| 예측 타겟 | **residual return** (coin_return - beta × BTC_return) | 베타 오염 제거. 절대 수익률 예측 시 BTC 방향에 종속됨. |
| 첫 모델 | **XGBoost** | feature importance로 팩터 기여 즉시 확인 가능. Ridge는 IC 확인 후 비교. |
| 리밸런싱 | **12h마다** (예측 호라이즌과 동일) | 예측창과 실행창 일치. |
| 활성 유니버스 | **freshness filter 후 최근 24h 거래대금 상위 100개** | 극저유동성 코인 환상을 줄이고 train/eval/live 계약을 일치시킴. |
| 팩터 정규화 | **크로스섹션 z-score** per timestamp, `\|z\| > 3` 클리핑 | 코인별 절대값 차이 제거. rank → z-score 순서. |
| 연구용 Long/Short | **각 20개 고정** | 기본 활성 유니버스 top 100 기준. |
| 라이브 프로브 | **SHORT 5 실행 + WATCH LONG 5** | Bitget USDT perp tradable 코인만 알림/CSV에 남김. |
| 베타 계산 | **168h rolling** (gan_t와 동일) | 크립토 레짐 빠름. 30일 window는 stale. |

---

## 4. 핵심 팩터 (시작 시 6개)

gan_t의 35개 피처 중 크로스섹션 랭킹에 즉시 유효한 것만:

| 팩터 | 정의 | 현재 해석 | 비고 |
|------|------|-----------|------|
| reversal_1h | -1 × 1h 수익률 | 직전 1h loser가 반등 | 기존 `momentum_1h` 부호/이름 정리 |
| reversal_4h | -1 × 4h 수익률 | 직전 4h loser가 반등 | 기존 `momentum_4h` 부호/이름 정리 |
| volatility_inv_24h | -1 × 실현 변동성 | 저변동성 우위 | 유지 |
| order_flow_bear | -1 × 6h 방향 가중 거래량 비율 | 최근 매도압이 강할수록 반등 | 기존 `order_flow_bull` 부호/이름 정리 |
| binance_lead_1h | Binance 1h return - Upbit 1h return | 글로벌 가격이 먼저 움직인 코인 추종 | 유지 |
| kimchi_inv | -1 × log(Upbit / Binance) | 김프가 낮은 코인 선호 | 유지 |

IC 기준선 확인 후 factor_size, factor_mom, factor_liq (gan_t에서 이미 계산됨) 추가.

---

## 5. IC 계산 — 표준 방법

```python
# 한 시점의 IC
from scipy.stats import spearmanr
ic_t = spearmanr(predicted_rank_t, actual_rank_t+12h)[0]

# NaN 코인: 해당 시점 랭킹에서 제외 (imputation 금지)
# 집계
mean_ic = ic_series.mean()
t_stat  = mean_ic / (ic_series.std() / sqrt(N_periods))
```

- **IC > 0.05** = 신호 존재 (백테스트 검토 가능)
- **IC > 0.10** = 실용적 수준 (배포 검토)
- **t-stat > 2.0** = 통계적 유의 (필수 조건)
- holdout: 마지막 20% 시간 구간. 마켓 단위 아님. 학습 중 절대 열지 않음.

---

## 6. 재활용 (gan_t → xsec_alpha, 복사 후 수정)

**그대로 쓸 것:**
- `data/collector.py` — Upbit API, 캔들 수집
- `data/database.py` — SQLite 읽기 (read-only bridge)
- `data/preprocessor.py` 中 `get_market_index()`, `get_crypto_factors()`, `calculate_technical_indicators()`
- `scripts/run_ops_job.py`, `scripts/ops_healthcheck.py`, `utils/run_lock.py`, `utils/telegram_bot.py`
- systemd 타이머 구조 (00:00, 12:00 UTC)

**수정할 것:**
- 시퀀싱 레이어 제거 (3D array → 2D point-in-time snapshot)
- 스케일러: per-market MinMax → per-timestamp 크로스섹션 z-score
- 타겟: `future_pct_change` → `residual_return` 랭킹

**가져오지 않을 것:**
- `models/`, `training/`, `inference/` 전체 (CVAE, GAN, MC-Dropout)

---

## 7. 우선순위 / 페이즈

### Phase 1 (3일): 데이터 + IC 기준선
- 파이프라인 동작 확인 (Upbit → 피처 → 랭킹)
- 단순 팩터 6개만으로 IC 측정
- **Gate: IC > 0.05 on ≥1 팩터**. 미달 시 데이터 버그 먼저.

### Phase 2 (5일): XGBoost + holdout 검증
- XGBoost 학습, feature importance 확인
- holdout IC 측정
- **Gate: IC > 0.10 AND t-stat > 2.0**

### Phase 3 (10일): 운영 연결
- systemd 타이머, 텔레그램, 대시보드
- **Gate: IC_live ≥ IC_holdout − 0.03**

### 롤백 기준
- Phase 1 종료 시 모든 팩터 IC < 0.02 → 프로젝트 중단, 데이터 파이프라인 감사
- Phase 3 중 IC < 0.05 연속 2회 → 포지션 동결, 원인 조사
- 연속 3회 → 전량 청산, 2주 운영 중단

---

## 8. 검증 규칙

**코드 변경 후 반드시:**
1. IC 측정 실행 (5분이면 됨 — 변명 없음)
2. 피처 NaN율 확인 (컬럼별 30% 초과 금지)
3. 예측값 범위 확인 (`mean ∈ [-0.05, +0.05]`, `std ∈ [0.01, 0.50]`)

**즉시 중단 조건:**
- RSI=0, volatility=0 발견 시 → 데이터 파이프라인 버그. 배포 전 수정.
- 예측 std < 0.005 → 제로 신호. 원인 찾기.
- Long/Short 각 10개 미만 → watch-only 모드로 출력.

**일일 운영 확인:**
- IC (rolling 7-day) 대시보드에서 확인
- `IC < 0.03` 이상 하락 시 텔레그램 알림

---

## 9. 런타임 진입점

| 용도 | 파일 |
|------|------|
| 메인 실행 (systemd) | `scripts/fetch_and_rank.py` |
| 설정 검증 | `scripts/validate_config.py` |
| 헬스체크 | `scripts/healthcheck.py` |
| 웹 대시보드 | `app.py` (Flask, :5555) |
| IC 측정 | `scripts/measure_ic.py` |

---

## 10. 성공 기준

- [ ] Phase 1: 단순 팩터 IC > 0.05 (t-stat > 1.5)
- [ ] Phase 2: XGBoost holdout IC > 0.10 (t-stat > 2.0)
- [ ] Phase 3: 매일 추천 CSV 생성, IC 대시보드 표시
- [ ] 베타 중립 포트폴리오 Sharpe > 0.5 (IC > 0.10 확인 후)
- [ ] 운영 안정성: 14일 연속 중단 없이 실행
