# xsec_alpha — Cross-Sectional Crypto Ranking System

gan_t(절대 수익률 예측)에서 실패한 교훈을 바탕으로,
**크로스섹션 상대 랭킹**으로 전환한 크립토 트레이딩 연구 프로젝트.

---

## 연구 일지

### 1. 출발점 — gan_t에서 배운 것

gan_t는 CVAE/GAN 기반 절대 수익률 예측 시스템이었다.
90일 백테스트 Sharpe 0.17 (랜덤과 구분 불가), 스크리너(55.8%) > 모델(53.1%).

**근본 문제:**
| 문제 | 증거 |
|------|------|
| 3h AC=0.025 (노이즈) | 예측 불가능한 timescale에서 학습 |
| KL=0 (posterior collapse) | CVAE 잠재공간 미사용 |
| fillna(0) | RSI=0, volatility=0 불가능 값 |
| train ⊃ test (100% overlap) | 모든 메트릭 무효 |
| confidence 상수 출력 | position sizing에 정보 없음 |
| 98% short + short 차단 | 실행 가능한 추천 거의 0 |

**핵심 교훈:** 절대 수익률 예측은 BTC 베타에 종속된다. 상대 랭킹으로 가야 베타 노이즈가 상쇄된다.

---

### 2. 설계 결정

| 항목 | gan_t | xsec_alpha |
|------|-------|------------|
| 예측 타겟 | 절대 수익률 | residual return (coin - beta × BTC) |
| 모델 | CVAE + GAN | XGBoost/LGBM/Ridge 앙상블 |
| 검증 | 22h 백테스트 루프 | IC 먼저, holdout, walk-forward 순서 |
| 정규화 | per-market MinMax | per-timestamp 크로스섹션 z-score |
| 결측치 | fillna(0) | ffill only, 첫 480행 drop |
| train/test | 100% overlap | temporal holdout 20% 분리 |

---

### 3. 유니버스 필터 — 현실적 신호 검증

IC를 전체 200+ 코인에서만 보면 극저유동성 코인에서만 나오는 환상적 신호를 잡을 수 있다.
그래서 **24h 거래대금 상위 100개**를 활성 유니버스로 고정하고, 학습/holdout/walk-forward/live 추천 5개 경로에 공통 적용했다.

- full universe → top100 → top50으로 줄여도 핵심 팩터 IC가 유지됨을 확인
- `config.Data.LIQUIDITY_TOP_N = 100`으로 한 곳에서 제어
- 공통 helper: `build_top_liquidity_universe_index()` in `data/features.py`

---

### 4. 팩터 설계 및 IC 측정

Upbit 200+ KRW 코인에서 8개 팩터를 계산하고, 활성 유니버스(top100)에서 IC를 측정했다.

**IC 결과 (120일, top100, 6h horizon):**
| 팩터 | Mean IC | t-stat | 해석 |
|------|---------|--------|------|
| kimchi_inv | +0.1663 | 높음 | 김프가 낮은 코인 선호 |
| binance_lead_1h | +0.1139 | 높음 | 글로벌 가격 선행 |
| volatility_inv_24h | +0.1045 | 높음 | 저변동성 우위 |
| reversal_1h | +0.0952 | 유의 | 1h 낙폭 반등 |
| reversal_4h | +0.0901 | 유의 | 4h 낙폭 반등 |
| order_flow_bear | +0.0615 | 유의 | 매도압 후 반등 |

- 전반/후반 half-split에서 전부 [STABLE]
- top50 유동성으로 줄여도 핵심 팩터 유지

**팩터 부호 정리:**
- 기존 `momentum_1h/4h` → `reversal_1h/4h` (부호 반전, 양의 IC 방향)
- 기존 `order_flow_bull` → `order_flow_bear` (부호 반전)
- 모델 wrapper에 legacy 호환 유지 (구 pickle도 동작)

---

### 5. Horizon 선택 — 1h vs 6h

`ic_decay.py`로 1h~48h IC를 측정한 뒤, 1h와 6h를 후보로 선정.

**1-bar lag falsification:**

| 지표 | 1h lag=0 | 1h lag=1 | 6h lag=0 | 6h lag=1 |
|------|----------|----------|----------|----------|
| Holdout IC | +0.307 | +0.081 | +0.197 | +0.109 |
| WF IC | +0.299 | +0.082 | +0.189 | +0.093 |
| WF Sharpe(net) | +98.3 | +14.3 | +21.2 | +10.8 |
| Holdout gate | PASS | **FAIL** | PASS | **PASS** |

**판정:** 1h는 lag 1바에서 IC가 74% 꺾임 → 마이크로스트럭처 의존으로 판단, **기각**.
6h는 lag=1에서도 holdout gate 통과 → **6h 유지**.

---

### 6. 포트폴리오 구조 실험

6h + lag=1 + buffer=10 + short cost 10bps 기준으로 구조별 비교:

**Holdout (net spread / 6h period):**
| 구조 | Net spread | Top-bottom gap z |
|------|-----------|-----------------|
| L/S 20/20 | +0.0034 | +2.19 |
| L/S 10/10 | +0.0052 | — |
| L/S 5/5 | +0.0081 | +4.16 |
| Short-only 5 | +0.0069 | — |

**Walk-forward (180일, short cost 10bps):**
| 구조 | Sharpe(net) | Total return | Win rate |
|------|-------------|-------------|----------|
| L/S 20/20, buffer 10 | +7.78 | +89.7% | — |
| L/S 5/5, buffer 10 | +9.95 | +789.6% | — |
| Short-only 5, buffer 10 | +10.07 | +910.4% | — |

> 주의: 이 숫자들은 액면 그대로 믿으면 안 된다. 특히 short-only 5의 +910%는
> 소수 극단 이벤트 의존 가능성이 있고, Sharpe 10은 현실에서 나오지 않는 수준이다.

---

### 7. Short 비용 민감도

| short cost | Sharpe(net) | Total return | 손실 윈도우 |
|-----------|-------------|-------------|------------|
| 0 bps | +11.18 | +156% | 0/6 |
| 10 bps | +7.25 | +83% | 2/6 |
| 20 bps | +3.33 | +31% | — |

- 10bps만 넣어도 6개 윈도우 중 2개가 음수로 전환
- 수익이 사실상 전부 short leg에서 발생 (avg long return = -0.04%)

---

### 8. 종목 집중도 감사

Short-only 5 walk-forward에서 종목별 기여도를 분석:

- Short 쪽 distinct name: **136개** (소수 독식 아님)
- Top1 net 기여 비중: 4.9%, Top5: 19.2%, Top10: 32.3%
- 20회 이상 반복 선택: 26개, 40회 이상: 3개
- 최악 단일 포지션: 한 캔들 +33.4% 역행 → -6.7% net contribution

**결론:** "두세 개 코인 독식"은 아니지만, tail-heavy short basket.

---

### 9. Regime 분석

BTC 7d return × 7d vol로 4구간 분류:

| Regime | n | IC | Gross | Net (10bps) | Hit |
|--------|---|-----|-------|-------------|-----|
| bear_highvol | 104 | +0.108 | +0.0045 | +0.0028 | 67.3% |
| bear_lowvol | 107 | +0.091 | +0.0037 | +0.0020 | 52.3% |
| bull_highvol | 68 | +0.085 | +0.0031 | +0.0014 | 60.3% |
| bull_lowvol | 57 | +0.077 | +0.0021 | +0.0004 | 57.9% |

- bull_lowvol이 가장 취약 (short cost 올리면 음수)
- 180일 데이터에 강한 추세장(|BTC 30d| >= 30%)이 거의 없음 (0.6%)

---

### 10. Long 실행 가능성 검증 — 첫 시도와 실패

같은 계약(6h, lag=1, buffer=10)에서 long-only 5를 테스트:

| 지표 | Long-only 5 |
|------|------------|
| Holdout IC | +0.1087 |
| Holdout net spread | +0.0013 |
| WF Total return | **-18.72%** |
| WF Sharpe(net) | **-0.78** |
| WF Max drawdown | -33.84% |
| 음수 윈도우 | 3/6 |

**판정:** IC는 있지만 long-only로 수익 전환 실패. 현재 모델은 **short 신호 전문**.

---

### 10-1. Long 전용 모델 연구 — v1 (momentum/trend 접근)

short 모델이 short 전문이므로, **long은 완전히 별도 모델로 분리**하기로 결정.

**v1 설계 가설:** "short 모델이 역추세(reversal)라면, long 모델은 순추세(momentum/trend)로 가야 한다."

v1 팩터 세트:
- momentum_12h, momentum_24h, breakout_24h, volume_surge_6h
- bullish_order_flow, volatility_inv_24h, binance_lead_1h

BTC 레짐 게이트 추가: bull(BTC 7d > 0%, 30d > -10%)일 때만 long 모델 활성.

**v1 holdout 결과:**
| 지표 | 값 |
|------|-----|
| Holdout IC | +0.1766 |
| t-stat | 5.99 |
| Hit rate | 72.7% |
| pred_std | 0.0027 (경고: 근상수 예측) |

겉으로는 좋아 보였다. **하지만 13개 에이전트 다각도 감사에서 심각한 문제 발견:**

---

### 10-2. 다각도 감사 — v1의 근본 결함 발견

**20+ 에이전트, 4라운드 검증** 결과:

**1) 모멘텀 IC가 전 구간에서 음수:**
| 팩터 | 6h | 12h | 24h | 48h | 72h |
|------|-----|------|------|------|------|
| momentum_12h | -0.080 | -0.074 | -0.082 | -0.070 | -0.059 |
| momentum_24h | -0.071 | -0.068 | -0.065 | -0.055 | — |
| relative_strength_6h | -0.093 | -0.073 | -0.068 | -0.058 | -0.048 |
| vol_weighted_mom_6h | -0.085 | -0.068 | -0.062 | -0.052 | -0.039 |

**어떤 호라이즌에서도 모멘텀 IC가 양수로 전환되지 않았다.** Bull 레짐에서도 크립토는 단기 mean-reversion이 지배적.

**2) 모델이 자동으로 부호를 뒤집고 있었음:**
- Ridge가 음수 IC 팩터에 음수 계수를 부여 → 사실상 reversal 신호로 사용
- "long 모델"이라고 불렀지만 실체는 "bull 레짐 한정 reversal 모델"

**3) Long leg는 사실상 작동 안 함:**
- Backtest long hit rate: **43.6%** (동전 던지기보다 나쁨)
- 수익의 **90.5%가 short leg**에서 발생
- Long avg return: +0.0008/period (거의 제로)

**4) 콜리니어리티:**
- momentum_12h × momentum_24h: r=0.633
- momentum_12h × breakout_24h: r=0.636
- 3개 팩터가 사실상 같은 정보를 중복 표현

**5) pred_std 문제:**
- pred_std=0.0027 (top5-bottom5 gap이 1.03%에 불과)
- 원인은 정규화가 아니라 **피처 예측력 부재** (R²=1%)
- alpha를 1.0→0.01로 바꿔도 계수가 동일

**핵심 결론:**
> "long이 안 되는 게 아니라, short 철학의 변형판을 long에 얹은 것이 문제였다."

---

### 10-3. 대안 팩터 탐색 — "조용한 코인이 터진다"

10개 대안 팩터를 bull 레짐 + top100에서 측정한 결과, **range_contraction_12h**가 발견됐다.

**IC > +0.05 달성 팩터 (long 시그널로 유효):**
| 팩터 | 6h IC | 12h IC | 24h IC | 해석 |
|------|-------|--------|--------|------|
| **range_contraction_12h** | +0.109 | +0.137 | **+0.179** | 고저폭 압축 → 폭발 전 횡보 |
| volatility_inv_24h | +0.112 | +0.144 | +0.168 | 저변동성 프리미엄 |
| binance_lead_1h | +0.117 | +0.088 | +0.064 | 글로벌 선행 (단기) |
| reversal_1h | +0.102 | +0.078 | +0.062 | 단기 딥 매수 |
| reversal_4h | +0.094 | +0.073 | +0.062 | 중기 딥 매수 |

**IC < 0 탈락 팩터:**
| 팩터 | 결과 |
|------|------|
| dip_from_7d_high | 추가 하락 예측 |
| upside_capture_24h | 과열 신호 |
| recovery_speed_6h | 이미 반등 완료 |

**range_contraction_12h 안정성 검증 (4분기 split):**
| 구간 | 6h IC | 12h IC | 24h IC |
|------|-------|--------|--------|
| Q1 | +0.097 | +0.113 | +0.155 |
| Q2 | +0.102 | +0.122 | +0.193 |
| Q3 | +0.095 | +0.136 | +0.170 |
| Q4 | +0.141 | +0.178 | +0.198 |

4개 분기 전부 양수, 모든 호라이즌에서. Rolling 7d 최솟값도 양수. 운이 아님.

**콜리니어리티 체크:**
- range_contraction × vol_inv: r=**0.785** → 둘 다 "조용한 코인"을 측정, 하나만 남겨야 함
- range_contraction IC가 더 높으므로 채택, vol_inv 제거
- 나머지 쌍(reversal_1h × reversal_4h = 0.411, reversal_1h × binance_lead = 0.414)은 허용 범위

---

### 10-4. Long v3 — compression + reversal 모델

**설계 원칙:**
- 모멘텀/트렌드 팩터 전부 제거 (전 호라이즌 음수 IC)
- range_contraction이 핵심 (IC +0.179, 4분기 안정)
- reversal은 bull 레짐에서도 양수 IC → 유지
- 호라이즌 12h (range_contraction이 12h에서 더 강하고, 24h는 holdout 시점 부족)

**v3 팩터 세트 (4개):**
| 팩터 | IC (12h, bull) | 역할 |
|------|---------------|------|
| range_contraction_12h | +0.137 | 고저폭 압축 → 다음 폭발 |
| reversal_1h | +0.078 | 단기 딥 매수 |
| reversal_4h | +0.073 | 중기 딥 매수 |
| binance_lead_1h | +0.088 | 글로벌 선행 |

**4모델 비교 (holdout, 12h horizon):**
| 모델 | IC | t-stat | Net spread | Pred std |
|------|-----|--------|------------|----------|
| Ridge | +0.1835 | 3.16 | +0.0198 | 0.0031 |
| XGBoost | +0.1677 | 3.61 | +0.0115 | 0.0059 |
| LightGBM | +0.1787 | 4.56 | +0.0195 | 0.0050 |
| **Ensemble** | **+0.2158** | **4.24** | **+0.0238** | 0.0034 |

**Ensemble(Ridge+LightGBM) 채택.**

**v3 vs v1 비교:**
| 지표 | v1 (momentum) | v3 (compression) |
|------|--------------|------------------|
| 팩터 설계 | 순추세 (IC 전부 음수) | 압축+반전 (IC 전부 양수) |
| Ridge 계수 부호 | 혼재 (모델이 반전시킴) | **전부 양수** (설계=결과 일치) |
| Long hit rate | 43.6% | **52.6%** |
| Long vs Random | 미측정 | **73.8%에서 초과** (p<0.0001) |
| Holdout IC | +0.1766 | **+0.2158** |

**Long vs Random 검증 (122기간):**
- 모델 top5 vs 랜덤 5: 초과수익 +0.57%/period
- 73.8%의 기간에서 모델이 랜덤 초과
- t-stat=7.34, p<0.0001
- Long 모델이 랜덤보다 유의미하게 나음

---

### 10-5. Long 모델 현재 상태 — watch 유지

**확인된 것:**
- v3 설계와 결과가 일치한다 (계수 전부 양수, IC 전부 양수)
- Bull 레짐에서 랜덤 대비 유의미한 long 선택 능력
- Short 모델과 독립적인 12h cadence로 운영 가능

**아직 부족한 것:**
- Holdout 11~22 시점 (6일) — 60+ 필요
- pred_std=0.0034 — 피처 예측력 한계 (R²=1%)
- Long hit rate 52.6% — 유의미하지만 strong은 아님
- Long contribution 18% — 수익 대부분은 여전히 short leg

**승격 조건 (config에 명시):**
- holdout >= 60 timestamps
- pred_std >= 0.01
- long-only IC > 0

**EXECUTION_MODE = "watch"** — 스코어링하고 CSV/텔레그램에 WATCH_LONG으로 기록하되, 실제 매매 신호로는 올리지 않음. 조건 충족 시 `config.py`에서 `"long_only"`로 전환.

**핵심 교훈:**
> "크립토에서 bull 레짐이라도 단기(6h~72h) 모멘텀은 mean-reversion에 밀린다.
> Long 신호는 '올라갈 코인'이 아니라 '조용해진 뒤 터질 코인'에서 나온다."

---

### 11. Tradability 검증

Bitget USDT perp 공개 API로 실제 숏 가능 여부 확인:

- Bitget 542개 USDT perp 중 WF short 후보 136개와 대조
- 이름 기준 커버리지: **93.4%** (127/136)
- Net contribution 기준 커버리지: **90.0%**
- 비상장: ARDR, TFUEL, QKC, STRAX 등 소수

---

### 12. 연구 루프를 닫는 결정

여기까지 오는 동안 반복된 패턴이 있었다:

1. IC 측정 → "좋다, 하지만 유니버스 확인하자"
2. 유니버스 확인 → "좋다, 하지만 horizon 비교하자"
3. Horizon 비교 → "좋다, 하지만 lag 테스트하자"
4. Lag 통과 → "좋다, 하지만 short cost 넣자"
5. Short cost 통과 → "좋다, 하지만 포지션 수 줄이자"
6. 포지션 줄임 → "좋다, 하지만 종목 집중도 보자"
7. 집중도 괜찮음 → "좋다, 하지만 tradability..."

**인식:** 이건 끝이 없다. 검증을 더 하면 할수록 새로운 "하지만"이 나오고, 실제로는 아무것도 결정하지 않고 있다. 백테스트로는 절대 답할 수 없는 것들(실제 슬리피지, 체결률, 펀딩비 변동)이 있고, 그건 라이브에서만 알 수 있다.

**결정:** 연구 루프를 닫고, tradability만 확인한 뒤 소규모 라이브 프로브로 검증 모드를 전환한다. 더 많은 백테스트보다 작은 라이브 프로브가 더 정보가 크다.

---

### 13. 현재 결론

**확인된 것:**
- 크립토 크로스섹션에서 떨어질 코인을 골라내는 신호가 존재한다 (6h, lag=1 IC +0.09)
- 핵심 팩터: kimchi_inv, binance_lead, volatility_inv, reversal 계열
- top100 유니버스, 전반/후반 split에서 안정적
- Bitget에서 실제 숏 가능

**아직 모르는 것:**
- 실전 슬리피지, 체결률, 펀딩비 변동에서 살아남는지
- 강한 상승장에서 short-only가 버티는지
- 실제 net PnL이 양수인지

**현재 판단:**
> "신호는 찾았다. 하지만 그걸로 수익을 내는 포트폴리오 구조는 아직 증명 안 됐다."

---

### 14. 라이브 프로브 계약 (2026-04-14 갱신)

| 항목 | Short 모델 | Long 모델 |
|------|-----------|-----------|
| 분석 데이터 | Upbit KRW (200+ 코인) | 동일 |
| 활성 유니버스 | 24h 거래대금 top 100 | 동일 |
| 실행 대상 | Bitget USDT perp tradable | 동일 |
| Horizon | **6h** | **12h** |
| 모델 | Ensemble (Ridge + LightGBM) | Ensemble (Ridge + LightGBM) |
| 팩터 | reversal, volatility_inv, kimchi_inv, binance_lead, order_flow_bear | range_contraction, reversal_1h/4h, binance_lead |
| 포지션 | **SHORT 5 (실행)** | **WATCH LONG 5 (관찰)** |
| 리밸런싱 | 매 run (6h마다) | 12h마다 (11:00/23:00 UTC) |
| 오프사이클 동작 | 새로 스코어링 | 이전 포지션 carry |
| 레짐 게이트 | 없음 (전 레짐 실행) | BTC 7d > 0% AND 30d > -10% |
| EXECUTION_MODE | short_only | **watch** (승격 조건 충족 전까지) |

**관측 인프라:**
- `output/recommendation_ledger.csv` — 만기 포지션 실현 손익 자동 누적
- `output/ic_history.json` — short 6h IC 추적
- `output/ic_history_long.json` — long 12h IC 추적 (bull 레짐만)
- 텔레그램: 이전 추천 성적표 + SHORT 5 실행 + WATCH LONG 5 참고

**Long 승격 조건 (config.py에 명시):**
- holdout timestamps >= 60
- pred_std >= 0.01
- long-only IC > 0
- 충족 시 `LongModel.EXECUTION_MODE = "long_only"`로 전환

---

### 15. 다음 단계

- [ ] 2~4주 라이브 프로브 실행 및 데이터 수집
- [x] Long 전용 모델 연구 → v3 완성 (compression+reversal, 12h, Ensemble)
- [ ] Long 승격 판정: 60+ holdout timestamps 축적 후 재평가
- [ ] app.py 대시보드에 ledger/next_rebalance/refresh_reason 노출
- [ ] Short 모델 재저장 (Ridge feature names 경고 제거)
- [ ] 360일+ 데이터로 추세장 포함 재검증
- [ ] 프로브 종료 후: 계속/조정/중단 결정

---

## 시스템 구조

```
Upbit API ──→ SQLite ──→ 팩터 계산 (크로스섹션 z-score)
                              │
Binance API ─────────────────→│
                              ↓
                 ┌─────────────┴─────────────┐
                 │                           │
          Short 모델 (6h)          Long 모델 (12h)
          reversal, vol_inv,       range_contraction,
          kimchi, binance_lead     reversal, binance_lead
          order_flow_bear          + BTC 레짐 게이트
                 │                           │
          Bitget tradable 필터     Bitget tradable 필터
                 │                           │
          SHORT 5 (실행)          WATCH LONG 5 (관찰)
                 │                           │
                 └─────────────┬─────────────┘
                               │
                    CSV + 텔레그램 + 대시보드
                               │
                    실현 성과 ledger (자동 누적)
```

**스케줄:** systemd timer, 6h 간격
- Short: 매 run 리밸런스
- Long: 12h마다 리밸런스 (오프사이클은 carry)

---

## 2부 — 운영 리얼리티 체크 (2026-04-25)

> 여기부터는 **위 연구가 실제로 돌아간 뒤 드러난 문제들과 그 해결**의 기록이다.
> 위 §1~15는 "연구 → 설계" 단계, 아래는 "운영 → 재설계" 단계.

---

### 16. 원장이 드러낸 진실

2-4주 라이브 프로브가 실제로 돌기 시작한 뒤, **추천-원장(`output/recommendation_ledger.csv`)** 을 감사하면서 세 가지 구조적 문제가 발견됐다.

**16-1. 텔레그램 WATCH_LONG 부호 버그**

`utils/telegram.py::format_performance`가 `side_name == "LONG"` 을 정확 문자열 비교로 처리하고 있어서 `"WATCH_LONG"`은 `else` 분기로 빠져 **SHORT 공식** `(entry−exit)/entry` 으로 계산됐다. 결과: 모든 WATCH LONG 행의 손익 부호가 반대로 표시됨. 원장은 맞게 기록했지만 텔레그램만 거짓말을 하고 있었다.

→ `utils.telegram.realized_return(side, entry, exit)` 단일 헬퍼 도입. `verify_telegram.py` 가 원장↔텔레그램 부호 일관성을 자동 검증 (옛 버그를 regression test로 재현 가능).

**16-2. LONG 82.7% 적중률 = survivorship bias**

처음엔 2σ+ 적중률 82.7%로 보였지만 이는 calibration 윈도우가 학습 데이터와 겹쳐 생긴 인위적 수치였다. 진짜 holdout에서는 **dir_hit 58%, IC 0.13** 수준. bull-regime-only 학습 + 매크로 피처가 "bull-conditional prediction"을 학습해서 bull 구간에서만 과적합한 결과.

**16-3. 90% 방향 불일치 (SHORT vs LONG)**

같은 코인에 대해 SHORT 6h 모델은 91/100 ↓, LONG 12h 모델은 99/100 ↑. 방향 합의 10%. Rank correlation은 +0.55이므로 "**같은 랭킹 순서**를 보고 있지만 **baseline(어디가 0이냐)**이 다름". 원인: SHORT는 전 regime 평균 ≈ 약간 음수, LONG은 bull-only 학습으로 평균 ≈ 양수.

---

### 17. 운영 자동화 인프라

버그가 몇 주간 모른 채 돌았다는 것 자체가 **감시/게이트가 운영에 없었다**는 증거. CLAUDE.md §7의 롤백 규칙은 문서로만 존재했고 자동 집행되지 않았다.

새로 만든 것:

| 파일 | 역할 |
|---|---|
| `scripts/health_snapshot.py` | 세션 킥오프 건강 스냅샷 — 데이터 신선도, IC 상태, 원장 실현 성적, 현재 픽 요약 |
| `utils/preflight.py` | fetch_and_rank 실행 전 게이트 — 데이터 신선도 ≤90min, NaN율 ≤20%, universe churn ≤30% |
| `utils/ic_gate.py` | CLAUDE.md §7 자동 집행 — OK/WARN/FREEZE/LIQUIDATE 상태 머신, 사이드별 추적 |
| `utils/drift_detector.py` | 팩터별 IC 24h vs 7d MA 비교, SIGN_FLIP / 50%+ drop 감지 |
| `utils/enrich.py` | 두 모델 consensus ⚡ + 원장 기반 per-coin 신뢰도 ⭐⚠ |
| `scripts/verify_telegram.py` | 원장↔텔레그램 부호/크기 일관성 regression test |
| `scripts/wf_holdout_harness.py` | 일일 walk-forward holdout IC 측정 + wf_history.json 누적 |

세션 킥오프 의례도 명문화 (CLAUDE.md §0): 모든 편집 전 `python scripts/health_snapshot.py` 먼저.

---

### 18. per-coin 확률 출력 설계

이전 텔레그램은 top-5 basket만 보냈고 "잔차 -0.5%" 같은 숫자가 무슨 의미인지 유저가 알 수 없었다. 전면 재설계:

**σ-bucket calibration (`output/calibration_sigma.json`):**
과거 전체 유니버스 7,600 예측을 스캔해서 `|σ| = |score|/batch_std` 구간별로:
- 방향 적중률 `hit_rate`
- 기대 수익률 `mean_signed_return_pct` (sign(score) × realized_return 평균)
- 평균 절대 수익률 `mean_abs_return_pct`
- 표준편차

**per-coin 출력 필드 (`utils/magnitude.py::predict_one`):**
```
{direction, direction_prob, expected_pct,
 ci_95_low, ci_95_high, sigma, tag (🔥/✅/▫/·),
 coin_vol_pct, position_size_pct}
```

CI는 **코인별 실현 변동성 × 1.96** 으로 계산 → tier가 같아도 코인마다 위험폭이 다름이 시각화됨. Position size는 σ-tier 기반 제안치 (🔥 3% / ✅ 2% / ▫ 1%).

Telegram 한 행:
```
🔥 KAITO ↓  65% 기대-1.29%  변±2.6%  [-6.4, +3.9]  💰 3.0%  ⚡⭐
```
↑ 방향 확률, 기대 %, 코인 자체 변동폭, 95% CI, 제안 사이즈, 두 모델 합의(⚡), 과거 적중 이력(⭐).

---

### 19. 20명 트레이더 토론 — 3 라운드

운영 문제 해결 과정에서 **20명 스페셜리스트(4 배치 × 5명)** 의 병렬 토론을 3회 수행. 우리 프로젝트만의 방식: 각 토론은 실제 코드 + 원장 데이터 접근 권한을 갖고 진행.

**라운드 1: "현재 시스템의 가장 강력한 형태"**
- Batch I 전통 퀀트 / II ML / III 크립토 / IV 실전 트레이더
- 결론: Tier 1 (확률 출력) → Tier 2 (ops) → Tier 3 (피처) → Tier 4 (재학습) 점진 빌드

**라운드 2: "주기·재학습·리셋 정책"**
- 14/20 F1 Unified Regime 선택 (bull 필터 제거)
- 주간 soft retrain + 월간 hard reset + drift 트리거
- Promotion gate: new_IC ≥ old_IC − 0.015 AND ≥ 0.040

**라운드 3: "최종 아키텍처 — 2 모델 유지 vs 통합 vs multi-horizon"**
- 13/20 F1 (피처 파이프라인 통합) — 구현 30분, 리스크 최소
- 5/20 F3 (meta-learner) — 나중에
- 3/20 F2 (multi-horizon single model) — 이 단계에선 과도

---

### 20. F1 통합 아키텍처 (최종 선택)

한 개의 통합 피처 라이브러리로 두 호라이즌을 학습한다.

**`data.features.compute_unified_factors()` — 10개 피처:**
```
reversal_1h, reversal_4h, volatility_inv_24h, order_flow_bear,
range_contraction_12h, binance_lead_1h, kimchi_inv, kimchi_zscore_24h,
dow_bull, hour_vol
```

**두 모델, 같은 입력:**
| 파일 | 호라이즌 | 타겟 | Regime 필터 |
|---|---|---|---|
| `models/xsec_6h.pkl` | 6h | absolute coin_return | 전 regime |
| `models/xsec_12h.pkl` | 12h | absolute coin_return | 전 regime |

**개선 지표:**
| 측정 | 이전 (분리 피처) | F1 (통합) |
|---|---|---|
| 방향 합의 (SHORT↔LONG) | 10-91% | **92%** |
| Rank correlation | +0.55~0.61 | **+0.908** |
| 6h 2σ+ 적중 | 63.7% | **64.6%** |
| 12h 2σ+ 적중 | 64.3% | **70.6%** |
| 6h 2σ+ E[signed] | +1.23% | **+1.29%** |
| 12h 2σ+ E[signed] | +0.37% (bull bias) | **+1.85%** |

Bull regime 필터를 없애고도 12h 성능이 오히려 좋아졌다. 이전의 82.7%는 survivorship bias였음이 확인됨.

---

### 21. 재학습·운영 자동화

**`scripts/retrain_pipeline.py` (주 1회, systemd):**
1. Pre-flight — 데이터 freshness, lock, 디스크
2. 두 모델 후보 학습 (`xsec_6h_candidate.pkl`, `xsec_12h_candidate.pkl`)
3. 홀드아웃 IC 측정 (신/구 모델 양쪽)
4. **Promotion gate:**
   - `new_ic ≥ old_ic − 0.015` AND `new_ic ≥ 0.040`
   - 통과: archive 백업 → 배포 → calibration 재생성
   - 실패: 후보 삭제, 이전 모델 유지, 로그 기록
5. 결과 `output/retrain_history.json` 누적

**Systemd 타이머:**
| 유닛 | 주기 | 역할 |
|---|---|---|
| `xsec-alpha.timer` | 매 6h (05/11/17/23 UTC) | 예측 실행 + 텔레그램 |
| `xsec-measure.timer` | 매일 00:00 UTC | 홀드아웃 IC + drift 체크 |
| `xsec-retrain.timer` | 매주 일 20:00 UTC | F1 retrain_pipeline |

`deploy/install_f1.sh` 한 번 실행으로 전체 systemd 배포.

---

### 최종 시스템 다이어그램

```
                    ┌─────────────────────────────┐
                    │  Upbit + Binance 1h OHLCV   │
                    └─────────────┬───────────────┘
                                  ↓
                 ┌────────────────┴────────────────┐
                 │  compute_unified_factors()       │
                 │  10 features × 100 coins × T    │
                 └────────────────┬────────────────┘
                                  ↓ cross-section z-score
                 ┌────────────────┴────────────────┐
                 ↓                                 ↓
        ┌────────────────┐                ┌────────────────┐
        │ xsec_6h.pkl    │                │ xsec_12h.pkl   │
        │ (Ridge+LGBM)   │                │ (Ridge+LGBM)   │
        │ 6h horizon     │                │ 12h horizon    │
        └────────┬───────┘                └────────┬───────┘
                 │          (92% 합의, rc 0.91)     │
                 └─────────────────┬───────────────┘
                                   ↓
                 ┌─────────────────┴──────────────────┐
                 │  σ-bucket calibration              │
                 │  (direction_prob, expected_pct,    │
                 │   ci_95, position_size)            │
                 └─────────────────┬──────────────────┘
                                   ↓
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
       ┌──────────┐         ┌──────────┐         ┌──────────┐
       │ enrich   │         │ preflight│         │ ic_gate  │
       │ ⚡⭐⚠ 태그 │         │ NaN/fresh│         │ §7 게이트 │
       └────┬─────┘         └────┬─────┘         └────┬─────┘
            └────────────────────┼────────────────────┘
                                 ↓
                    ┌────────────┴────────────┐
                    │   텔레그램 + CSV 저장    │
                    │  + recommendation_ledger │
                    └─────────────────────────┘

    ⏱ Automation:
    ── 매 6h (예측)        : xsec-alpha.timer
    ── 매일 00:00 UTC (IC) : xsec-measure.timer
    ── 주 1회 일 20:00 UTC : xsec-retrain.timer
                            └→ retrain_pipeline.py (promotion gate)
                                 └→ pass → archive old + deploy new + rebuild calibration
                                 └→ fail → keep old + log
```

---

## 라이선스

개인 연구용. 투자 조언이 아님.
